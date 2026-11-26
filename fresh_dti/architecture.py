"""Standalone DTI architecture with no dataset, checkpoint or path dependencies."""

import math

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GINConv, HeteroConv


CONFIG = {
    "drug_input_dim": 2048,
    "protein_input_dim": 2560,
    "hidden_dim": 512,
    "view_depths": [2, 4],
    "gat_heads": 4,
    "dropout": 0.3,
    "pbcnet_feature_dim": 96,
    "global_factor_dim": 32,
    "gate_inputs": "global_factors_only",
}


def drug_relation(include_center):
    hidden = CONFIG["hidden_dim"]
    return GINConv(
        nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ELU(), nn.Linear(hidden, hidden)),
        eps=0.0 if include_center else -1.0,
        train_eps=include_center,
    )


class RelationView(nn.Module):
    def __init__(self, depth):
        super().__init__()
        hidden, heads = CONFIG["hidden_dim"], CONFIG["gat_heads"]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(CONFIG["dropout"])
        for index in range(depth):
            final = index == depth - 1
            out_channels, layer_heads = (hidden, 1) if final else (hidden // heads, heads)
            relations = {
                ("drug", "similar_to", "drug"): drug_relation(True),
                ("protein", "interacted_by", "drug"): drug_relation(False),
                ("drug", "interacts_with", "protein"): GATv2Conv(
                    (hidden, hidden), out_channels, heads=layer_heads, concat=not final, add_self_loops=False,
                ),
                ("protein", "similar_to", "protein"): GATv2Conv(
                    hidden, out_channels, heads=layer_heads, concat=not final, add_self_loops=True,
                ),
            }
            self.convs.append(HeteroConv(relations, aggr="mean"))
            if not final:
                self.norms.append(nn.ModuleDict({name: nn.LayerNorm(hidden) for name in ("drug", "protein")}))

    def forward(self, x_dict, edge_index_dict):
        features = x_dict
        for index, layer in enumerate(self.convs):
            features = layer(features, edge_index_dict)
            if index < len(self.norms):
                features = {
                    name: self.dropout(self.activation(self.norms[index][name](values)))
                    for name, values in features.items()
                }
        return features


class EntityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = CONFIG["hidden_dim"]
        self.drug_proj = nn.Linear(CONFIG["drug_input_dim"], hidden)
        self.prot_proj = nn.Linear(CONFIG["protein_input_dim"], hidden)
        self.view1_gnn = RelationView(CONFIG["view_depths"][0])
        self.view2_gnn = RelationView(CONFIG["view_depths"][1])
        self.view_logits = nn.Parameter(torch.zeros(2))

    def forward(self, x_dict, edge_index_dict):
        projected = {"drug": self.drug_proj(x_dict["drug"]), "protein": self.prot_proj(x_dict["protein"])}
        first = self.view1_gnn(projected, edge_index_dict)
        second = self.view2_gnn(projected, edge_index_dict)
        weights = self.view_logits.softmax(dim=0)
        merged = {name: weights[0] * first[name] + weights[1] * second[name] for name in projected}
        return first, second, merged


class ScoringHead(nn.Module):
    def __init__(self):
        super().__init__()
        hidden, feature, width = CONFIG["hidden_dim"], CONFIG["pbcnet_feature_dim"], CONFIG["global_factor_dim"]
        self.local_readout = nn.Sequential(nn.LayerNorm(feature), nn.Linear(feature, width), nn.SiLU(), nn.Linear(width, 1))
        self.drug_global = nn.Linear(hidden, width, bias=False)
        self.protein_global = nn.Linear(hidden, width, bias=False)
        self.gate = nn.Sequential(nn.LayerNorm(3 * width), nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def score(self, embeddings, pairs, pbcnet_features):
        if pairs.ndim != 2 or pairs.shape[0] != 2 or pairs.dtype != torch.long:
            raise ValueError("pairs must be int64 compound-target indices with shape [2, B]")
        if pbcnet_features.shape != (pairs.shape[1], CONFIG["pbcnet_feature_dim"]):
            raise ValueError("Each explicit pair needs a corresponding 96-dimensional PBCNet feature row")
        if not torch.isfinite(pbcnet_features).all():
            raise ValueError("PBCNet features must be finite")
        drug = self.drug_global(embeddings["drug"][pairs[0]])
        protein = self.protein_global(embeddings["protein"][pairs[1]])
        product = drug * protein
        local = self.local_readout(pbcnet_features).squeeze(-1)
        global_score = product.sum(dim=-1) / math.sqrt(CONFIG["global_factor_dim"])
        gate = self.gate(torch.cat((drug, protein, product), dim=-1)).squeeze(-1).sigmoid()
        return gate * local + (1 - gate) * global_score, local, global_score, gate


class FreshDTI(nn.Module):
    """All task parameters start fresh and are trainable, including protein GAT."""

    def __init__(self):
        super().__init__()
        self.encoder = EntityEncoder()
        self.head = ScoringHead()

    def forward(self, x_dict, edge_index_dict, pairs, pbcnet_features):
        _, _, embeddings = self.encoder(x_dict, edge_index_dict)
        return self.head.score(embeddings, pairs, pbcnet_features)


def new_model(seed):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FreshDTI()
    model.requires_grad_(True)
    return model
