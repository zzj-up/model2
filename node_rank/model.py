"""Two/four-layer geometric attention followed by global + atom-pair BAN readout.

This is a task-specific implementation of low-rank bilinear attention, not a
copy of the official BAN/DrugBAN model or PBCNet's original affinity head.
Frozen ESM2 and PBCNet node embeddings are supplied by the feature pipeline.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import softmax

from .graph import CONTACT_RELATION, RELATIONS, build_graph


ARCHITECTURE = "esm_pbc_node_dualview_geometric_ban_rank_v1"
CONFIG = {
    "sequence_dim": 2560,
    "atom_dim": 96,
    "hidden_dim": 128,
    "heads": 4,
    "view_depths": [2, 4],
    "dropout": 0.1,
    "radial_bins": 16,
    "spatial_cutoff": 4.5,
    "contact_cutoff": 5.0,
    "contact_logit_boost": 1.0,
    "ban_glimpses": 4,
    "ban_rank": 32,
}


class RadialBasis(nn.Module):
    def __init__(self, bins, cutoff):
        super().__init__()
        self.register_buffer("centres", torch.linspace(0.0, cutoff, bins))
        self.width = cutoff / (bins - 1)

    def forward(self, distance):
        return torch.exp(-0.5 * ((distance[:, None] - self.centres) / self.width).square())


class GeometryAttention(nn.Module):
    """Compete across all incoming relations before aggregation.

    A positive contact prior therefore changes contact-vs-background attention,
    instead of cancelling inside a separately normalized contact relation.
    The prior is learned, and content/distance terms can outweigh it.
    """

    def __init__(self, config):
        super().__init__()
        width, heads = config["hidden_dim"], config["heads"]
        self.heads, self.head_width = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.relation = nn.Embedding(len(RELATIONS), width)
        self.radial = RadialBasis(config["radial_bins"], max(config["contact_cutoff"], config["spatial_cutoff"]))
        self.distance_embedding = nn.Linear(config["radial_bins"], width, bias=False)
        self.edge_bias = nn.Linear(width, heads, bias=False)
        self.edge_value = nn.Linear(width, width, bias=False)
        prior = config["contact_logit_boost"]
        self.contact_strength = nn.Parameter(torch.full((heads,), math.log(math.expm1(prior))))
        self.output = nn.Linear(width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x, graph, return_attention=False):
        count = len(x)
        q, k, v = self.qkv(x).reshape(count, 3, self.heads, self.head_width).unbind(1)
        source, destination = graph.edge_index
        geometry = self.distance_embedding(self.radial(graph.edge_distance))
        geometry = geometry * graph.edge_has_distance[:, None]
        edge = self.relation(graph.edge_type) + geometry
        logits = (q[destination] * k[source]).sum(-1) / math.sqrt(self.head_width)
        logits = logits + self.edge_bias(edge)
        logits = logits + (graph.edge_type == CONTACT_RELATION)[:, None] * F.softplus(self.contact_strength)
        attention = softmax(logits, destination, num_nodes=count)
        values = v[source] + self.edge_value(edge).reshape(-1, self.heads, self.head_width)
        messages = self.dropout(attention).unsqueeze(-1) * values
        aggregate = x.new_zeros((count, self.heads, self.head_width))
        aggregate.index_add_(0, destination, messages)
        x = self.norm1(x + self.dropout(self.output(aggregate.flatten(1))))
        result = self.norm2(x + self.dropout(self.ff(x)))
        return (result, attention) if return_attention else result


class GeometryView(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.layers = nn.ModuleList([GeometryAttention(config) for _ in range(depth)])

    def forward(self, x, graph):
        for layer in self.layers:
            x = layer(x, graph)
        return x


class GeometryBAN(nn.Module):
    """Low-rank bilinear attention over the eligible ligand x pocket pairs.

    Sparse contacts implement a hard mask: absent pairs have exactly zero
    attention. Each glimpse is normalized across all eligible pairs of this
    complex, then pools their bilinear interaction features.
    """

    def __init__(self, config):
        super().__init__()
        width = config["hidden_dim"]
        self.glimpses, self.rank = config["ban_glimpses"], config["ban_rank"]
        self.left = nn.Linear(width, self.glimpses * self.rank)
        self.right = nn.Linear(width, self.glimpses * self.rank)
        self.attention_vector = nn.Parameter(torch.empty(self.glimpses, self.rank))
        nn.init.xavier_uniform_(self.attention_vector)
        self.radial = RadialBasis(config["radial_bins"], config["contact_cutoff"])
        self.distance_bias = nn.Linear(config["radial_bins"], self.glimpses, bias=False)
        self.projection = nn.Sequential(nn.Linear(self.glimpses * self.rank, width), nn.LayerNorm(width))

    def forward(self, ligand, pocket, contact_index, distance):
        left = torch.tanh(self.left(ligand)).reshape(-1, self.glimpses, self.rank)
        right = torch.tanh(self.right(pocket)).reshape(-1, self.glimpses, self.rank)
        interaction = left[contact_index[0]] * right[contact_index[1]]
        logits = (interaction * self.attention_vector).sum(-1) / math.sqrt(self.rank)
        logits = logits + self.distance_bias(self.radial(distance))
        attention = torch.softmax(logits, dim=0)
        pooled = (attention.unsqueeze(-1) * interaction).sum(0).flatten()
        return self.projection(pooled), attention


class NodeRanker(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, config=None):
        super().__init__()
        supplied = {} if config is None else dict(config)
        if set(supplied) - set(CONFIG):
            raise ValueError(f"Unknown model configuration keys: {sorted(set(supplied) - set(CONFIG))}")
        self.config = {**CONFIG, **supplied}
        c = self.config
        c["view_depths"] = list(c["view_depths"])
        if c["view_depths"] != [2, 4]:
            raise ValueError("This architecture requires independent 2-layer and 4-layer views.")
        for name in ("sequence_dim", "atom_dim", "hidden_dim", "heads", "radial_bins", "ban_glimpses", "ban_rank"):
            if not isinstance(c[name], int) or isinstance(c[name], bool) or c[name] <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if c["hidden_dim"] % c["heads"] or c["radial_bins"] < 2:
            raise ValueError("hidden_dim must divide by heads and radial_bins must be >= 2.")
        for name in ("dropout", "contact_cutoff", "spatial_cutoff", "contact_logit_boost"):
            if not isinstance(c[name], (int, float)) or isinstance(c[name], bool) or not math.isfinite(c[name]):
                raise ValueError(f"{name} must be a finite number.")
        if not 0 <= c["dropout"] < 1 or min(c["contact_cutoff"], c["spatial_cutoff"], c["contact_logit_boost"]) <= 0:
            raise ValueError("Invalid dropout, geometry cutoff, or positive contact prior.")
        width = c["hidden_dim"]
        self.sequence_projection = nn.Sequential(nn.Linear(c["sequence_dim"], width), nn.LayerNorm(width))
        self.ligand_projection = nn.Sequential(nn.Linear(c["atom_dim"], width), nn.LayerNorm(width))
        self.pocket_projection = nn.Sequential(nn.Linear(c["atom_dim"], width), nn.LayerNorm(width))
        self.node_type_embedding = nn.Embedding(3, width)
        self.views = nn.ModuleList([GeometryView(c, depth) for depth in c["view_depths"]])
        self.view_logits = nn.Parameter(torch.zeros(3, 2))
        self.global_readout = nn.Sequential(nn.Linear(3 * width, width), nn.LayerNorm(width))
        self.ban = GeometryBAN(c)
        # Deliberately the only final score transform: no sigmoid and no MLP.
        self.score_output = nn.Linear(2 * width, 1)

    def forward(self, sample, return_attention=False):
        device = self.score_output.weight.device
        fields = ("sequence_features", "ligand_features", "pocket_features", "ligand_pos", "pocket_pos")
        data = dict(sample)
        for name in fields:
            value = data[name]
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating-point tensor.")
            data[name] = value.to(device=device, dtype=torch.float32)
        for name, width in (("sequence_features", self.config["sequence_dim"]),
                            ("ligand_features", self.config["atom_dim"]),
                            ("pocket_features", self.config["atom_dim"])):
            if data[name].ndim != 2 or data[name].shape[1] != width:
                raise ValueError(f"{name} must have shape [N,{width}]; pooled vectors are not accepted.")
        graph = build_graph(data, contact_cutoff=self.config["contact_cutoff"], spatial_cutoff=self.config["spatial_cutoff"])
        x = torch.cat((self.sequence_projection(data["sequence_features"]),
                       self.ligand_projection(data["ligand_features"]),
                       self.pocket_projection(data["pocket_features"])), dim=0)
        x = x + self.node_type_embedding(graph.node_type)
        first, second = (view(x, graph) for view in self.views)
        weights = self.view_logits.softmax(-1)[graph.node_type]
        merged = weights[:, :1] * first + weights[:, 1:] * second
        sequence, ligand, pocket = merged.split(graph.counts)
        z_global = self.global_readout(torch.cat((sequence.mean(0), ligand.mean(0), pocket.mean(0))))
        z_ban, attention = self.ban(ligand, pocket, graph.contact_index, graph.contact_distance)
        score = self.score_output(torch.cat((z_global, z_ban))).squeeze(-1)
        result = {"score": score, "z_global": z_global, "z_ban": z_ban}
        if return_attention:
            result.update({"contact_index": graph.contact_index, "contact_distance": graph.contact_distance,
                           "ban_attention": attention, "view_weights": self.view_logits.softmax(-1),
                           "node_features": {"sequence": sequence, "ligand": ligand, "pocket": pocket}})
        return result
