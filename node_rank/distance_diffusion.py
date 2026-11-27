"""Diffuse ligand-to-pocket distance maps from independently prepared structures."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .graph import CONTACT_RELATION, RELATIONS
from .model import GeometryView


DISTANCE_ARCHITECTURE = "dualview_ligand_pocket_distance_diffusion_v1"
DISTANCE_CONFIG = {
    "ligand_dim": 25,
    "pocket_dim": 39,
    "hidden_dim": 64,
    "heads": 4,
    "pair_dim": 32,
    "dropout": 0.1,
    "radial_bins": 16,
    "spatial_cutoff": 10.0,
    "ligand_neighbors": 12,
    "pocket_neighbors": 16,
    "cross_anchors": 8,
    "distance_cap": 24.0,
    "diffusion_steps": 200,
}


@dataclass
class _GeometryGraph:
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_distance: torch.Tensor
    edge_has_distance: torch.Tensor


def _spatial_edges(positions, neighbors, cutoff):
    distances = torch.cdist(positions, positions, compute_mode="donot_use_mm_for_euclid_dist")
    distances.fill_diagonal_(torch.inf)
    values, source = distances.topk(min(neighbors, len(positions)), largest=False)
    destination, column = (values <= cutoff).nonzero(as_tuple=True)
    return torch.stack((source[destination, column], destination)), values[destination, column]


def _geometry_graph(ligand, pocket, bonds, config):
    """Use intra-structure distances and pose-independent bipartite links."""
    device = ligand.device
    ligand_count, pocket_count = len(ligand), len(pocket)
    node = torch.arange(ligand_count + pocket_count, device=device)
    node_type = (node >= ligand_count).long()
    indices, relations, distances, distance_masks = [], [], [], []

    def add(edges, relation, lengths=None):
        size = edges.shape[1]
        indices.append(edges)
        relations.append(torch.full((size,), relation, device=device, dtype=torch.long))
        distances.append(ligand.new_zeros(size) if lengths is None else lengths)
        distance_masks.append(torch.full((size,), lengths is not None, device=device, dtype=torch.bool))

    add(torch.stack((node, node)), RELATIONS.index("self"))
    bonds = torch.unique(torch.cat((bonds, bonds.flip(0)), dim=1), dim=1)
    bond_lengths = torch.linalg.vector_norm(ligand[bonds[0]] - ligand[bonds[1]], dim=-1)
    add(bonds, RELATIONS.index("ligand_bond"), bond_lengths)

    ligand_edges, ligand_lengths = _spatial_edges(ligand, config["ligand_neighbors"], config["spatial_cutoff"])
    if bonds.numel():
        bond_keys = bonds[0] * ligand_count + bonds[1]
        spatial_keys = ligand_edges[0] * ligand_count + ligand_edges[1]
        keep = ~torch.isin(spatial_keys, bond_keys)
        ligand_edges, ligand_lengths = ligand_edges[:, keep], ligand_lengths[keep]
    add(ligand_edges, RELATIONS.index("ligand_spatial"), ligand_lengths)
    pocket_edges, pocket_lengths = _spatial_edges(pocket, config["pocket_neighbors"], config["spatial_cutoff"])
    add(pocket_edges + ligand_count, RELATIONS.index("pocket_spatial"), pocket_lengths)

    # A small set of receptor anchors lets information pass between the two
    # graphs. Their selection and links are independent of the ligand pose.
    centrality = (pocket - pocket.mean(0)).square().sum(-1)
    anchors = centrality.topk(min(config["cross_anchors"], pocket_count), largest=False).indices
    source = node[:ligand_count].repeat_interleave(len(anchors))
    destination = anchors.repeat(ligand_count) + ligand_count
    cross = torch.stack((source, destination))
    add(torch.cat((cross, cross.flip(0)), dim=1), CONTACT_RELATION)
    return _GeometryGraph(
        node_type, torch.cat(indices, dim=1), torch.cat(relations),
        torch.cat(distances), torch.cat(distance_masks),
    )


def _normal_noise(shape, device, generator):
    random_device = device if generator is None else generator.device
    return torch.randn(shape, device=random_device, dtype=torch.float32, generator=generator).to(device)


class DistanceDiffusion(nn.Module):
    """Two- and four-layer geometric encoder and conditional Gaussian diffusion."""

    architecture = DISTANCE_ARCHITECTURE

    def __init__(self, config=None):
        super().__init__()
        supplied = {} if config is None else dict(config)
        unknown = set(supplied) - set(DISTANCE_CONFIG)
        if unknown:
            raise ValueError(f"Unknown distance model configuration keys: {sorted(unknown)}")
        self.config = c = {**DISTANCE_CONFIG, **supplied}
        for name in (
            "ligand_dim", "pocket_dim", "hidden_dim", "heads", "pair_dim", "radial_bins",
            "ligand_neighbors", "pocket_neighbors", "cross_anchors", "diffusion_steps",
        ):
            if type(c[name]) is not int or c[name] < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if c["hidden_dim"] % c["heads"] or min(c["radial_bins"], c["diffusion_steps"]) < 2:
            raise ValueError("Invalid number of heads, radial bins, or diffusion steps.")
        for name in ("distance_cap", "spatial_cutoff", "dropout"):
            if not math.isfinite(c[name]):
                raise ValueError(f"{name} must be finite.")
        if min(c["distance_cap"], c["spatial_cutoff"]) <= 0 or not 0 <= c["dropout"] < 1:
            raise ValueError("Invalid distance bound, spatial cutoff, or dropout.")

        width, pair_width = c["hidden_dim"], c["pair_dim"]
        self.ligand_projection = nn.Sequential(nn.Linear(c["ligand_dim"], width), nn.LayerNorm(width))
        self.pocket_projection = nn.Sequential(nn.Linear(c["pocket_dim"], width), nn.LayerNorm(width))
        self.node_type_embedding = nn.Embedding(2, width)
        # Candidate cross links carry no distance and have a near-zero prior.
        view_config = {**c, "contact_cutoff": c["spatial_cutoff"], "contact_logit_boost": 1e-4}
        self.views = nn.ModuleList(GeometryView(view_config, depth) for depth in (2, 4))
        self.view_logits = nn.Parameter(torch.zeros(2, 2))

        self.ligand_pair = nn.Linear(width, 2 * pair_width)
        self.pocket_pair = nn.Linear(width, 2 * pair_width)
        self.noisy_distance = nn.Linear(1, pair_width)
        self.timestep = nn.Sequential(nn.Linear(64, pair_width), nn.SiLU(), nn.Linear(pair_width, pair_width))
        self.pair_row = nn.Linear(pair_width, pair_width, bias=False)
        self.pair_column = nn.Linear(pair_width, pair_width, bias=False)
        self.noise_output = nn.Sequential(nn.LayerNorm(pair_width), nn.Linear(pair_width, 1))
        self.score_output = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, 1))

        times = torch.linspace(0, 1, c["diffusion_steps"] + 1, dtype=torch.float64)
        cumulative = torch.cos((times + 0.008) / 1.008 * (math.pi / 2)).square()
        cumulative = (cumulative / cumulative[0]).clamp_min(1e-7)
        self.register_buffer("alpha_bar", cumulative.float())
        self.register_buffer("time_frequencies", torch.exp(-math.log(10000) * torch.arange(32) / 31))

    def _encode(self, sample):
        device = self.view_logits.device
        ligand_pos = sample["ligand_pos"].to(device=device, dtype=torch.float32)
        pocket_pos = sample["pocket_pos"].to(device=device, dtype=torch.float32)
        ligand_feat = sample["ligand_features"].to(device=device, dtype=torch.float32)
        pocket_feat = sample["pocket_features"].to(device=device, dtype=torch.float32)
        bonds = sample["ligand_bonds"].to(device=device, dtype=torch.long)
        for name, coords, features, width in (
            ("ligand", ligand_pos, ligand_feat, self.config["ligand_dim"]),
            ("pocket", pocket_pos, pocket_feat, self.config["pocket_dim"]),
        ):
            if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords):
                raise ValueError(f"{name}_pos must be a nonempty [atoms, 3] tensor.")
            if features.shape != (len(coords), width):
                raise ValueError(f"{name}_features must have shape [atoms, {width}].")
        if bonds.ndim != 2 or bonds.shape[0] != 2:
            raise ValueError("ligand_bonds must have shape [2, bonds].")

        graph = _geometry_graph(ligand_pos, pocket_pos, bonds, self.config)
        x = torch.cat((self.ligand_projection(ligand_feat), self.pocket_projection(pocket_feat)))
        x = x + self.node_type_embedding(graph.node_type)
        first, second = (view(x, graph) for view in self.views)
        weights = self.view_logits.softmax(-1)[graph.node_type]
        merged = weights[:, :1] * first + weights[:, 1:] * second
        return merged.split((len(ligand_pos), len(pocket_pos)))

    def _score(self, ligand, pocket):
        left, right = ligand.mean(0), pocket.mean(0)
        return self.score_output(torch.cat((left, right, left * right))).squeeze(-1)

    def score_sample(self, sample):
        """Score the pair without pose generation or diffusion sampling."""
        return self._score(*self._encode(sample))

    def _pair_context(self, ligand, pocket):
        left, left_product = self.ligand_pair(ligand).chunk(2, dim=-1)
        right, right_product = self.pocket_pair(pocket).chunk(2, dim=-1)
        return left[:, None] + right[None, :] + left_product[:, None] * right_product[None, :]

    def _predict_noise(self, context, noisy, timestep):
        angles = timestep.to(torch.float32) * self.time_frequencies
        time = self.timestep(torch.cat((angles.sin(), angles.cos())))
        pairs = F.silu(context + self.noisy_distance(noisy.unsqueeze(-1)) + time)
        row_context = self.pair_row(pairs.mean(dim=1, keepdim=True))
        column_context = self.pair_column(pairs.mean(dim=0, keepdim=True))
        return self.noise_output(pairs + row_context + column_context).squeeze(-1).float()

    def training_loss(self, sample, distances_angstrom, *, generator=None):
        """Train epsilon prediction with a fresh timestep and noise for one complex."""
        ligand, pocket = self._encode(sample)
        context = self._pair_context(ligand, pocket)
        device = context.device
        distances = distances_angstrom.to(device=device, dtype=torch.float32)
        if distances.shape != context.shape[:2]:
            raise ValueError("Distance labels must have shape [ligand atoms, pocket anchors].")
        clean = distances.clamp(0, self.config["distance_cap"]) * (2 / self.config["distance_cap"]) - 1
        random_device = device if generator is None else generator.device
        timestep = torch.randint(1, len(self.alpha_bar), (), device=random_device, generator=generator).to(device)
        signal = self.alpha_bar[timestep]
        noise = _normal_noise(clean.shape, device, generator)
        noisy = signal.sqrt() * clean + (1 - signal).sqrt() * noise
        predicted = self._predict_noise(context, noisy, timestep)
        return {"distance_loss": F.mse_loss(predicted, noise), "score": self._score(ligand, pocket), "timestep": timestep}

    @torch.no_grad()
    def sample_distances(self, sample, num_steps=25, generator=None):
        """Sample a distance map via DDIM, using fresh Gaussian initial noise."""
        total_steps = self.config["diffusion_steps"]
        if type(num_steps) is not int or not 1 <= num_steps <= total_steps:
            raise ValueError(f"num_steps must be between 1 and {total_steps}.")
        ligand, pocket = self._encode(sample)
        context = self._pair_context(ligand, pocket)
        device = context.device
        noisy = _normal_noise(context.shape[:2], device, generator)
        steps = [round(total_steps * step / num_steps) for step in range(num_steps, -1, -1)]
        for current, following in zip(steps, steps[1:]):
            signal = self.alpha_bar[current]
            next_signal = self.alpha_bar[following]
            timestep = torch.tensor(current, device=device)
            predicted = self._predict_noise(context, noisy, timestep)
            clean = ((noisy - (1 - signal).sqrt() * predicted) / signal.sqrt()).clamp(-1, 1)
            # Recompute epsilon consistently if the clean estimate was clipped.
            predicted = (noisy - signal.sqrt() * clean) / (1 - signal).sqrt()
            noisy = next_signal.sqrt() * clean + (1 - next_signal).sqrt() * predicted
        distances = (noisy + 1) * (self.config["distance_cap"] / 2)
        return {"distances": distances, "score": self._score(ligand, pocket)}
