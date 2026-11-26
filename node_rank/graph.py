"""Build a heterogeneous complex graph without assigning coordinates to sequence nodes."""

from dataclasses import dataclass

import torch


RELATIONS = (
    "self", "sequence_adjacent", "ligand_bond", "pocket_bond",
    "ligand_spatial", "pocket_spatial", "ligand_pocket_contact", "atom_residue_mapping",
)
CONTACT_RELATION = RELATIONS.index("ligand_pocket_contact")


@dataclass
class ComplexGraph:
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_distance: torch.Tensor
    edge_has_distance: torch.Tensor
    contact_index: torch.Tensor  # local ligand and pocket indices, one direction only
    contact_distance: torch.Tensor
    counts: tuple


def _matrix(sample, name, width=None):
    value = sample[name]
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < 1:
        raise ValueError(f"{name} must be a nonempty rank-two tensor.")
    if width is not None and value.shape[1] != width:
        raise ValueError(f"{name} requires width {width}, received {value.shape[1]}.")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values.")
    return value


def _bonds(value, count, device, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 2 or value.dtype != torch.long:
        raise ValueError(f"{name} must be an int64 tensor of shape [2,E].")
    value = value.to(device)
    if value.numel() and (value.min() < 0 or value.max() >= count):
        raise ValueError(f"{name} has an out-of-range atom index.")
    if value.numel() and (value[0] == value[1]).any():
        raise ValueError(f"{name} must not contain self bonds.")
    return torch.unique(torch.cat((value, value.flip(0)), dim=1), dim=1)


def radius_pairs(first, second, cutoff, same=False, chunk=256):
    """Sparse exact-radius pairs; chunking bounds temporary distance-matrix memory."""
    indices, distances = [], []
    for start in range(0, len(first), chunk):
        dist = torch.cdist(first[start:start + chunk], second,
                           compute_mode="donot_use_mm_for_euclid_dist")
        keep = dist <= cutoff
        if same:
            row = torch.arange(len(dist), device=dist.device)
            keep[row, row + start] = False
        if (keep & (dist < 1e-5)).any():
            raise ValueError("Distinct atoms have coincident coordinates; verify the pose and receptor preparation.")
        local = keep.nonzero(as_tuple=False).T
        indices.append(torch.stack((local[0] + start, local[1])))
        distances.append(dist[local[0], local[1]])
    return torch.cat(indices, dim=1), torch.cat(distances)


def build_graph(sample, *, contact_cutoff=5.0, spatial_cutoff=4.5):
    """Node order is sequence, ligand atoms, pocket atoms. Distances are in Å.

    Sequence adjacency defaults to tensor row order. ``sequence_index`` can
    explicitly specify biological positions when rows are permuted. Mapping
    indices always address the current sequence tensor rows; -1 means an
    explicitly unmapped pocket atom and creates no artificial mapping edge.
    """
    if contact_cutoff <= 0 or spatial_cutoff <= 0:
        raise ValueError("Graph distance cutoffs must be positive.")
    seq = _matrix(sample, "sequence_features")
    lig = _matrix(sample, "ligand_features")
    poc = _matrix(sample, "pocket_features")
    lp, pp = _matrix(sample, "ligand_pos", 3), _matrix(sample, "pocket_pos", 3)
    ns, nl, np = len(seq), len(lig), len(poc)
    if len(lp) != nl or len(pp) != np:
        raise ValueError("Atom coordinates and node features have different lengths.")
    device = seq.device
    if any(value.device != device for value in (lig, poc, lp, pp)):
        raise ValueError("Graph input tensors must share a device.")
    lb = _bonds(sample["ligand_bonds"], nl, device, "ligand_bonds")
    pb = _bonds(sample["pocket_bonds"], np, device, "pocket_bonds")
    mapping = sample["pocket_to_sequence"]
    if not isinstance(mapping, torch.Tensor) or mapping.dtype != torch.long or mapping.shape != (np,):
        raise ValueError("pocket_to_sequence must be int64 with one entry per pocket atom.")
    mapping = mapping.to(device)
    if (mapping < -1).any() or (mapping >= ns).any():
        raise ValueError("Pocket-to-sequence mapping is out of range.")
    if not (mapping >= 0).any():
        raise ValueError("At least one verified structure-to-sequence mapping is required.")
    order = sample.get("sequence_index", torch.arange(ns, device=device))
    if not isinstance(order, torch.Tensor) or order.dtype != torch.long or order.shape != (ns,):
        raise ValueError("sequence_index must contain one int64 biological position per residue.")
    order = order.to(device)
    if not torch.equal(order.sort().values, torch.arange(ns, device=device)):
        raise ValueError("sequence_index must be a permutation of 0..N_sequence-1.")
    bio_to_row = order.argsort()
    seq_edges = torch.stack((bio_to_row[:-1], bio_to_row[1:]))
    seq_edges = torch.cat((seq_edges, seq_edges.flip(0)), dim=1)
    total = ns + nl + np
    edges, types, ds, has_ds = [], [], [], []

    def add(index, relation, distance=None):
        count = index.shape[1]
        edges.append(index)
        types.append(torch.full((count,), relation, dtype=torch.long, device=device))
        ds.append(torch.zeros(count, device=device) if distance is None else distance.float())
        has_ds.append(torch.full((count,), distance is not None, dtype=torch.bool, device=device))

    all_nodes = torch.arange(total, device=device)
    add(torch.stack((all_nodes, all_nodes)), 0)
    add(seq_edges, 1)
    add(lb + ns, 2, torch.linalg.vector_norm(lp[lb[0]] - lp[lb[1]], dim=1))
    add(pb + ns + nl, 3, torch.linalg.vector_norm(pp[pb[0]] - pp[pb[1]], dim=1))
    for pos, bonds, offset, relation in ((lp, lb, ns, 4), (pp, pb, ns + nl, 5)):
        index, distance = radius_pairs(pos, pos, spatial_cutoff, same=True)
        # A chemical bond is represented once, by its chemical edge relation.
        if bonds.numel():
            keep = ~torch.isin(index[0] * len(pos) + index[1], bonds[0] * len(pos) + bonds[1])
            index, distance = index[:, keep], distance[keep]
        add(index + offset, relation, distance)
    contact, distance = radius_pairs(lp, pp, contact_cutoff)
    if not contact.numel():
        raise ValueError("No ligand-pocket contacts inside the contact cutoff; verify the bound pose and coordinate frame.")
    contact_global = contact + torch.tensor([[ns], [ns + nl]], device=device)
    add(torch.cat((contact_global, contact_global.flip(0)), dim=1), CONTACT_RELATION,
        torch.cat((distance, distance)))
    mapped_atoms = (mapping >= 0).nonzero(as_tuple=True)[0]
    mapping_edges = torch.stack((mapped_atoms + ns + nl, mapping[mapped_atoms]))
    add(torch.cat((mapping_edges, mapping_edges.flip(0)), dim=1), 7)
    return ComplexGraph(
        node_type=torch.cat([torch.full((n,), i, dtype=torch.long, device=device)
                             for i, n in enumerate((ns, nl, np))]),
        edge_index=torch.cat(edges, dim=1), edge_type=torch.cat(types),
        edge_distance=torch.cat(ds), edge_has_distance=torch.cat(has_ds),
        contact_index=contact, contact_distance=distance, counts=(ns, nl, np),
    )
