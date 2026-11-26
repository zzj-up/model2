"""Meaningful synthetic checks of architecture correctness, not predictive accuracy."""

import argparse
import copy
import io
import json
from pathlib import Path

import torch
from torch import nn

from .graph import CONTACT_RELATION, build_graph
from .model import ARCHITECTURE, CONFIG, GeometryAttention, NodeRanker


def synthetic_sample(sequence_dim=16, atom_dim=8, seed=17):
    generator = torch.Generator().manual_seed(seed)
    return {
        "sequence_features": torch.randn(7, sequence_dim, generator=generator),
        "ligand_features": torch.randn(4, atom_dim, generator=generator),
        "pocket_features": torch.randn(6, atom_dim, generator=generator),
        "ligand_pos": torch.tensor([[0., 0., 0.], [1.3, 0., 0.], [0., 1.4, 0.], [0., 0., 1.5]]),
        "pocket_pos": torch.tensor([[2., 2., 0.], [3., 2., 1.], [1., 3., 1.], [4., 1., 0.], [2., 1., 2.], [6., 0., 0.]]),
        "ligand_bonds": torch.tensor([[0, 0, 0], [1, 2, 3]]),
        "pocket_bonds": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "pocket_to_sequence": torch.tensor([0, 1, 2, 4, 5, -1]),
        "compound_id": "synthetic_compound", "target_id": "synthetic_target", "pair_id": "synthetic_pair",
    }


def permuted_sample(sample):
    result = copy.deepcopy(sample)
    sp, lp, pp = torch.tensor([2, 0, 5, 1, 6, 3, 4]), torch.tensor([2, 0, 3, 1]), torch.tensor([5, 2, 0, 4, 1, 3])
    si, li, pi = sp.argsort(), lp.argsort(), pp.argsort()
    result["sequence_features"] = sample["sequence_features"][sp]
    result["sequence_index"] = sp
    for name in ("ligand_features", "ligand_pos"):
        result[name] = sample[name][lp]
    for name in ("pocket_features", "pocket_pos"):
        result[name] = sample[name][pp]
    result["ligand_bonds"] = li[sample["ligand_bonds"]]
    result["pocket_bonds"] = pi[sample["pocket_bonds"]]
    mapping = sample["pocket_to_sequence"][pp]
    result["pocket_to_sequence"] = torch.where(mapping >= 0, si[mapping.clamp_min(0)], -1)
    return result


def run_checks(device="cpu"):
    torch.manual_seed(20261117)
    torch.set_num_threads(1)
    config = {**CONFIG, "sequence_dim": 16, "atom_dim": 8, "hidden_dim": 32,
              "ban_rank": 8, "ban_glimpses": 2, "dropout": 0.0}
    model = NodeRanker(config).to(device).eval()
    sample = synthetic_sample()
    output = model(sample, return_attention=True)
    assert output["score"].ndim == 0 and torch.isfinite(output["score"])
    assert [len(view.layers) for view in model.views] == [2, 4]
    assert isinstance(model.score_output, nn.Linear)
    assert not any("gate" in name.lower() for name, _ in model.named_parameters())
    assert model.views[0].layers[0].qkv.weight.data_ptr() != model.views[1].layers[0].qkv.weight.data_ptr()
    weights = output["ban_attention"]
    assert torch.allclose(weights.sum(0), torch.ones(weights.shape[1], device=weights.device), atol=1e-6)
    actual_distance = torch.cdist(sample["ligand_pos"], sample["pocket_pos"])
    eligible = (actual_distance <= config["contact_cutoff"]).nonzero().T
    assert torch.equal(output["contact_index"].cpu(), eligible)
    assert eligible.shape[1] < actual_distance.numel()
    assert torch.allclose(output["contact_distance"].cpu(), actual_distance[eligible[0], eligible[1]], atol=1e-6)
    # All learned submodules and all three input feature sources affect the score.
    gradient_sample = copy.deepcopy(sample)
    for name in ("sequence_features", "ligand_features", "pocket_features"):
        gradient_sample[name].requires_grad_(True)
    model(gradient_sample)["score"].square().backward()
    missing = [name for name, value in model.named_parameters()
               if value.grad is None or not torch.isfinite(value.grad).all() or value.grad.abs().sum() == 0]
    assert not missing, f"Parameters missing finite nonzero gradients: {missing}"
    for name in ("sequence_features", "ligand_features", "pocket_features"):
        assert gradient_sample[name].grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    # Rigid motion cannot change score; sequence rows need no 3D coordinates.
    rotated = copy.deepcopy(sample)
    rotation, _ = torch.linalg.qr(torch.tensor([[0.2, 0.4, 0.8], [0.7, -0.3, 0.1], [0.5, 0.6, -0.4]]))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    for name in ("ligand_pos", "pocket_pos"):
        rotated[name] = sample[name] @ rotation + torch.tensor([9.0, -3.0, 2.0])
    with torch.no_grad():
        rigid_score = model(rotated)["score"]
        permutation_score = model(permuted_sample(sample))["score"]
    assert torch.allclose(output["score"], rigid_score, atol=2e-5, rtol=1e-5)
    assert torch.allclose(output["score"], permutation_score, atol=2e-5, rtol=1e-5)
    # A contact prior must change contact-vs-other attention after normalization.
    graph = build_graph(sample)
    layer = GeometryAttention(config).eval()
    x = torch.randn(sum(graph.counts), config["hidden_dim"])
    with torch.no_grad():
        layer.qkv.weight.zero_()
        layer.edge_bias.weight.zero_()
        layer.contact_strength.fill_(-8.)
        _, low = layer(x, graph, return_attention=True)
        layer.contact_strength.fill_(2.)
        _, high = layer(x, graph, return_attention=True)
    contacts = graph.edge_type == CONTACT_RELATION
    assert (high[contacts] > low[contacts]).all()
    # Serialization is weights-only compatible, exact, and architecture-specific.
    buffer = io.BytesIO()
    torch.save({"architecture": ARCHITECTURE, "config": model.config, "state_dict": model.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location=device, weights_only=True)
    restored = NodeRanker(checkpoint["config"]).to(device).eval()
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    with torch.no_grad():
        # CUDA scatter/index_add reductions can differ at floating-point roundoff.
        assert torch.allclose(model(sample)["score"], restored(sample)["score"], atol=1e-6, rtol=1e-5)
        # Independent complexes do not affect one another or share a softmax.
        before = restored(sample)["score"]
        restored(synthetic_sample(seed=25))
        after = restored(sample)["score"]
        assert torch.allclose(before, after, atol=1e-6, rtol=1e-5)
    invalids = []
    far = copy.deepcopy(sample)
    far["pocket_pos"] += 100.
    invalids.append(("no_contacts", far))
    bad_map = copy.deepcopy(sample)
    bad_map["pocket_to_sequence"][0] = 100
    invalids.append(("bad_mapping", bad_map))
    pooled = copy.deepcopy(sample)
    pooled["sequence_features"] = pooled["sequence_features"].mean(0)
    invalids.append(("pooled_features", pooled))
    nonfinite = copy.deepcopy(sample)
    nonfinite["ligand_pos"][0, 0] = float("nan")
    invalids.append(("nonfinite_coordinates", nonfinite))
    coincident = copy.deepcopy(sample)
    coincident["pocket_pos"][0] = coincident["ligand_pos"][0]
    invalids.append(("coincident_distinct_atoms", coincident))
    for name, value in invalids:
        try:
            restored(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid input: {name}")
    return {
        "architecture": ARCHITECTURE, "device": device, "passed": True,
        "checks": ["parallel_independent_2_4_views", "single_linear_output_no_gating", "sparse_contact_mask",
                   "ban_glimpse_normalization", "all_parameter_and_input_gradients", "rigid_motion_invariance",
                   "typed_node_permutation_invariance", "local_prior_changes_shared_attention",
                   "weights_only_reload_exact_parameters", "independent_complex_scoring", "invalid_input_rejection"],
        "synthetic_only": True, "training_steps": 0,
        "contact_count": eligible.shape[1],
        "rigid_score_abs_difference": float((output["score"] - rigid_score).abs().detach()),
        "permutation_score_abs_difference": float((output["score"] - permutation_score).abs().detach()),
        "parameter_count_small_test_model": sum(p.numel() for p in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        raise FileExistsError(f"Refusing to overwrite an existing verification report: {args.report}")
    report = run_checks(args.device)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
