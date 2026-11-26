"""Temporary synthetic integration checks; deployed checkpoint is never optimized."""
import argparse
import copy
import csv
import json
import tempfile
from pathlib import Path

import torch

from . import predict, train
from .io import (FEATURE_SCHEMA, initialize, load_checkpoint, load_features,
                 rank_rows, read_labels, save_new, sha256)
from .verify_model import synthetic_sample


def rejects(call, exception=ValueError):
    try:
        call()
    except exception:
        return
    raise AssertionError("Invalid operation was accepted.")


def run_checks():
    torch.set_num_threads(1)
    checks = []
    with tempfile.TemporaryDirectory(prefix="node_rank_verify_") as directory:
        root = Path(directory)
        first = synthetic_sample(2560, 96, 17)
        second = synthetic_sample(2560, 96, 18)
        first.update(pair_id="synthetic_A", target_id="target_A")
        second.update(pair_id="synthetic_B", target_id="target_B")
        samples = [first, second]
        policy = {"diagnostic": "synthetic_test_only_no_official_features"}
        feature_path = root / "features.pt"
        save_new(feature_path, {"schema": FEATURE_SCHEMA, "samples": samples, "feature_policy": policy})
        loaded = load_features(feature_path)
        assert len(loaded["samples"]) == 2
        checks.append("full_width_node_feature_contract")
        bad = copy.deepcopy(loaded)
        bad["samples"][0]["sequence_features"] = torch.zeros(2560)
        save_new(root / "bad_features.pt", bad)
        rejects(lambda: load_features(root / "bad_features.pt"))
        checks.append("pooled_features_rejected")

        def labels(path, rows):
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["pair_id", "relevance", "comparison_group", "endpoint"])
                writer.writeheader()
                writer.writerows(rows)

        label_rows = [dict(pair_id="synthetic_A", relevance=2, comparison_group="same_assay", endpoint="relevance"),
                      dict(pair_id="synthetic_B", relevance=1, comparison_group="same_assay", endpoint="relevance")]
        labels(root / "labels.csv", label_rows)
        groups = read_labels(root / "labels.csv", samples)
        assert groups[0][1] == [(0, 1)]
        different_compounds = copy.deepcopy(samples)
        different_compounds[1]["compound_id"] = "another_compound"
        rejects(lambda: read_labels(root / "labels.csv", different_compounds))
        mixed = copy.deepcopy(label_rows)
        mixed[1]["endpoint"] = "other_endpoint"
        labels(root / "mixed.csv", mixed)
        rejects(lambda: read_labels(root / "mixed.csv", samples))
        mixed = copy.deepcopy(label_rows)
        mixed[1]["relevance"] = 2
        labels(root / "ties.csv", mixed)
        rejects(lambda: read_labels(root / "ties.csv", samples))
        duplicate = copy.deepcopy(samples)
        duplicate[1]["target_id"] = duplicate[0]["target_id"]
        rejects(lambda: read_labels(root / "labels.csv", duplicate))
        checks.append("reject_cross_compound_mixed_endpoint_all_ties_and_duplicate_target_training")
        checkpoint = root / "init.pt"
        initialize(checkpoint, config={"hidden_dim": 32, "ban_rank": 8, "ban_glimpses": 2, "dropout": 0.0})
        original_hash = sha256(checkpoint)
        model, payload = load_checkpoint(checkpoint)
        rejects(lambda: initialize(checkpoint), FileExistsError)
        tampered = copy.deepcopy(payload)
        tampered["model_state"]["score_output.weight"][0, 0] += 1
        save_new(root / "tampered.pt", tampered)
        rejects(lambda: load_checkpoint(root / "tampered.pt"))
        save_new(root / "legacy.pt", {"schema": "legacy_affinity"})
        rejects(lambda: load_checkpoint(root / "legacy.pt"))
        checks.append("initialization_reproducibility_no_overwrite_and_legacy_rejection")
        common = ["--features", str(feature_path), "--checkpoint", str(checkpoint), "--device", "cpu", "--threads", "1"]
        rejects(lambda: predict.main(common + ["--output-dir", str(root / "blocked")]))
        assert not (root / "blocked").exists()
        predict.main(common + ["--output-dir", str(root / "diagnostics"), "--allow-untrained-diagnostics", "--save-attention"])
        summary = json.loads((root / "diagnostics/summary.json").read_text())
        assert summary["diagnostic_only"] and summary["compound_groups_with_multiple_targets"] == 1
        checks.append("untrained_predictions_require_explicit_diagnostic_flag")
        model.eval()
        with torch.no_grad():
            score = model(first)["score"]
            changed_labels = dict(first, p_affinity=-999, ranking_label=999, target_id="changed_label", compound_id="changed")
            assert torch.equal(score, model(changed_labels)["score"])
        checks.append("experimental_labels_and_identifiers_do_not_enter_forward")
        train.main(common + ["--labels", str(root / "labels.csv"), "--run-dir", str(root / "training"), "--epochs", "1"])
        trained, trained_payload = load_checkpoint(root / "training/last.pt")
        assert trained_payload["training_steps"] == 1
        assert any(not torch.equal(p, trained_payload["model_state"][key]) for key, p in payload["model_state"].items())
        assert sha256(checkpoint) == original_hash
        checks.append("temporary_pairwise_optimizer_step_checkpoint_reload_initial_weights_unchanged")
        predict.main(["--features", str(feature_path), "--checkpoint", str(root / "training/last.pt"),
                      "--output-dir", str(root / "trained_predictions"), "--device", "cpu", "--threads", "1"])
        mismatch = copy.deepcopy(loaded)
        mismatch["feature_policy"] = {"different_policy": True}
        save_new(root / "mismatched.pt", mismatch)
        rejects(lambda: predict.main(["--features", str(root / "mismatched.pt"), "--checkpoint", str(root / "training/last.pt"),
                                      "--output-dir", str(root / "rejected_policy"), "--device", "cpu"]))
        checks.append("trained_inference_and_feature_policy_guard")
        rows = [dict(pair_id="c", compound_id="x", target_id="c", score=1.),
                dict(pair_id="a", compound_id="x", target_id="a", score=2.),
                dict(pair_id="b", compound_id="x", target_id="b", score=2.),
                dict(pair_id="d", compound_id="y", target_id="d", score=100.)]
        result = rank_rows(rows)
        assert [row["rank"] for row in result] == [1, 1, 3, 1]
        assert [row["pair_id"] for row in result] == ["a", "b", "c", "d"]
        checks.append("descending_per_compound_ranking_with_ties")
    return {"status": "passed", "checks": checks, "data": "temporary_synthetic_only",
            "synthetic_optimizer_steps": 1, "deployed_weights_updated": False,
            "real_data_training_performed": False, "predictive_performance_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_checks()
    if args.report:
        save_new(args.report, report, json_file=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
