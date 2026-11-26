"""CPU-only synthetic integration checks of affinity-score pause/resume.

All features, labels, initialization and trained checkpoints live in a temporary
directory. No real PDBbind split, deployed weight or official encoder is loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from .io import initialize, sha256
from .train_pdbbind import (INDEX_SCHEMA, SCORE_MEANING, TASK, FeatureIndex,
                            _validate, audit_inputs, load_trained_score, parser,
                            run_training)
from .verify_model import synthetic_sample


def assert_equal(first, second, path="root"):
    """Compare serialized optimizer, RNG and progress trees without tolerances."""
    assert type(first) is type(second), f"Different types at {path}."
    if isinstance(first, torch.Tensor):
        assert first.dtype == second.dtype and first.shape == second.shape, path
        assert torch.equal(first, second), f"Tensor differs at {path}."
    elif isinstance(first, dict):
        assert first.keys() == second.keys(), f"Different keys at {path}."
        for key in first:
            assert_equal(first[key], second[key], f"{path}.{key}")
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second), f"Different lengths at {path}."
        for index, (left, right) in enumerate(zip(first, second)):
            assert_equal(left, right, f"{path}[{index}]")
    else:
        assert first == second, f"Value differs at {path}: {first!r} vs {second!r}."


def feature_index(directory, split, labels):
    directory.mkdir()
    rows = []
    for index, label in enumerate(labels):
        sample = synthetic_sample(2560, 96, seed=100 + index + (10 if split == "validation" else 0))
        identity = f"synthetic_{split}_{index}"
        sample.update(pair_id=identity, compound_id=identity + "_compound",
                      target_id=identity + "_target", protein_sequence="ACDEFGH")
        sequence = sample.pop("sequence_features")
        sample_path = directory / (identity + "_atoms.pt")
        sequence_path = directory / (identity + "_sequence.pt")
        torch.save(sample, sample_path)
        torch.save(sequence, sequence_path)
        rows.append({"pair_id": sample["pair_id"], "compound_id": sample["compound_id"],
                     "target_id": sample["target_id"], "compound_group": identity + "_parent",
                     "p_affinity": label, "affinity_endpoint": "Kd" if index % 2 == 0 else "Ki",
                     "sample_path": sample_path.name, "sample_sha256": sha256(sample_path),
                     "sequence_path": sequence_path.name, "sequence_sha256": sha256(sequence_path)})
    path = directory / "features.json"
    path.write_text(json.dumps({"schema": INDEX_SCHEMA, "split": split, "samples": rows,
                               "feature_policy": {"synthetic_only": True,
                                                  "diagnostic": "no_official_encoder_or_real_split"}}),
                    encoding="utf-8")
    return path


def run_checks():
    torch.set_num_threads(1)
    checks = []
    with tempfile.TemporaryDirectory(prefix="node_rank_score_resume_verify_") as temporary:
        root = Path(temporary)
        train_path = feature_index(root / "train", "train", [1.0, 3.0, 5.0])
        validation_path = feature_index(root / "validation", "validation", [8.0, 10.0])
        train = FeatureIndex(train_path, "train")
        validation = FeatureIndex(validation_path, "validation")
        audit = audit_inputs(train, validation)
        assert audit["scaling"]["mean"] == 3.0
        assert math.isclose(audit["scaling"]["std"], math.sqrt(8.0 / 3.0), rel_tol=1e-15)
        assert audit["scaling"]["source"] == "training_labels_only"
        assert audit["holdout_used"] is False
        assert audit["missing_interactions_used_as_negatives"] is False
        checks.append("full_width_sharded_features_compound_disjoint_and_train_only_scaling")

        initial_path = root / "initial.pt"
        # Nonzero dropout exercises Torch RNG restoration while keeping this
        # integration test small. The input widths and 2/4 views stay unchanged.
        initialize(initial_path, seed=20261118,
                   config={"hidden_dim": 32, "ban_rank": 8, "ban_glimpses": 2, "dropout": 0.1})
        initial_digest = sha256(initial_path)
        common = ["--train-index", str(train_path), "--validation-index", str(validation_path),
                  "--initial-checkpoint", str(initial_path), "--epochs", "2", "--accumulate", "2",
                  "--seed", "20261118", "--device", "cpu", "--threads", "1",
                  "--checkpoint-every", "1", "--checkpoint-seconds", "120"]

        def invoke(arguments):
            with contextlib.redirect_stdout(io.StringIO()):
                return run_training(parser().parse_args(arguments))

        continuous_dir, resumed_dir = root / "continuous", root / "resumed"
        continuous_summary = invoke(common + ["--run-dir", str(continuous_dir)])
        pause_summary = invoke(common + ["--run-dir", str(resumed_dir), "--stop-after-updates", "1"])
        assert pause_summary["status"] == "paused" and pause_summary["training_steps"] == 1
        assert pause_summary["epoch"] == 1 and pause_summary["epoch_cursor"] == 2
        paused = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=True)
        assert paused["progress"]["order"] is not None
        assert paused["progress"]["cursor"] == 2 and paused["progress"]["epoch_samples"] == 2
        checks.append("mid_epoch_pause_saves_order_cursor_optimizer_and_rng")

        resumed_summary = invoke(["--run-dir", str(resumed_dir), "--resume"])
        assert continuous_summary["status"] == resumed_summary["status"] == "complete"
        assert resumed_summary["completed_epochs"] == 2
        assert resumed_summary["training_steps"] == 4 and resumed_summary["samples_seen"] == 6
        continuous = torch.load(continuous_dir / "last.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=True)
        compared = ("model_state", "optimizer_state", "rng_state", "progress", "data_audit",
                    "target_scaling", "training_steps", "epochs_requested", "optimization_seed",
                    "initial_checkpoint_sha256", "training_task", "score_meaning", "feature_policy")
        for key in compared:
            assert_equal(continuous[key], resumed[key], key)
        assert_equal(json.loads((continuous_dir / "metrics.json").read_text()),
                     json.loads((resumed_dir / "metrics.json").read_text()), "published_metrics")
        checks.append("continuous_vs_resumed_exact_weights_optimizer_rng_progress_and_metrics")

        for name in ("last.pt", "best_validation.pt"):
            model, payload = load_trained_score(resumed_dir / name, "cpu")
            assert payload["training_task"] == TASK
            assert payload["score_meaning"] == SCORE_MEANING
            assert payload["heldout_test_performed"] is False
            assert payload["external_encoders"] == "official_pretrained_frozen"
            assert payload["data_audit"]["holdout_used"] is False
            assert payload["target_scaling"] == audit["scaling"]
            assert payload["task_status"] == "trained"
            model.eval()
            with torch.inference_mode():
                score = model(train.sample(0))["score"]
                assert score.ndim == 0 and torch.isfinite(score)
            if name == "best_validation.pt":
                expected_best = min(resumed["progress"]["metrics"],
                                    key=lambda item: item["validation"]["standardized_mse"])
                measured = _validate(model, validation, audit["scaling"], SimpleNamespace(reason=None))
                assert_equal(measured, expected_best["validation"], "best_validation_recomputed")
                assert payload["progress"]["best_epoch"] == expected_best["epoch"]
        assert sha256(initial_path) == initial_digest
        checks.append("best_checkpoint_reloads_and_matches_selected_validation_metrics")
        checks.append("affinity_score_contract_frozen_encoders_no_holdout_and_initial_weights_unchanged")

        def rejected(arguments):
            try:
                invoke(arguments)
            except ValueError:
                return
            raise AssertionError("Resume accepted a changed optimizer setting.")

        rejected(["--run-dir", str(resumed_dir), "--resume", "--lr", "0.0002"])
        checks.append("resume_rejects_changed_optimizer_hyperparameters")

    return {"passed": True, "device": "cpu", "threads": 1, "synthetic_only": True,
            "training_task": TASK, "score_meaning": SCORE_MEANING, "checks": checks,
            "continuous_epochs": 2, "continuous_optimizer_steps": 4,
            "pause_after_optimizer_steps": 1, "resumed_total_optimizer_steps": 4,
            "exact_resume_equivalence": True, "nonzero_dropout": True,
            "real_training_performed": False, "real_holdout_read_or_evaluated": False,
            "official_encoders_loaded": False, "deployed_weights_updated": False,
            "temporary_training_artifacts_removed": True}


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--report", type=Path)
    args = arguments.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError(f"Choose a new verification report path: {args.report}")
    report = run_checks()
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
