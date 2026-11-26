"""Resumable PDBbind affinity-supervised score training, without holdout evaluation.

The model architecture is unchanged. MSE learns a unitless standardized pAffinity
score; this is not the strict within-compound preference objective in train.py.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import signal
import time

import torch

from .io import (DEFAULT_CHECKPOINT, allowed_path, load_checkpoint, replace_run_file,
                 save_new, sha256, source_hashes, utc_now)
from .model import ARCHITECTURE, NodeRanker
from .trainer_compatibility import reports_only_upgrade


SCHEMA = "node_rank_pdbbind_score_checkpoint_v1"
INDEX_SCHEMA = "node_rank_pdbbind_feature_index_v1"
TASK = "affinity_supervised_score"
SCORE_MEANING = "unitless_affinity_supervised_score_not_probability"
DEFAULTS = {"epochs": 50, "accumulate": 8, "lr": 1e-4, "weight_decay": 1e-4,
            "grad_clip": 1.0, "seed": 20261118, "device": "cuda", "threads": 4,
            "checkpoint_every": 8, "checkpoint_seconds": 120.0}


def atomic_json(path, payload):
    """Replace only a status/metric file owned by the selected training run."""
    import os
    import tempfile
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verified_tensor_file(path, expected_hash):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"A feature shard checksum changed: {path}")
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


class FeatureIndex:
    """Load one complex at a time, sharing a small CPU residue-feature cache."""

    def __init__(self, path, expected_split, sequence_cache_size=16):
        self.path = allowed_path(path)
        self.sha256 = sha256(self.path)
        self.payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if self.payload.get("schema") != INDEX_SCHEMA or self.payload.get("split") != expected_split:
            raise ValueError(f"Expected a {expected_split} {INDEX_SCHEMA} index; holdout indices are not accepted.")
        self.feature_policy = self.payload.get("feature_policy")
        self.rows = self.payload.get("samples")
        if not isinstance(self.feature_policy, dict) or not self.feature_policy or not isinstance(self.rows, list) or not self.rows:
            raise ValueError("Feature index requires a nonempty feature_policy and samples list.")
        self.sequence_cache_size = sequence_cache_size
        self.sequence_cache = OrderedDict()
        identifiers = set()
        self.paths = []
        for row in self.rows:
            for field in ("pair_id", "compound_id", "target_id", "compound_group"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Feature row requires nonempty {field}.")
            if row["pair_id"] in identifiers:
                raise ValueError("Duplicate pair_id in the feature index.")
            identifiers.add(row["pair_id"])
            label = row.get("p_affinity")
            if isinstance(label, bool) or not isinstance(label, (int, float)) or not math.isfinite(label):
                raise ValueError("p_affinity must be a finite numeric measurement.")
            if row.get("affinity_endpoint") not in ("Kd", "Ki"):
                raise ValueError("This training objective accepts explicitly identified Kd/Ki endpoints only.")
            paths = {}
            for kind in ("sample", "sequence"):
                expected = row.get(kind + "_sha256")
                if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                    raise ValueError(f"Invalid {kind} shard SHA256.")
                paths[kind] = allowed_path(self.path.parent / row[kind + "_path"])
                if not paths[kind].is_file():
                    raise FileNotFoundError(paths[kind])
            self.paths.append(paths)

    def __len__(self):
        return len(self.rows)

    def sample(self, index):
        row, paths = self.rows[index], self.paths[index]
        value = verified_tensor_file(paths["sample"], row["sample_sha256"])
        if not isinstance(value, dict):
            raise ValueError("A sample shard must be a node-feature dictionary.")
        for field in ("pair_id", "compound_id", "target_id"):
            if value.get(field) != row[field]:
                raise ValueError(f"Sample shard identity differs from its index: {field}.")
        key = (str(paths["sequence"]), row["sequence_sha256"])
        if key in self.sequence_cache:
            sequence = self.sequence_cache.pop(key)
        else:
            sequence = verified_tensor_file(paths["sequence"], row["sequence_sha256"])
        self.sequence_cache[key] = sequence
        while len(self.sequence_cache) > self.sequence_cache_size:
            self.sequence_cache.popitem(last=False)
        value = dict(value)
        value["sequence_features"] = sequence
        for field, width in (("sequence_features", 2560), ("ligand_features", 96), ("pocket_features", 96)):
            tensor = value.get(field)
            if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[0] == 0
                    or tensor.shape[1] != width or not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid per-node tensor {field} in {row['pair_id']}.")
        if not isinstance(value.get("protein_sequence"), str) or len(value["protein_sequence"]) != len(sequence):
            raise ValueError("Residue tensor must cover the complete declared protein sequence.")
        return value


def audit_inputs(train, validation):
    if train.feature_policy != validation.feature_policy:
        raise ValueError("Training and validation feature policies differ.")
    for field in ("pair_id", "compound_id", "compound_group"):
        if {row[field] for row in train.rows} & {row[field] for row in validation.rows}:
            raise ValueError(f"Training/validation leakage: overlapping {field}.")
    values = torch.tensor([row["p_affinity"] for row in train.rows], dtype=torch.float64)
    mean, std = float(values.mean()), float(values.std(unbiased=False))
    if len(values) < 2 or not math.isfinite(std) or std <= 1e-8:
        raise ValueError("At least two varying training labels are required for train-only scaling.")
    return {"training_task": TASK, "score_meaning": SCORE_MEANING,
            "train_samples": len(train), "validation_samples": len(validation),
            "train_index_sha256": train.sha256, "validation_index_sha256": validation.sha256,
            "scaling": {"mean": mean, "std": std, "source": "training_labels_only", "std_unbiased": False,
                        "endpoints_pooled": True, "endpoint_specific_normalization": False},
            "train_endpoints": dict(Counter(row["affinity_endpoint"] for row in train.rows)),
            "validation_endpoints": dict(Counter(row["affinity_endpoint"] for row in validation.rows)),
            "mixed_endpoint_limitation": "Kd and Ki are pooled measured pK supervision; they are not identical assay endpoints.",
            "validation_usage": "checkpoint_selection_only", "holdout_used": False,
            "missing_interactions_used_as_negatives": False, "feature_policy": train.feature_policy}


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def rng_state(device):
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else []}


def restore_rng(value, device):
    random.setstate(value["python"])
    torch.set_rng_state(value["torch"].cpu())
    if device == "cuda":
        if len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved RNG state.")
        torch.cuda.set_rng_state_all(value["cuda"])


def load_trained_score(path, device="cpu"):
    """Return the unchanged architecture with its explicit affinity-score contract."""
    payload = torch.load(allowed_path(path), map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("training_task") != TASK
            or payload.get("score_meaning") != SCORE_MEANING or payload.get("architecture") != ARCHITECTURE):
        raise ValueError("Not an affinity-supervised node_rank checkpoint with the required score contract.")
    if payload.get("model_source_sha256") != source_hashes():
        raise ValueError("Model architecture sources differ from the checkpoint.")
    steps = payload.get("training_steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("Invalid training history.")
    scaling = payload.get("target_scaling", {})
    if (scaling.get("source") != "training_labels_only" or not math.isfinite(scaling.get("mean", float("nan")))
            or not math.isfinite(scaling.get("std", float("nan"))) or scaling["std"] <= 0):
        raise ValueError("Invalid train-only target scaling.")
    with torch.random.fork_rng(devices=[]):
        model = NodeRanker(payload["config"])
    for tensor in payload["model_state"].values():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite model checkpoint.")
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device), payload


class PauseRequest:
    def __init__(self):
        self.reason = None
        self.previous = {}

    def __enter__(self):
        def handler(number, frame):
            self.reason = signal.Signals(number).name
        for number in (signal.SIGTERM, signal.SIGINT):
            self.previous[number] = signal.signal(number, handler)
        return self

    def __exit__(self, *args):
        for number, handler in self.previous.items():
            signal.signal(number, handler)


def _payload(model, optimizer, state, audit, args, initial, status):
    return {"schema": SCHEMA, "architecture": ARCHITECTURE, "training_task": TASK,
            "score_meaning": SCORE_MEANING, "score_direction": "higher_is_stronger_predicted_binding",
            "task_status": "trained" if state["steps"] else "untrained", "training_steps": state["steps"],
            "created_at_utc": utc_now(), "status": status, "config": model.config,
            "model_source_sha256": source_hashes(), "trainer_source_sha256": sha256(__file__),
            "model_state": _cpu_copy(model.state_dict()), "optimizer_state": _cpu_copy(optimizer.state_dict()),
            "rng_state": rng_state(args.device), "progress": copy.deepcopy(state),
            "target_scaling": audit["scaling"], "feature_policy": audit["feature_policy"], "data_audit": audit,
            "epochs_requested": args.epochs, "initial_checkpoint_sha256": initial["sha256"],
            "initialization_seed": initial["seed"], "external_encoders": "official_pretrained_frozen",
            "heldout_test_performed": False, "optimization_seed": args.seed}


def _status(state, args, status, **extra):
    return {"status": status, "updated_at_utc": utc_now(), "training_task": TASK,
            "epoch": state["epoch"], "epochs_requested": args.epochs, "phase": state["phase"],
            "training_steps": state["steps"], "samples_seen": state["samples_seen"],
            "epoch_cursor": state["cursor"], "best_validation_mse": state["best_validation_mse"],
            "best_epoch": state["best_epoch"], "holdout_used": False, **extra}


def _validate(model, dataset, scaling, pause):
    model.eval()
    totals = {"squared": 0., "absolute": 0., "count": 0}
    endpoints = {}
    with torch.inference_mode():
        for index, row in enumerate(dataset.rows):
            if pause.reason:
                return None
            score = float(model(dataset.sample(index))["score"].cpu())
            target = (row["p_affinity"] - scaling["mean"]) / scaling["std"]
            squared = (score - target) ** 2
            absolute = abs(score - target)
            if not math.isfinite(squared):
                raise RuntimeError("Nonfinite validation prediction.")
            totals["squared"] += squared
            totals["absolute"] += absolute
            totals["count"] += 1
            value = endpoints.setdefault(row["affinity_endpoint"], {"squared": 0., "absolute": 0., "count": 0})
            value["squared"] += squared
            value["absolute"] += absolute
            value["count"] += 1
    def metrics(value):
        mse = value["squared"] / value["count"]
        return {"samples": value["count"], "standardized_mse": mse,
                "rmse_score": math.sqrt(mse), "mae_score": value["absolute"] / value["count"]}
    return {**metrics(totals), "per_endpoint": {key: metrics(value) for key, value in endpoints.items()}}


def run_training(args):
    run_dir = allowed_path(args.run_dir)
    saved_config = None
    resumed = None
    if args.resume:
        saved_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        resumed = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        for name in ("train_index", "validation_index", "initial_checkpoint"):
            previous = Path(saved_config["arguments"][name])
            supplied = getattr(args, name)
            if supplied is not None and allowed_path(supplied) != previous:
                raise ValueError(f"Resume cannot change {name}.")
            setattr(args, name, previous)
    for name, default in DEFAULTS.items():
        previous = saved_config["arguments"][name] if saved_config else default
        if name == "epochs" and resumed:
            previous = resumed["epochs_requested"]
        supplied = getattr(args, name)
        if supplied is None:
            setattr(args, name, previous)
        elif saved_config and supplied != previous:
            if name != "epochs" or supplied < previous:
                raise ValueError(f"Resume must preserve {name}; only an explicit increase of epochs is supported.")
    if args.initial_checkpoint is None:
        args.initial_checkpoint = DEFAULT_CHECKPOINT
    if args.train_index is None or args.validation_index is None:
        raise ValueError("New training requires --train-index and --validation-index.")
    if min(args.epochs, args.accumulate, args.threads, args.checkpoint_every) < 1:
        raise ValueError("Epochs, gradient accumulation, threads, and checkpoint interval must be positive.")
    if (not 0 < args.lr < 1 or not 0 <= args.weight_decay < 1 or not math.isfinite(args.grad_clip)
            or args.grad_clip <= 0 or not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds <= 0):
        raise ValueError("Invalid optimizer/checkpoint numeric settings.")
    if args.stop_after_updates is not None and args.stop_after_updates < 1:
        raise ValueError("--stop-after-updates must be positive and counts updates in this invocation.")
    torch.set_num_threads(args.threads)
    train, validation = FeatureIndex(args.train_index, "train"), FeatureIndex(args.validation_index, "validation")
    audit = audit_inputs(train, validation)
    public_audit = {key: value for key, value in audit.items() if key != "scaling"}
    print(json.dumps(public_audit, ensure_ascii=False), flush=True)
    if args.dry_run:
        return {"status": "dry_run", "audit": public_audit}
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu explicitly for a new run.")
    if args.resume:
        saved_sha, current_sha = resumed.get("trainer_source_sha256"), sha256(__file__)
        upgrade_reports = reports_only_upgrade(saved_sha, current_sha)
        if (saved_sha != current_sha and not upgrade_reports) or resumed.get("data_audit") != audit:
            raise ValueError("Resume requires a compatible trainer source and identical indices, feature policy, and scaling.")
        model, checkpoint = load_trained_score(run_dir / "last.pt", args.device)
        state = checkpoint["progress"]
        if upgrade_reports:
            for epoch in state["metrics"]:
                validation_report = epoch["validation"]
                for metrics in (validation_report, *validation_report["per_endpoint"].values()):
                    del metrics["rmse_pk"]
                    metrics["rmse_score"] = math.sqrt(metrics["standardized_mse"])
                    metrics["mae_score"] = metrics.pop("mae_pk") / audit["scaling"]["std"]
        initial = {"sha256": checkpoint["initial_checkpoint_sha256"], "seed": checkpoint["initialization_seed"]}
        if state["phase"] == "complete" and state["epoch"] <= args.epochs:
            state["phase"] = "train"
    else:
        if run_dir.exists():
            raise FileExistsError("Run directory exists; choose a new directory or explicitly --resume.")
        model, checkpoint = load_checkpoint(args.initial_checkpoint, args.device)
        if checkpoint["training_steps"] != 0:
            raise ValueError("Start a fresh run from an untrained node_rank checkpoint, or use --resume.")
        initial = {"sha256": sha256(args.initial_checkpoint), "seed": checkpoint["seed"]}
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        state = {"epoch": 1, "phase": "train", "order": None, "cursor": 0, "steps": 0, "samples_seen": 0,
                 "epoch_loss_sum": 0., "epoch_samples": 0, "metrics": [], "best_validation_mse": None, "best_epoch": None}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng(checkpoint["rng_state"], args.device)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        arguments = vars(args).copy()
        for name in ("train_index", "validation_index", "initial_checkpoint", "run_dir"):
            arguments[name] = str(allowed_path(arguments[name]))
        save_new(run_dir / "run_config.json", {"created_at_utc": utc_now(), "arguments": arguments,
                 "audit": audit, "initial_checkpoint_sha256": initial["sha256"],
                 "trainer_source_sha256": sha256(__file__), "model_source_sha256": source_hashes()}, json_file=True)
    start_steps, last_save = state["steps"], time.monotonic()

    def snapshot(status, *, best=False, **extra):
        nonlocal last_save
        payload = _payload(model, optimizer, state, audit, args, initial, status)
        replace_run_file(run_dir / ("best_validation.pt" if best else "last.pt"), payload)
        atomic_json(run_dir / "status.json", _status(state, args, status, **extra))
        last_save = time.monotonic()

    snapshot("running")
    with PauseRequest() as pause:
        while state["epoch"] <= args.epochs:
            if state["phase"] == "train":
                model.train()
                if state["order"] is None:
                    state["order"] = list(range(len(train)))
                    random.shuffle(state["order"])
                while state["cursor"] < len(state["order"]):
                    if pause.reason:
                        snapshot("paused", reason=pause.reason)
                        return _status(state, args, "paused", reason=pause.reason)
                    chosen = state["order"][state["cursor"]:state["cursor"] + args.accumulate]
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum = 0.
                    for index in chosen:
                        score = model(train.sample(index))["score"]
                        target = (train.rows[index]["p_affinity"] - audit["scaling"]["mean"]) / audit["scaling"]["std"]
                        loss = (score - target).square()
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite training loss; resume the last complete update.")
                        (loss / len(chosen)).backward()
                        loss_sum += float(loss.detach().cpu())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                    optimizer.step()
                    state["steps"] += 1
                    state["cursor"] += len(chosen)
                    state["samples_seen"] += len(chosen)
                    state["epoch_loss_sum"] += loss_sum
                    state["epoch_samples"] += len(chosen)
                    if args.stop_after_updates is not None and state["steps"] - start_steps >= args.stop_after_updates:
                        pause.reason = "stop_after_updates"
                    due = state["steps"] % args.checkpoint_every == 0 or time.monotonic() - last_save >= args.checkpoint_seconds
                    if due or pause.reason:
                        snapshot("paused" if pause.reason else "running", reason=pause.reason)
                    else:
                        atomic_json(run_dir / "status.json", _status(state, args, "running"))
                    if pause.reason:
                        return _status(state, args, "paused", reason=pause.reason)
                state["phase"] = "validation"
                snapshot("running")
            if state["phase"] == "validation":
                validation_metrics = _validate(model, validation, audit["scaling"], pause)
                if validation_metrics is None:
                    snapshot("paused", reason=pause.reason)
                    return _status(state, args, "paused", reason=pause.reason)
                metric = {"epoch": state["epoch"], "training_steps": state["steps"],
                          "train_online_standardized_mse": state["epoch_loss_sum"] / state["epoch_samples"],
                          "train_samples": state["epoch_samples"], "validation": validation_metrics}
                best = state["best_validation_mse"] is None or validation_metrics["standardized_mse"] < state["best_validation_mse"]
                if best:
                    state["best_validation_mse"] = validation_metrics["standardized_mse"]
                    state["best_epoch"] = state["epoch"]
                state["metrics"].append(metric)
                state.update(epoch=state["epoch"] + 1, phase="train", order=None, cursor=0, epoch_loss_sum=0., epoch_samples=0)
                if best:
                    snapshot("running", best=True)
                snapshot("running")
                atomic_json(run_dir / "metrics.json", state["metrics"])
                print(json.dumps(metric, ensure_ascii=False), flush=True)
        state["phase"] = "complete"
        if sha256(train.path) != train.sha256 or sha256(validation.path) != validation.sha256:
            raise RuntimeError("A dataset index changed during training; invalidate this run.")
        snapshot("complete")
        summary = {**_status(state, args, "complete"), "completed_epochs": len(state["metrics"]),
                   "initial_checkpoint_sha256": initial["sha256"], "last_checkpoint_sha256": sha256(run_dir / "last.pt"),
                   "best_validation_checkpoint_sha256": sha256(run_dir / "best_validation.pt"),
                   "score_meaning": SCORE_MEANING, "validation_metric_units": "raw_score",
                   "external_encoders_updated": False, "heldout_test_performed": False,
                   "validation_is_selection_not_final_test": True}
        atomic_json(run_dir / "training_summary.json", summary)
        return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--train-index", type=Path)
    result.add_argument("--validation-index", type=Path)
    result.add_argument("--initial-checkpoint", type=Path)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--stop-after-updates", type=int, help="Pause after this many complete updates in this invocation.")
    for name in ("epochs", "accumulate", "seed", "threads", "checkpoint_every"):
        result.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("lr", "weight_decay", "grad_clip", "checkpoint_seconds"):
        result.add_argument("--" + name.replace("_", "-"), type=float)
    result.add_argument("--device", choices=("cpu", "cuda"))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    result = run_training(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
