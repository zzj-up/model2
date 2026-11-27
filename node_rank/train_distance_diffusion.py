"""Train a fresh pose-independent distance diffusion model on locked PDBbind splits."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid

import torch

from . import distance_data
from .distance_diffusion import DISTANCE_ARCHITECTURE, DISTANCE_CONFIG, DistanceDiffusion
from .io import allowed_path, replace_run_file, save_new, sha256, source_hashes, utc_now
from .train_pdbbind import PauseRequest, _cpu_copy, atomic_json, restore_rng, rng_state, verified_tensor_file


SCHEMA = "node_rank_distance_diffusion_checkpoint_v1"
SCORE_MEANING = "unitless_affinity_supervised_score_not_probability"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = ROOT.parent / "inputs/pdbbind2020_retrain_20261118/dataset/splits_v2"
DEFAULTS = {
    "epochs": 50, "accumulate": 8, "seed": 20261126, "device": "cuda", "threads": 4,
    "lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1., "score_weight": .2,
    "checkpoint_every": 8, "checkpoint_seconds": 120., "progress_every": 64,
    "sample_validation": 8, "sampling_steps": 15,
}


def _sources():
    return source_hashes() | {"distance_diffusion.py": sha256(Path(__file__).with_name("distance_diffusion.py"))}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _checkpoint_fields(payload):
    if (not isinstance(payload, dict) or payload.get("schema") != SCHEMA
            or payload.get("architecture") != DISTANCE_ARCHITECTURE
            or payload.get("model_source_sha256") != _sources()
            or payload.get("score_meaning") != SCORE_MEANING
            or payload.get("score_direction") != "higher_is_stronger_predicted_binding"
            or type(payload.get("training_steps")) is not int or payload["training_steps"] < 0):
        raise ValueError("Wrong diffusion checkpoint schema, architecture, score contract or source revision.")
    scale = payload.get("target_scaling", {})
    if (scale.get("source") != "training_labels_only" or not math.isfinite(scale.get("mean", math.nan))
            or not math.isfinite(scale.get("std", math.nan)) or scale["std"] <= 0):
        raise ValueError("Invalid checkpoint target scaling.")
    return payload


def load_trained_distance(path, device="cpu"):
    """Safely restore a trained Model 2 without perturbing the caller's random state."""
    payload = _checkpoint_fields(torch.load(allowed_path(path), map_location="cpu", weights_only=True))
    if payload["training_steps"] == 0:
        raise ValueError("The distance checkpoint has no trained updates yet.")
    with torch.random.fork_rng(devices=[]):
        model = DistanceDiffusion(payload["config"])
    if any(not torch.isfinite(value).all() for value in payload["model_state"].values() if value.is_floating_point()):
        raise ValueError("Distance checkpoint contains non-finite parameters.")
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval(), payload


class LockedDistanceSplit:
    def __init__(self, data, split_dir, lock, name):
        self.name = name
        self.directory = data / name
        self.dataset = distance_data.DistanceDataset(self.directory)
        self.index = self.directory / "index.json"
        self.index_hash = sha256(self.index)
        self.report = report = self.dataset.manifest
        self.manifest = split_dir / f"{name}_manifest.json"
        self.manifest_hash = sha256(self.manifest)
        source = lock["partitions"][name]
        if (self.manifest_hash != source["sha256"] or report.get("manifest_sha256") != self.manifest_hash
                or report.get("requested_count") != source["pairs"]
                or report.get("feature_source_sha256") != sha256(Path(distance_data.__file__))):
            raise ValueError(f"The {name} cache must use the complete locked manifest and current feature source.")
        locked_rows = _json(self.manifest)
        self.rows_by_id = {row["pair_id"]: row for row in locked_rows}
        self.groups = {row["compound_group"] for row in locked_rows}
        ids = [row["pair_id"] for row in report["samples"] + report["failures"]]
        if len(ids) != len(locked_rows) or len(set(ids)) != len(ids) or set(ids) != set(self.rows_by_id):
            raise ValueError(f"The {name} cache omits or repeats records from the locked split.")
        self.policy = {key: value for key, value in report.items() if key not in {
            "manifest_sha256", "requested_count", "samples", "failures",
            "proposal_within_4_angstrom", "proposal_within_8_angstrom",
        }}
        self.rows, self.files = [], []
        for saved in self.dataset.samples:
            filename = saved["file"]
            if Path(filename).name != filename or not filename.endswith(".pt"):
                raise ValueError(f"Unsafe sample filename in the {name} index.")
            path = self.directory / filename
            self.rows.append(self.rows_by_id[saved["pair_id"]])
            self.files.append((path, sha256(path)))
        payload_key = hashlib.sha256("\n".join(digest for _, digest in self.files).encode()).hexdigest()
        self.audit = {"requested": len(ids), "used": len(self.rows), "failed": len(report["failures"]),
                      "manifest_sha256": self.manifest_hash, "index_sha256": self.index_hash,
                      "sample_payloads_sha256": payload_key,
                      "proposal_within_4_angstrom": report["proposal_within_4_angstrom"],
                      "proposal_within_8_angstrom": report["proposal_within_8_angstrom"]}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, digest = self.files[index]
        sample, row = verified_tensor_file(path, digest), self.rows[index]
        if (any(sample.get(key) != row[key] for key in ("pair_id", "compound_group", "target_id"))
                or sample.get("source_sha256") != [row["protein_pdb_sha256"], row["ligand_sdf_sha256"]]
                or not math.isclose(float(sample["affinity"]), row["p_affinity"], rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError(f"Cached sample differs from the locked record {row['pair_id']}.")
        return sample

    def assert_unchanged(self):
        if sha256(self.index) != self.index_hash or sha256(self.manifest) != self.manifest_hash:
            raise RuntimeError(f"The {self.name} index or locked manifest changed during training.")


def _splits(data, split_dir):
    lock_path = split_dir / "SPLIT_LOCK.json"
    lock = _json(lock_path)
    if lock.get("schema") != "pdbbind_compound_group_split_v2":
        raise ValueError("A locked compound-disjoint PDBbind v2 split is required.")
    train, valid = [LockedDistanceSplit(data, split_dir, lock, name) for name in ("train", "validation")]
    if train.policy != valid.policy or train.groups & valid.groups or train.rows_by_id.keys() & valid.rows_by_id.keys():
        raise ValueError("Feature policy mismatch or train/validation data overlap.")
    labels = torch.tensor([row["p_affinity"] for row in train.rows], dtype=torch.float64)
    mean, std = float(labels.mean()), float(labels.std(unbiased=False))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 1e-8:
        raise ValueError("At least two different finite training labels are required for the score task.")
    audit = {"train": train.audit, "validation": valid.audit, "feature_policy": train.policy,
             "split_lock_sha256": sha256(lock_path), "holdout_used": False,
             "validation_pocket_policy": "nearest_receptor_only_proposal_to_native_ligand_center",
             "negative_interaction_labels_used": False}
    scale = {"mean": mean, "std": std, "source": "training_labels_only", "std_unbiased": False}
    return train, valid, audit, scale


def _seed(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def _score_target(row, scale):
    return (row["p_affinity"] - scale["mean"]) / scale["std"]


def _validation(model, dataset, scale, args, pause, report_progress):
    model.eval()
    indices = set(random.Random(args.seed + 101).sample(range(len(dataset)), min(len(dataset), args.sample_validation)))
    noise_sum = score_sum = score_absolute = 0.
    sampled_distance_errors, sampled_contact_errors = [], []
    with torch.inference_mode():
        for index, row in enumerate(dataset.rows):
            if pause.reason:
                return None
            sample = dataset[index]
            truth = sample["target_distances"]
            result = model.training_loss(sample, truth, generator=_seed(args.seed + 1000 + index))
            error = float(result["score"]) - _score_target(row, scale)
            noise = float(result["distance_loss"])
            if not math.isfinite(noise) or not math.isfinite(error):
                raise RuntimeError(f"Non-finite validation loss for {row['pair_id']}.")
            noise_sum += noise
            score_sum += error ** 2
            score_absolute += abs(error)
            if index in indices:
                predicted = model.sample_distances(sample, num_steps=args.sampling_steps,
                                                   generator=_seed(args.seed + 50000 + index))["distances"].cpu()
                if predicted.shape != truth.shape or not torch.isfinite(predicted).all():
                    raise RuntimeError(f"Invalid sampled distances for {row['pair_id']}.")
                clipped_error = (predicted - truth.clamp(max=model.config["distance_cap"])).abs()
                sampled_distance_errors.append(float(clipped_error.mean()))
                contacts = truth <= 5.
                if contacts.any():
                    sampled_contact_errors.append(float(clipped_error[contacts].mean()))
            if (index + 1) % args.progress_every == 0 or index + 1 == len(dataset):
                report_progress(index + 1)
    n = len(dataset)
    noise_mse, score_mse = noise_sum / n, score_sum / n
    return {"samples": n, "distance_noise_mse": noise_mse, "score_mse": score_mse,
            "score_mae": score_absolute / n, "combined_loss": noise_mse + args.score_weight * score_mse,
            "ddim_steps": args.sampling_steps, "sampled_complexes": len(sampled_distance_errors),
            "sampled_distance_mae_angstrom_clipped": (sum(sampled_distance_errors) / len(sampled_distance_errors)
                                                      if sampled_distance_errors else None),
            "sampled_contact_complexes": len(sampled_contact_errors),
            "sampled_contact_mae_angstrom_within_5_angstrom": (sum(sampled_contact_errors) / len(sampled_contact_errors)
                                                              if sampled_contact_errors else None)}


@contextmanager
def _exclusive_run(path):
    # The training runtime is WSL. Its advisory lock is released even after SIGKILL.
    import fcntl

    with (path / ".trainer.lock").open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another trainer already owns {path}.") from error
        yield


def _settings(args, config):
    selected = {}
    for key, default in {"data": None, "split_dir": DEFAULT_SPLIT_DIR, **DEFAULTS}.items():
        supplied = getattr(args, key)
        recorded = config["arguments"][key] if config else default
        if key in {"data", "split_dir"} and supplied is not None:
            supplied = str(allowed_path(supplied))
        value = recorded if supplied is None else supplied
        if value is None:
            raise ValueError("A new run requires --data with train/ and validation/ distance caches.")
        if key in {"data", "split_dir"}:
            value = str(allowed_path(value))
        if config and value != recorded and not (key == "epochs" and value >= recorded):
            raise ValueError(f"Resume cannot change {key}; only increasing the epoch budget is supported.")
        selected[key] = value
        setattr(args, key, value)
    if (min(args.epochs, args.accumulate, args.threads, args.checkpoint_every, args.progress_every) < 1
            or not 1 <= args.sampling_steps <= DISTANCE_CONFIG["diffusion_steps"] or args.sample_validation < 0
            or not 0 <= args.seed < 2**31):
        raise ValueError("Invalid training, sampling, progress or seed option.")
    floats = [args.lr, args.weight_decay, args.grad_clip, args.score_weight, args.checkpoint_seconds]
    if (not all(map(math.isfinite, floats)) or not 0 < args.lr < 1 or not 0 <= args.weight_decay < 1
            or args.grad_clip <= 0 or args.score_weight < 0 or args.checkpoint_seconds <= 0):
        raise ValueError("Invalid optimizer or checkpoint option.")
    if args.stop_after_updates is not None and args.stop_after_updates < 1:
        raise ValueError("--stop-after-updates must be positive.")
    return selected


def _train(args, path, arguments, train, valid, audit, scale, config):
    source, trainer_hash = _sources(), sha256(Path(__file__))
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; launch a new run with --device cpu to use the CPU.")
    if args.resume:
        checkpoint = _checkpoint_fields(torch.load(path / "last.pt", map_location="cpu", weights_only=True))
        if (checkpoint.get("run_id") != config.get("run_id") or checkpoint.get("data_audit") != audit
                or checkpoint.get("trainer_source_sha256") != trainer_hash
                or checkpoint.get("arguments") != config["arguments"] or args.epochs < checkpoint["epochs_requested"]):
            raise ValueError("Resume requires this run's own last checkpoint, unchanged cached data and the same trainer.")
        model = DistanceDiffusion(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        state = checkpoint["progress"]
        if state["phase"] == "complete" and state["epoch"] <= args.epochs:
            state["phase"] = "train"
    else:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = DistanceDiffusion()
        config = {"run_id": uuid.uuid4().hex, "arguments": arguments, "data_audit": audit,
                  "trainer_source_sha256": trainer_hash, "model_source_sha256": source,
                  "architecture": DISTANCE_ARCHITECTURE, "created_at_utc": utc_now()}
        save_new(path / "run_config.json", config, json_file=True)
        state = {"epoch": 1, "phase": "train", "cursor": 0, "steps": 0, "samples_seen": 0,
                 "noise_sum": 0., "score_sum": 0., "best_epoch": None, "best_loss": None, "metrics": []}
    if args.sampling_steps > model.config["diffusion_steps"]:
        raise ValueError("Validation sampling steps exceed model diffusion schedule.")
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng(checkpoint["rng_state"], args.device)
        del checkpoint
    base_steps, last_save = state["steps"], time.monotonic()

    def status(stage, **extra):
        value = {"status": stage, "updated_at_utc": utc_now(), "architecture": DISTANCE_ARCHITECTURE,
                 "phase": state["phase"], "epoch": state["epoch"], "epochs_requested": args.epochs,
                 "training_steps": state["steps"], "samples_seen": state["samples_seen"],
                 "completed_epochs": len(state["metrics"]),
                 "epoch_cursor": state["cursor"], "train_samples": len(train), "validation_samples": len(valid),
                 "best_epoch": state["best_epoch"], "best_validation_combined_loss": state["best_loss"],
                 "holdout_used": False, **extra}
        atomic_json(path / "training_status.json", value)
        return value

    def snapshot(stage, *, best=False, **extra):
        nonlocal last_save
        payload = {"schema": SCHEMA, "architecture": DISTANCE_ARCHITECTURE, "run_id": config["run_id"],
                   "training_task": "distance_diffusion_and_affinity_score", "score_meaning": SCORE_MEANING,
                   "score_direction": "higher_is_stronger_predicted_binding", "score_trained": args.score_weight > 0 and state["steps"] > 0,
                   "training_steps": state["steps"], "epochs_requested": args.epochs, "progress": state,
                   "config": model.config, "target_scaling": scale, "feature_policy": audit["feature_policy"],
                   "data_audit": audit, "arguments": config["arguments"], "model_source_sha256": source,
                   "trainer_source_sha256": trainer_hash, "model_state": _cpu_copy(model.state_dict()),
                   "optimizer_state": _cpu_copy(optimizer.state_dict()), "rng_state": rng_state(args.device),
                   "saved_at_utc": utc_now(), "holdout_used": False}
        if best:
            replace_run_file(path / "best_validation.pt", payload)
        replace_run_file(path / "last.pt", payload)
        last_save = time.monotonic()
        return status(stage, **extra)

    snapshot("running")
    with PauseRequest() as pause:
        try:
            while state["epoch"] <= args.epochs:
                if state["phase"] == "train":
                    model.train()
                    order = list(range(len(train)))
                    random.Random(args.seed + state["epoch"]).shuffle(order)
                    while state["cursor"] < len(order):
                        if pause.reason:
                            return snapshot("paused", reason=pause.reason)
                        chosen = order[state["cursor"]:state["cursor"] + args.accumulate]
                        optimizer.zero_grad(set_to_none=True)
                        noise_sum = score_sum = 0.
                        for index in chosen:
                            sample = train[index]
                            conformers = sample["ligand_conformers"]
                            sample["ligand_pos"] = conformers[random.randrange(len(conformers))]
                            result = model.training_loss(sample, sample["target_distances"])
                            distance_loss = result["distance_loss"]
                            score_loss = (result["score"] - _score_target(train.rows[index], scale)).square()
                            loss = distance_loss + args.score_weight * score_loss
                            if not torch.isfinite(loss):
                                raise RuntimeError(f"Non-finite training loss for {train.rows[index]['pair_id']}.")
                            (loss / len(chosen)).backward()
                            noise_sum += float(distance_loss.detach())
                            score_sum += float(score_loss.detach())
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                        optimizer.step()
                        state["steps"] += 1
                        state["cursor"] += len(chosen)
                        state["samples_seen"] += len(chosen)
                        state["noise_sum"] += noise_sum
                        state["score_sum"] += score_sum
                        if args.stop_after_updates and state["steps"] - base_steps >= args.stop_after_updates:
                            pause.reason = "stop_after_updates"
                        if pause.reason:
                            return snapshot("paused", reason=pause.reason)
                        if state["steps"] % args.checkpoint_every == 0 or time.monotonic() - last_save >= args.checkpoint_seconds:
                            snapshot("running")
                        if state["cursor"] % args.progress_every < len(chosen):
                            print(json.dumps(status("running")), flush=True)
                    state["phase"] = "validation"
                    snapshot("running", validation_cursor=0)
                if state["phase"] != "validation":
                    raise ValueError(f"Unexpected training phase {state['phase']!r}.")
                report = _validation(model, valid, scale, args, pause,
                                     lambda count: status("running", validation_cursor=count))
                if report is None:
                    return snapshot("paused", reason=pause.reason)
                train.assert_unchanged()
                valid.assert_unchanged()
                noise, score = state["noise_sum"] / len(train), state["score_sum"] / len(train)
                record = {"epoch": state["epoch"], "training_steps": state["steps"],
                          "train": {"samples": len(train), "distance_noise_mse": noise, "score_mse": score,
                                    "combined_loss": noise + args.score_weight * score}, "validation": report}
                best = state["best_loss"] is None or report["combined_loss"] < state["best_loss"]
                if best:
                    state["best_loss"], state["best_epoch"] = report["combined_loss"], state["epoch"]
                state["metrics"].append(record)
                state.update(epoch=state["epoch"] + 1, phase="train", cursor=0, noise_sum=0., score_sum=0.)
                snapshot("running", best=best)
                atomic_json(path / "metrics.json", state["metrics"])
                print(json.dumps(record), flush=True)
            state["phase"] = "complete"
            summary = snapshot("complete") | {"completed_epochs": len(state["metrics"]), "data_audit": audit,
                                               "best_checkpoint_sha256": sha256(path / "best_validation.pt"),
                                               "last_checkpoint_sha256": sha256(path / "last.pt")}
            atomic_json(path / "training_summary.json", summary)
            return summary
        except BaseException as error:
            status("error", reason=f"{type(error).__name__}: {error}", resume_from="last.pt")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Directory containing train and validation distance caches.")
    parser.add_argument("--split-dir", type=Path, help="Locked PDBbind v2 manifest directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Verify the split and cache without initializing training.")
    parser.add_argument("--stop-after-updates", type=int, help="Pause after this many complete updates in this process.")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    for key, default in DEFAULTS.items():
        if key != "device":
            parser.add_argument("--" + key.replace("_", "-"), type=type(default))
    args = parser.parse_args(argv)
    path = allowed_path(args.run_dir)
    config = _json(path / "run_config.json") if args.resume else None
    if args.resume and args.epochs is None:
        checkpoint = torch.load(path / "last.pt", map_location="cpu", weights_only=True, mmap=True)
        args.epochs = _checkpoint_fields(checkpoint)["epochs_requested"]
    if not args.resume and path.exists():
        raise FileExistsError("Choose a new run directory or resume the exact distance diffusion run.")
    arguments = _settings(args, config)
    train, valid, audit, scale = _splits(Path(args.data), Path(args.split_dir))
    print(json.dumps({"stage": "data_verified", **audit}), flush=True)
    if not args.dry_run:
        if not args.resume:
            path.mkdir(parents=True, exist_ok=False)
        with _exclusive_run(path):
            result = _train(args, path, arguments, train, valid, audit, scale, config)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
