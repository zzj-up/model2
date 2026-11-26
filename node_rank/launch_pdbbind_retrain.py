"""Durable local preparation/training worker with explicit pause/resume controls.

The worker calls local Python only. It does not call an LLM or inspect Codex
account allowance. A timed pause is an explicit local control, not a quota hook.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .prepare_pdbbind import write_json

ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/dti/.venvs/dti/bin/python")
PBC_PYTHON = Path("/home/dti/.venvs/pbcnet_compat/bin/python")


def now():
    return datetime.now(timezone.utc).isoformat()


def control(job):
    path = job / "control.json"
    return json.loads(path.read_text()) if path.exists() else {"state": "run"}


def phases(config):
    dataset = Path(config["dataset"])
    cache = Path(config["cache"])
    job = Path(config["job"])
    audit = dataset / "pbc_audit"
    splits = Path(config.get("split_dir", dataset / "splits"))
    indexes = dataset / "feature_indexes"
    esm_audit = Path(config.get("esm_audit", dataset / "esm_audit.json"))
    def command(executable, module, *arguments):
        return [str(executable), "-u", "-m", module, *map(str, arguments)]
    return [
        ("pbc_atoms", command(PBC_PYTHON, "node_rank.pdbbind_features", "pbc", "--manifest", dataset / "candidate_manifest.json",
                              "--cache-dir", cache, "--audit-dir", audit,
                              "--workers", config.get("pbc_workers", 2)), audit / "pbc_index.json"),
        ("lock_split", command(PBC_PYTHON, "node_rank.prepare_pdbbind", "finalize", "--manifest", dataset / "candidate_manifest.json",
                               "--pbc-index", audit / "pbc_index.json", "--output", splits, "--seed", config["seed"]), splits / "SPLIT_LOCK.json"),
        ("esm_residues", command(PYTHON, "node_rank.pdbbind_features", "esm", "--train-manifest", splits / "train_manifest.json",
                                 "--validation-manifest", splits / "validation_manifest.json", "--cache-dir", cache,
                                 "--audit-output", esm_audit), esm_audit),
        ("train_feature_index", command(PBC_PYTHON, "node_rank.pdbbind_features", "index", "--manifest", splits / "train_manifest.json",
                                       "--split", "train", "--output", indexes / "train_features.json",
                                       "--pbc-index", audit / "pbc_index.json", "--esm-audit", esm_audit), indexes / "train_features.json"),
        ("validation_feature_index", command(PBC_PYTHON, "node_rank.pdbbind_features", "index", "--manifest", splits / "validation_manifest.json",
                                            "--split", "validation", "--output", indexes / "validation_features.json",
                                            "--pbc-index", audit / "pbc_index.json", "--esm-audit", esm_audit), indexes / "validation_features.json"),
        ("training", command(PYTHON, "node_rank.train_pdbbind", "--train-index", indexes / "train_features.json",
                             "--validation-index", indexes / "validation_features.json", "--run-dir", job / "model",
                             "--epochs", config["epochs"], "--accumulate", config["accumulate"], "--lr", config["lr"],
                             "--seed", config["seed"], "--device", "cuda"), job / "model/training_summary.json"),
    ]


def worker(config_path):
    config = json.loads(config_path.read_text())
    job = config_path.parent
    lock_stream = (job / "worker.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another worker already owns this job.")
    status_path = job / "job_status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    completed = set(previous.get("completed_phases", []))
    state = {"status": "starting", "worker_pid": os.getpid(), "created_at_utc": now(),
             "completed_phases": sorted(completed), "dataset": config["dataset"],
             "training_task": "affinity_supervised_score", "holdout_evaluation_performed": False,
             "codex_quota_detection_supported": False, "runs_independently_of_conversation": True}

    def update(**changes):
        state.update(changes, updated_at_utc=now(), completed_phases=sorted(completed))
        write_json(status_path, state)

    def await_run():
        while True:
            instruction = control(job)
            if instruction.get("state") != "pause":
                return
            resume_at = instruction.get("resume_at_unix")
            if resume_at is not None and time.time() >= resume_at:
                write_json(job / "control.json", {"state": "run", "reason": "explicit_timed_pause_elapsed", "updated_at_utc": now()})
                return
            update(status="paused", pause_until_unix=resume_at)
            time.sleep(2)

    try:
        for name, base_command, expected in phases(config):
            if name in completed:
                if not expected.exists():
                    raise RuntimeError(f"A completed phase lost its output: {name}: {expected}")
                continue
            while True:
                await_run()
                command = list(base_command)
                if name == "training" and (job / "model/last.pt").exists():
                    command.append("--resume")
                update(status="running", phase=name, command=command, phase_started_at_utc=now())
                environment = dict(os.environ, TOKENIZERS_PARALLELISM="false", DGLBACKEND="pytorch", DGL_FFI="ctypes",
                                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                with (job / (name + ".log")).open("a", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                               stderr=subprocess.STDOUT, env=environment)
                    update(phase_pid=process.pid)
                    pause_sent = False
                    while process.poll() is None:
                        if control(job).get("state") == "pause" and not pause_sent:
                            process.send_signal(signal.SIGTERM)
                            pause_sent = True
                            update(status="pausing", phase=name)
                        time.sleep(2)
                    code = process.returncode
                if pause_sent:
                    update(status="paused", phase_exit_code=code, phase_pid=None)
                    continue
                if code != 0:
                    raise RuntimeError(f"Phase {name} exited with {code}; inspect {name}.log. Resume only after fixing its cause.")
                if name == "training":
                    trainer_status = json.loads((job / "model/status.json").read_text())
                    if trainer_status.get("status") != "complete":
                        raise RuntimeError("Trainer returned without a completed status.")
                if not expected.exists():
                    raise RuntimeError(f"Phase returned without its promised output: {expected}")
                completed.add(name)
                update(status="phase_completed", phase_exit_code=code, phase_pid=None)
                break
        update(status="completed", finished_at_utc=now(), phase="complete")
    except BaseException as error:
        update(status="failed", error_type=type(error).__name__, error=str(error), finished_at_utc=now())
        raise


def launch(job):
    with (job / "worker.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([str(PYTHON), "-u", "-m", "node_rank.launch_pdbbind_retrain", "worker", "--job-dir", str(job)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(job / "launch.json", {"worker_pid": process.pid, "launched_at_utc": now()})
    print(json.dumps({"launched": True, "worker_pid": process.pid, "job_dir": str(job)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    start = sub.add_parser("start")
    start.add_argument("--dataset", type=Path, required=True)
    start.add_argument("--cache", type=Path, required=True)
    start.add_argument("--job-dir", type=Path, required=True)
    start.add_argument("--epochs", type=int, default=50)
    start.add_argument("--accumulate", type=int, default=8)
    start.add_argument("--lr", type=float, default=1e-4)
    start.add_argument("--seed", type=int, default=20261118)
    start.add_argument("--pbc-workers", type=int, default=2)
    for action in ("worker", "status", "pause", "resume"):
        tool = sub.add_parser(action)
        tool.add_argument("--job-dir", type=Path, required=True)
        if action == "pause":
            tool.add_argument("--hours", type=float)
    args = parser.parse_args()
    job = args.job_dir.resolve()
    if args.action == "start":
        if args.epochs < 1 or args.accumulate < 1 or args.pbc_workers < 1 or not 0 < args.lr < 1:
            parser.error("Invalid training settings.")
        if not (args.dataset / "candidate_manifest.json").is_file():
            raise ValueError("Complete structural scan before launching.")
        job.mkdir(parents=True, exist_ok=False)
        write_json(job / "job_config.json", {"dataset": str(args.dataset.resolve()), "cache": str(args.cache.resolve()),
                   "job": str(job), "epochs": args.epochs, "accumulate": args.accumulate, "lr": args.lr,
                   "seed": args.seed, "pbc_workers": args.pbc_workers,
                   "created_at_utc": now(), "quota_autodetection": False})
        write_json(job / "control.json", {"state": "run"})
        launch(job)
    elif args.action == "worker":
        worker(job / "job_config.json")
    elif args.action == "status":
        print((job / "job_status.json").read_text())
    elif args.action == "pause":
        if args.hours is not None and (not 0 < args.hours <= 168):
            parser.error("Timed pause must be positive and <=168 hours.")
        write_json(job / "control.json", {"state": "pause", "resume_at_unix":
                   time.time() + args.hours * 3600 if args.hours is not None else None, "requested_at_utc": now()})
        print(json.dumps({"pause_requested": True, "job": str(job), "resume_after_hours": args.hours}))
    elif args.action == "resume":
        write_json(job / "control.json", {"state": "run", "requested_at_utc": now()})
        # Nonblocking file lock determines whether a living worker already owns
        # this job. Never signal a PID from an old PID file.
        with (job / "worker.lock").open("a+") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps({"resume_requested": True, "existing_worker": True}))
                return
        launch(job)


if __name__ == "__main__":
    main()
