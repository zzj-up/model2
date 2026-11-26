"""Synthetic integration checks for the resumable PDBbind launcher.

Uses temporary files and mocked stage execution. No encoder is loaded, no
training is performed, and no real dataset or held-out manifest is opened.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from . import launch_pdbbind_retrain as launcher
from . import pdbbind_features as features
from .prepare_pdbbind import write_json


def configuration(directory):
    return {
        "dataset": str(directory / "dataset"), "cache": str(directory / "cache"),
        "job": str(directory / "job"), "epochs": 3, "accumulate": 2,
        "lr": 1e-4, "seed": 20261118,
    }


def verify_phase_arguments(config):
    """Exercise each real CLI parser while replacing its expensive handler."""
    checked = []
    for name, command, expected in launcher.phases(config):
        module = importlib.import_module(command[3])
        argv = command[4:]
        if command[3] == "node_rank.pdbbind_features":
            handler = {"pbc": "pbc_stage", "esm": "esm_stage", "index": "index_stage"}[argv[0]]
        elif command[3] == "node_rank.prepare_pdbbind":
            handler = "finalize"
        elif command[3] == "node_rank.train_pdbbind":
            handler = "run_training"
        else:
            raise AssertionError(f"Unexpected launcher stage module: {command[3]}")
        with patch.object(module, handler, return_value={"synthetic_only": True}) as stage:
            with patch.object(sys, "argv", [command[3], *argv]), contextlib.redirect_stdout(io.StringIO()):
                module.main()
        stage.assert_called_once()
        parsed = stage.call_args.args[0]
        assert Path(expected).is_relative_to(Path(config["dataset"])) or name == "training"
        if name == "esm_residues":
            assert parsed.audit_output == expected
        if name.endswith("feature_index"):
            assert parsed.output == expected
            assert parsed.pbc_index == Path(config["dataset"]) / "pbc_audit/pbc_index.json"
            assert parsed.esm_audit == Path(config["dataset"]) / "esm_audit.json"
        if name == "training":
            assert parsed.epochs == config["epochs"] and parsed.device == "cuda"
            assert parsed.accumulate == config["accumulate"] and parsed.lr == config["lr"]
        checked.append(name)
    return checked


def verify_worker(directory, trainer_status="complete"):
    config = configuration(directory)
    job = Path(config["job"])
    job.mkdir(parents=True)
    config_path = job / "job_config.json"
    write_json(config_path, config)
    write_json(job / "control.json", {"state": "run"})
    phase_list = launcher.phases(config)
    commands = {tuple(command): (name, expected) for name, command, expected in phase_list}
    seen = []

    class FinishedStage:
        returncode = 0
        pid = 999999

        def __init__(self, command, **kwargs):
            name, expected = commands[tuple(command)]
            seen.append(name)
            assert kwargs["stdin"] == launcher.subprocess.DEVNULL
            assert kwargs["env"]["DGLBACKEND"] == "pytorch"
            write_json(expected, {"synthetic_phase": name})
            if name == "training":
                write_json(job / "model/status.json", {"status": trainer_status})

        def poll(self):
            return self.returncode

    with patch.object(launcher.subprocess, "Popen", FinishedStage):
        if trainer_status == "complete":
            launcher.worker(config_path)
        else:
            try:
                launcher.worker(config_path)
            except RuntimeError as error:
                assert "without a completed status" in str(error)
            else:
                raise AssertionError("Worker accepted an uncompleted trainer.")
    status = json.loads((job / "job_status.json").read_text())
    assert seen == [name for name, _, _ in phase_list]
    if trainer_status != "complete":
        assert status["status"] == "failed"
        assert "training" not in status["completed_phases"]
        return
    assert status["status"] == "completed" and status["phase"] == "complete"
    assert set(status["completed_phases"]) == set(seen)
    assert status["holdout_evaluation_performed"] is False
    with patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("Completed stage relaunched")):
        launcher.worker(config_path)
    assert json.loads((job / "job_status.json").read_text())["status"] == "completed"


def verify_index_idempotence(directory):
    directory.mkdir()
    record = {
        "pair_id": "synthetic_pair", "compound_id": "synthetic_compound",
        "target_id": "synthetic_target", "compound_group": "synthetic_parent",
        "p_affinity": 7.0, "affinity_endpoint": "Kd",
    }
    manifest = directory / "train_manifest.json"
    sample = directory / "synthetic_sample.pt"
    sequence = directory / "synthetic_sequence.pt"
    write_json(manifest, [record])
    sample.write_bytes(b"synthetic PBC shard; never loaded as a tensor")
    sequence.write_bytes(b"synthetic ESM shard; never loaded as a tensor")
    pbc_index = directory / "pbc_index.json"
    esm_audit = directory / "esm_audit.json"
    write_json(pbc_index, {
        "schema": features.PBC_SCHEMA, "status": "complete", "pbc_policy": {"synthetic": "pbc"},
        "samples": [{**record, "sample_path": sample.name,
                     "sample_sha256": features.sha256(sample), "cache_key": "synthetic_cache_key"}],
    })
    write_json(esm_audit, {
        "schema": "node_rank_pdbbind_esm_audit_v1", "esm_policy": {"synthetic": "esm"},
        "manifests": {"train": {"sha256": features.sha256(manifest)}},
        "sequences": [{"target_id": record["target_id"], "sequence_path": str(sequence),
                       "sequence_sha256": features.sha256(sequence)}],
    })
    output = directory / "features.json"
    args = SimpleNamespace(manifest=manifest, pbc_index=pbc_index, esm_audit=esm_audit,
                           split="train", output=output)
    with patch.object(features, "read_manifest", return_value=[record]):
        with contextlib.redirect_stdout(io.StringIO()):
            features.index_stage(args)
        before = output.read_bytes()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            features.index_stage(args)
        assert json.loads(log.getvalue())["status"] == "verified_existing"
        assert output.read_bytes() == before
        changed = json.loads(before)
        changed["samples"][0]["p_affinity"] = 999.0
        write_json(output, changed)
        try:
            features.index_stage(args)
        except ValueError as error:
            assert "Existing feature index differs" in str(error)
        else:
            raise AssertionError("An altered published feature index was accepted.")
        assert json.loads(output.read_text()) == changed


def verify():
    with tempfile.TemporaryDirectory(prefix="node_rank_launcher_verify_") as temporary:
        directory = Path(temporary)
        checked = verify_phase_arguments(configuration(directory / "cli"))
        verify_worker(directory / "completed")
        verify_worker(directory / "incomplete", trainer_status="paused")
        verify_index_idempotence(directory / "index")
    return {
        "passed": True, "synthetic_only": True, "encoder_loaded": False,
        "training_performed": False, "real_dataset_or_holdout_read": False,
        "phase_argument_parsers_verified": checked,
        "checks": ["real_stage_cli_compatibility", "expected_esm_audit_path",
                   "explicit_feature_index_split_and_sources", "ordered_worker_phase_completion",
                   "trainer_complete_status_accepted", "trainer_paused_status_rejected",
                   "completed_worker_restart_skips_all_stages", "published_index_idempotent",
                   "modified_index_rejected_without_overwrite"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError(f"Choose a new report path: {args.report}")
    report = verify()
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
