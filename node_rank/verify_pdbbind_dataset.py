"""Synthetic checks of compound-disjoint splitting and full-audit requirements.

Uses temporary synthetic manifests only; does not read or evaluate held-out data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

from .prepare_pdbbind import SPLIT_RULE, SPLIT_VERSION, digest, finalize, split_records, write_json


def synthetic_records():
    rows = []
    for group in range(100):
        for target in range(1 + group % 5):
            rows.append({
                "pair_id": f"synthetic_{group:03d}_{target}",
                "compound_group": f"parent_{group:03d}",
                "target_id": f"target_{target}",
                "smiles": f"synthetic_SMILES_{group}",
                "affinity_endpoint": "Ki" if group % 2 else "Kd",
                "p_affinity": float(group + target) / 10,
                "retained_metadata": {"example": "synthetic_only", "target": target},
            })
    return rows


def assignments(splits):
    return {row["pair_id"]: split for split, rows in splits.items() for row in rows}


def sparse_repeated_records():
    """Most compounds are singletons, with a few three-target compounds."""
    return [{"pair_id": f"sparse_{group:03d}_{target}", "compound_group": f"parent_{group:03d}",
             "target_id": f"target_{target}", "affinity_endpoint": "Kd", "p_affinity": 6. + target}
            for group in range(110) for target in range(3 if group < 10 else 1)]


def verify():
    records = synthetic_records()
    first = split_records(records, 20261118)
    assert first == split_records(records, 20261118)
    shuffled = copy.deepcopy(records)
    random.Random(2).shuffle(shuffled)
    assert first == split_records(shuffled, 20261118)
    relabelled = copy.deepcopy(records)
    for index, row in enumerate(relabelled):
        row["p_affinity"] = 9999. - index
        row["affinity_endpoint"] = "irrelevant_for_splitting"
    assert assignments(first) == assignments(split_records(relabelled, 20261118))
    assert assignments(first) != assignments(split_records(records, 20261119))
    expected = {row["pair_id"]: row for row in records}
    all_rows = [row for rows in first.values() for row in rows]
    assert len(all_rows) == len(records)
    assert {row["pair_id"]: row for row in all_rows} == expected
    sets = {name: {row["compound_group"] for row in rows} for name, rows in first.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert sets[a].isdisjoint(sets[b])
    max_group = 5
    for split, fraction in (("train", .7), ("validation", .15), ("holdout", .15)):
        assert first[split]
        assert abs(len(first[split]) - fraction * len(records)) <= max_group

    sparse = sparse_repeated_records()
    sparse_split = split_records(sparse, 20261118)
    sparse_counts = {name: Counter(row["compound_group"] for row in rows)
                     for name, rows in sparse_split.items()}
    sparse_repeated = {name: sum(count > 1 for count in counts.values())
                       for name, counts in sparse_counts.items()}
    assert all(sparse_repeated.values()), "Sparse repeated groups were systematically excluded from a partition."
    sparse_shuffled = copy.deepcopy(sparse)
    random.Random(2).shuffle(sparse_shuffled)
    assert sparse_split == split_records(sparse_shuffled, 20261118)
    for row in sparse_shuffled:
        row["p_affinity"] = -9999.
        row["affinity_endpoint"] = "irrelevant_for_splitting"
    assert assignments(sparse_split) == assignments(split_records(sparse_shuffled, 20261118))
    assert len(assignments(sparse_split)) == len(sparse)
    sparse_names = list(sparse_counts)
    for index, name in enumerate(sparse_names):
        for other in sparse_names[index + 1:]:
            assert set(sparse_counts[name]).isdisjoint(sparse_counts[other])
    for split, fraction in (("train", .7), ("validation", .15), ("holdout", .15)):
        assert abs(len(sparse_split[split]) - fraction * len(sparse)) <= 3

    rejected = []
    with tempfile.TemporaryDirectory(prefix="node_rank_dataset_verify_") as temporary:
        directory = Path(temporary)
        manifest = directory / "candidates.json"
        audit_path = directory / "pbc_index.json"
        write_json(manifest, records)
        audit = {
            "schema": "node_rank_pdbbind_pbc_index_v1", "status": "complete",
            "manifest_sha256": digest(manifest),
            "source_records_count": len(records), "attempted_records_count": len(records),
            "samples": [{"pair_id": row["pair_id"]} for row in records[:-1]],
            "failures": [{"pair_id": records[-1]["pair_id"], "reason": "synthetic_failure"}],
        }
        invalid = []
        value = copy.deepcopy(audit)
        value["status"] = "partial"
        invalid.append(("partial_status", value))
        value = copy.deepcopy(audit)
        value["attempted_records_count"] -= 1
        invalid.append(("incomplete_attempt_count", value))
        value = copy.deepcopy(audit)
        value["samples"] = value["samples"][:-1]
        invalid.append(("missing_candidate_result", value))
        value = copy.deepcopy(audit)
        value["failures"][0]["pair_id"] = value["samples"][0]["pair_id"]
        invalid.append(("duplicate_success_failure_id", value))
        value = copy.deepcopy(audit)
        value["samples"][0]["pair_id"] = "unknown_pair"
        invalid.append(("unknown_candidate_id", value))
        value = copy.deepcopy(audit)
        value["manifest_sha256"] = "0" * 64
        invalid.append(("manifest_identity_mismatch", value))
        for name, value in invalid:
            write_json(audit_path, value)
            output = directory / name
            args = SimpleNamespace(manifest=manifest, pbc_index=audit_path, output=output, seed=20261118)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    finalize(args)
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError(f"Incomplete or invalid PBC audit accepted: {name}")
            assert not (output / "SPLIT_LOCK.json").exists()

        write_json(audit_path, audit)
        output = directory / "valid"
        args = SimpleNamespace(manifest=manifest, pbc_index=audit_path, output=output, seed=20261118)
        with contextlib.redirect_stdout(io.StringIO()):
            finalize(args)
        lock_path = output / "SPLIT_LOCK.json"
        before = lock_path.read_bytes()
        lock = json.loads(before)
        assert lock["schema"] == SPLIT_VERSION and lock["rule"] == SPLIT_RULE
        published = []
        for part in lock["partitions"].values():
            path = Path(part["path"])
            assert digest(path) == part["sha256"]
            published.extend(json.loads(path.read_text()))
        assert {row["pair_id"]: row for row in published} == {
            row["pair_id"]: row for row in records[:-1]
        }
        assert lock["test_evaluation_performed"] is False
        with contextlib.redirect_stdout(io.StringIO()):
            finalize(args)
        assert lock_path.read_bytes() == before
        different_seed = SimpleNamespace(**{**vars(args), "seed": 20261119})
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                finalize(different_seed)
        except ValueError:
            rejected.append("locked_split_reshuffle")
        else:
            raise AssertionError("An existing split was reshuffled with a different seed.")

        def reject_locked_change(name):
            changed_lock = lock_path.read_bytes()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    finalize(args)
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError(f"A changed locked split was accepted: {name}")
            assert lock_path.read_bytes() == changed_lock

        changed = copy.deepcopy(lock)
        changed["schema"] = "pdbbind_structural_score_dataset_v1"
        write_json(lock_path, changed)
        reject_locked_change("locked_split_old_schema")
        changed = copy.deepcopy(lock)
        changed["rule"] = "another_rule"
        write_json(lock_path, changed)
        reject_locked_change("locked_split_changed_rule")
        changed = copy.deepcopy(lock)
        changed["partitions"]["train"]["path"] = lock["partitions"]["validation"]["path"]
        write_json(lock_path, changed)
        reject_locked_change("locked_partition_wrong_path")
        changed = copy.deepcopy(lock)
        del changed["partitions"]["holdout"]
        write_json(lock_path, changed)
        reject_locked_change("locked_partition_missing_declaration")
        lock_path.write_bytes(before)
        train_path = Path(lock["partitions"]["train"]["path"])
        train_before = train_path.read_bytes()
        changed_rows = json.loads(train_before)
        changed_rows[0]["p_affinity"] += 1.
        write_json(train_path, changed_rows)
        reject_locked_change("locked_partition_checksum_changed")
        assert json.loads(train_path.read_text()) == changed_rows
        train_path.write_bytes(train_before)
        train_path.unlink()
        reject_locked_change("locked_partition_missing_file")
        train_path.write_bytes(train_before)
        with contextlib.redirect_stdout(io.StringIO()):
            finalize(args)
        assert lock_path.read_bytes() == before

    return {
        "passed": True, "synthetic_only": True,
        "training_performed": False, "real_holdout_read_or_evaluated": False,
        "checks": ["seed_repeatability", "input_order_independence", "label_independent_partition_assignment",
                   "different_seed_changes_assignment", "compound_group_disjointness", "complete_record_preservation",
                   "approximate_70_15_15_complex_counts", "full_audit_required", "failed_structures_excluded",
                   "published_partition_hashes", "existing_lock_idempotence", "existing_lock_reshuffle_rejected",
                   "sparse_repeated_groups_reach_all_partitions", "sparse_input_order_and_label_independence",
                   "locked_schema_and_rule_verified", "locked_partition_paths_verified",
                   "locked_partition_tampering_and_missing_files_rejected"],
        "split_schema": SPLIT_VERSION,
        "synthetic_partition_counts": {name: len(rows) for name, rows in first.items()},
        "sparse_synthetic_partition_counts": {name: len(rows) for name, rows in sparse_split.items()},
        "sparse_synthetic_repeated_groups": sparse_repeated,
        "invalid_inputs_rejected": rejected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError(f"Choose a new verification report path: {args.report}")
    report = verify()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
