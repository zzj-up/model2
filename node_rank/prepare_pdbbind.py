"""Scan local PDBbind structures and lock a compound-disjoint 70/15/15 split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import tempfile

ROOT = Path(__file__).resolve().parent.parent
VERSION = "pdbbind_structural_score_dataset_v1"
SPLIT_VERSION = "pdbbind_compound_group_split_v2"
SPLIT_RULE = "seeded_shuffled_compound_groups_nearest_cumulative_70_85_boundaries"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False, mode="w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def old_preparation():
    import sys
    spec = importlib.util.spec_from_file_location("pdbbind_original_preparation", ROOT / "inputs/pdbbind_1000/prepare_dataset.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def record_to_manifest(record):
    sequence = record["protein_sequence"]
    return {
        "pair_id": "PDBbind2020_" + record["pdb_id"], "sample_id": "PDBbind2020_" + record["pdb_id"],
        "pdb_id": record["pdb_id"], "smiles": record["smiles"], "protein_sequence": sequence,
        "protein_chain": record["protein_chain"], "pose_index": 0,
        "protein_pdb": record["source_protein_pdb"], "ligand_sdf": record["source_ligand_sdf"],
        "compound_group": record["dedup_identity"], "dedup_smiles": record["dedup_smiles"],
        "dedup_inchi_key": record["dedup_inchi_key"],
        "target_id": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "p_affinity": record["p_affinity"], "affinity_endpoint": record["affinity_endpoint"],
        "resolution_angstrom": record["resolution_angstrom"],
        "protein_pdb_sha256": record["source_protein_sha256"],
        "ligand_sdf_sha256": record["source_ligand_sha256"],
    }


def scan(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    helper = old_preparation()
    index, index_rejected = helper.read_index(args.index, args.max_resolution)
    structures = helper.discover_structures(args.source_root)
    identity = {"source_root": str(args.source_root.resolve()), "index_sha256": digest(args.index),
                "scan_script_sha256": digest(__file__), "base_preparation_sha256": digest(Path(helper.__file__)),
                "max_resolution": args.max_resolution, "min_heavy": 8, "max_heavy": 70, "min_chain_length": 30}
    config = output / "scan_config.json"
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError("Existing scan uses different inputs/code. Choose a new output directory.")
    write_json(config, identity)
    cache = output / "inspection_rows"
    cache.mkdir(exist_ok=True)
    candidates, rejected, jobs = [], list(index_rejected), []
    for pdb_id, label in sorted(index.items()):
        protein, ligand = structures.get(pdb_id, (None, None))
        row_file = cache / (pdb_id + ".json")
        if row_file.exists():
            saved = json.loads(row_file.read_text())
            if saved["candidate"] is not None:
                row = saved["candidate"]
                if digest(row["source_protein_pdb"]) != row["source_protein_sha256"] or digest(row["source_ligand_sdf"]) != row["source_ligand_sha256"]:
                    raise ValueError("A cached structural input changed.")
                candidates.append(row)
            else:
                rejected.append(saved["rejection"])
        else:
            jobs.append((pdb_id, label, str(protein) if protein else None, str(ligand) if ligand else None, 8, 70, 30))
    for number, (candidate, rejection) in enumerate(helper._inspection_results(jobs, args.workers), 1):
        pdb_id = (candidate or rejection)["pdb_id"]
        write_json(cache / (pdb_id + ".json"), {"candidate": candidate, "rejection": rejection})
        if candidate is not None:
            candidates.append(candidate)
        else:
            rejected.append(rejection)
        if number % 100 == 0 or number == len(jobs):
            print(json.dumps({"stage": "structure_scan", "newly_inspected": number, "remaining_jobs": len(jobs), "accepted": len(candidates)}), flush=True)
    # Preserve distinct targets for a compound; repeated crystal structures of
    # the same compound and target are represented by the best resolution only.
    candidates.sort(key=lambda r: (r["resolution_angstrom"], r["pdb_id"]))
    seen, unique, duplicates = set(), [], []
    for record in candidates:
        key = (record["dedup_identity"], record["protein_sequence"])
        if key in seen:
            duplicates.append({"pdb_id": record["pdb_id"], "reason": "duplicate_compound_and_target_best_resolution_retained"})
        else:
            seen.add(key)
            unique.append(record)
    rows = [record_to_manifest(row) for row in sorted(unique, key=lambda r: r["pdb_id"])]
    write_json(output / "candidate_manifest.json", rows)
    write_json(output / "structural_rejections.json", rejected)
    write_json(output / "duplicate_pair_structures.json", duplicates)
    summary = {"schema": VERSION, "scope": args.dataset_version, "local_structure_pairs": len(structures),
               "index_rows_accepted": len(index), "structurally_accepted": len(candidates),
               "candidate_pairs": len(rows), "duplicate_pair_structures": len(duplicates),
               "structural_rejections": len(rejected), "rejection_reasons": dict(Counter(r["reason"].split(":")[0] for r in rejected)),
               "source_identity": identity, "pbc_feature_validation_pending": True, "split_locked": False,
               "training_performed": False, "official_pretraining_overlap_audited": False}
    write_json(output / "scan_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def split_records(records, seed=20261118):
    groups = defaultdict(list)
    for record in records:
        groups[record["compound_group"]].append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 10:
        raise ValueError("At least ten compound groups required for this training split.")
    # Keep the seeded random group order. Sorting by size and filling the
    # largest absolute deficit would systematically send repeated compounds
    # into the 70% training partition before either smaller partition fills.
    cumulative = [0]
    for key in keys:
        cumulative.append(cumulative[-1] + len(groups[key]))
    train_end = min(range(1, len(keys) - 1),
                    key=lambda boundary: (abs(cumulative[boundary] - .70 * len(records)), boundary))
    validation_end = min(range(train_end + 1, len(keys)),
                         key=lambda boundary: (abs(cumulative[boundary] - .85 * len(records)), boundary))
    group_partitions = {"train": keys[:train_end], "validation": keys[train_end:validation_end],
                        "holdout": keys[validation_end:]}
    splits = {name: [record for key in selected for record in groups[key]]
              for name, selected in group_partitions.items()}
    for rows in splits.values():
        rows.sort(key=lambda row: row["pair_id"])
    group_sets = [{r["compound_group"] for r in rows} for rows in splits.values()]
    for a in range(3):
        for b in range(a + 1, 3):
            if group_sets[a] & group_sets[b]:
                raise AssertionError("Compound group leakage across partitions.")
    return splits


def finalize(args):
    rows = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    pbc = json.loads(args.pbc_index.read_text(encoding="utf-8-sig"))
    if (pbc.get("schema") != "node_rank_pdbbind_pbc_index_v1" or pbc.get("status") != "complete"
            or pbc.get("manifest_sha256") != digest(args.manifest)
            or pbc.get("source_records_count") != len(rows) or pbc.get("attempted_records_count") != len(rows)):
        raise ValueError("Require complete full-candidate PBC audit with exact manifest identity, not a benchmark subset.")
    audited = [sample["pair_id"] for sample in pbc["samples"]] + [sample["pair_id"] for sample in pbc["failures"]]
    if len(audited) != len(set(audited)) or set(audited) != {row["pair_id"] for row in rows}:
        raise ValueError("PBC success and failure rows must cover each candidate exactly once.")
    # The feature stage publishes success identifiers only after validation.
    successes = {sample["pair_id"] for sample in pbc["samples"]}
    selected = [row for row in rows if row["pair_id"] in successes]
    if len(selected) != len(successes):
        raise ValueError("PBC index contains unknown/duplicate candidate pair identities.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "SPLIT_LOCK.json"
    if lock.exists():
        saved = json.loads(lock.read_text())
        if saved.get("schema") != SPLIT_VERSION or saved.get("rule") != SPLIT_RULE:
            raise ValueError("Existing split uses another schema or rule; choose a new output directory.")
        if saved.get("candidate_manifest_sha256") != digest(args.manifest) or saved.get("pbc_index_sha256") != digest(args.pbc_index) or saved.get("seed") != args.seed:
            raise ValueError("Split already locked for different inputs; refusing to reshuffle.")
        partitions = saved.get("partitions")
        if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "holdout"}:
            raise ValueError("Locked split must declare exactly the three expected partitions.")
        for name, part in partitions.items():
            expected = output / (name + "_manifest.json")
            if (not isinstance(part, dict) or not isinstance(part.get("path"), str)
                    or not Path(part["path"]).is_absolute() or Path(part["path"]).resolve() != expected):
                raise ValueError(f"Locked {name} manifest path differs from its expected output location.")
            if not expected.is_file() or digest(expected) != part.get("sha256"):
                raise ValueError(f"Locked {name} manifest is missing or its checksum changed.")
        print(json.dumps(saved, indent=2))
        return
    partitions = split_records(selected, args.seed)
    paths = {}
    for split, records in partitions.items():
        path = output / (split + "_manifest.json")
        write_json(path, records)
        paths[split] = {"path": str(path), "sha256": digest(path), "pairs": len(records),
                        "fraction_of_usable_pairs": len(records) / len(selected),
                        "compound_groups": len({r["compound_group"] for r in records}),
                        "endpoints": dict(Counter(r["affinity_endpoint"] for r in records))}
    summary = {"schema": SPLIT_VERSION, "seed": args.seed, "usable_pairs": len(selected),
               "requested_fractions": {"train": .7, "validation": .15, "holdout": .15},
               "candidate_manifest_sha256": digest(args.manifest), "pbc_index_sha256": digest(args.pbc_index),
               "partitions": paths, "compound_group_overlap": 0, "test_evaluation_performed": False,
               "rule": SPLIT_RULE,
               "unusable_structures_are_not_training_examples": True}
    write_json(lock, summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan_parser = sub.add_parser("scan")
    scan_parser.add_argument("--source-root", type=Path, default=ROOT / "inputs/pdbbind_1000/source_2020/refined-set")
    scan_parser.add_argument("--index", type=Path, default=ROOT / "inputs/pdbbind_1000/source_2020/index/INDEX_refined_data.2020")
    scan_parser.add_argument("--dataset-version", default="PDBbind2020-refined-local")
    scan_parser.add_argument("--max-resolution", type=float, default=2.5)
    scan_parser.add_argument("--workers", type=int, default=2)
    scan_parser.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--manifest", type=Path, required=True)
    final.add_argument("--pbc-index", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.command == "scan":
        if args.workers < 1 or not math.isfinite(args.max_resolution) or args.max_resolution <= 0:
            parser.error("Positive workers and resolution required.")
        scan(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
