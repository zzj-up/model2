"""Build a label-free sequence panel from the locked PDBbind split.

The panel contains every exact training protein sequence plus every exact
sequence recorded for the selected holdout compound group. Each target offers
up to three native receptors for preparation fallback, preferring a native
ligand unrelated to both queries, then higher resolution and lexical PDB ID.

Boxes are centered on the native ligand's heavy-atom centroid. Each side is
twice the maximum native coordinate deviation on that axis plus 12 angstroms
(6 angstroms on each side), clamped to [16, 32] angstroms. These native-site
boxes define the candidate pockets; no native ligand coordinates are supplied
as a query placement. All emitted structural paths are absolute WSL paths.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIRECTORY = PROJECT_ROOT / "inputs/pdbbind2020_retrain_20261118/dataset/splits_v2"
NAME_INDEX = PROJECT_ROOT / "inputs/pdbbind_1000/source_2020/index/INDEX_refined_name.2020"
SCHEMA = "model2_locked_sequence_panel_v1"
BOX_POLICY = "native_heavy_centroid_twice_max_axis_deviation_plus_12A_clamped_16_to_32A"


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _load_partitions():
    lock_path = SPLIT_DIRECTORY / "SPLIT_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    if lock.get("schema") != "pdbbind_compound_group_split_v2":
        raise ValueError("The panel requires the locked v2 compound-disjoint split.")
    result = {}
    for name in ("train", "holdout"):
        path = SPLIT_DIRECTORY / (name + "_manifest.json")
        entry = lock.get("partitions", {}).get(name, {})
        if entry.get("sha256") != _sha256(path):
            raise ValueError(f"The locked {name} manifest checksum differs.")
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(values, list) or len(values) != entry.get("pairs"):
            raise ValueError(f"The locked {name} manifest count differs.")
        for row in values:
            target_id = hashlib.sha256(row["protein_sequence"].encode("ascii")).hexdigest()
            if row.get("target_id") != target_id:
                raise ValueError(f"{row.get('pdb_id')}: target identity differs from its exact sequence.")
        result[name] = values
    train_groups = {row["compound_group"] for row in result["train"]}
    if train_groups & {row["compound_group"] for row in result["holdout"]}:
        raise ValueError("The training and holdout compound groups overlap.")
    return result, lock_path


def _annotations():
    result = {}
    for line in NAME_INDEX.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split(None, 3)
        if len(fields) != 4:
            raise ValueError("Malformed local PDBbind target-name annotation.")
        pdb_id, _, accession, name = fields
        result[pdb_id.lower()] = {
            "target_name": name.strip(),
            "accession": accession if re.fullmatch(r"[A-Za-z0-9]{6}|[A-Za-z0-9]{10}", accession) else None,
        }
    return result


def _query(rows, split, pdb_id):
    if not isinstance(pdb_id, str) or not re.fullmatch(r"[A-Za-z0-9]{4}", pdb_id):
        raise ValueError("Query PDB IDs must contain four letters or digits.")
    pdb_id = pdb_id.lower()
    selected = [row for row in rows if row["pdb_id"] == pdb_id]
    if len(selected) != 1:
        raise ValueError(f"PDB {pdb_id} must occur exactly once in the locked {split} partition.")
    selected = selected[0]
    known_targets = sorted({row["target_id"] for row in rows
                            if row["compound_group"] == selected["compound_group"]})
    return {"split": split, "pdb_id": pdb_id, "smiles": selected["smiles"],
            "compound_group": selected["compound_group"], "known_target_ids": known_targets}


def _absolute_source(row, field):
    source = Path(row[field])
    if not source.is_absolute() or not source.as_posix().startswith("/mnt/"):
        raise ValueError(f"{row['pdb_id']}: expected an absolute WSL structural path for {field}.")
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    expected = row.get(field + "_sha256")
    if not expected or _sha256(source) != expected:
        raise ValueError(f"{row['pdb_id']}: original {field} checksum differs.")
    return source


def _structure(row, query_groups):
    from rdkit import Chem, rdBase

    protein = _absolute_source(row, "protein_pdb")
    ligand = _absolute_source(row, "ligand_sdf")
    pose_index = row.get("pose_index", 0)
    if type(pose_index) is not int or pose_index < 0:
        raise ValueError(f"{row['pdb_id']}: invalid native pose index.")
    # Several verified PDBbind files retain a 2D header despite nonzero Z
    # coordinates. RDKit promotes those to 3D; suppress the repeated parser
    # warning while preserving explicit checks and errors below.
    with rdBase.BlockLogs():
        supplier = Chem.SDMolSupplier(str(ligand), removeHs=False, sanitize=True)
        if pose_index >= len(supplier):
            raise ValueError(f"{row['pdb_id']}: invalid native pose index.")
        molecule = supplier[pose_index]
    if molecule is None or molecule.GetNumConformers() != 1:
        raise ValueError(f"{row['pdb_id']}: native pose could not be read.")
    conformer = molecule.GetConformer()
    if not conformer.Is3D():
        raise ValueError(f"{row['pdb_id']}: native ligand lacks three-dimensional coordinates.")
    coordinates = [[float(value) for value in conformer.GetAtomPosition(atom.GetIdx())]
                   for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    if not coordinates or not all(math.isfinite(value) for xyz in coordinates for value in xyz):
        raise ValueError(f"{row['pdb_id']}: native heavy-atom coordinates are invalid.")
    center = [sum(xyz[axis] for xyz in coordinates) / len(coordinates) for axis in range(3)]
    requested = [2 * max(abs(xyz[axis] - center[axis]) for xyz in coordinates) + 12
                 for axis in range(3)]
    size = [max(16., min(32., side)) for side in requested]
    return {
        "pdb_id": row["pdb_id"], "protein_pdb": str(protein),
        "protein_chain": row["protein_chain"],
        "box_center": [round(value, 4) for value in center],
        "box_size": [round(value, 4) for value in size],
        "known_query_reference": row["compound_group"] in query_groups,
        "native_ligand_sdf": str(ligand), "native_ligand_sdf_sha256": row["ligand_sdf_sha256"],
        "protein_pdb_sha256": row["protein_pdb_sha256"],
        "native_box_upper_clamped": any(value > 32 for value in requested),
    }


def build_panel(train_pdb, holdout_pdb):
    """Return the agreed ``queries`` and per-sequence ``targets`` panel."""
    partitions, lock_path = _load_partitions()
    queries = [_query(partitions["train"], "train", train_pdb),
               _query(partitions["holdout"], "holdout", holdout_pdb)]
    query_groups = {query["compound_group"] for query in queries}
    heldout_group = queries[1]["compound_group"]
    eligible = partitions["train"] + [row for row in partitions["holdout"]
                                         if row["compound_group"] == heldout_group]
    by_sequence = defaultdict(list)
    for row in eligible:
        by_sequence[row["target_id"]].append(row)
    annotations = _annotations()
    known = [set(query["known_target_ids"]) for query in queries]
    targets = []
    for target_id, rows in sorted(by_sequence.items()):
        ordered = sorted(rows, key=lambda row: (row["compound_group"] in query_groups,
                                               row["resolution_angstrom"], row["pdb_id"]))
        representatives = ordered[:3]
        primary = representatives[0]
        annotation = annotations.get(primary["pdb_id"], {"target_name": primary["pdb_id"], "accession": None})
        targets.append({
            "id": "seq_" + target_id, "panel_id": "seq_" + target_id,
            "target_id": target_id, "target_name": annotation["target_name"],
            "accession": annotation["accession"], "protein_sequence": primary["protein_sequence"],
            "known_for_train_query": target_id in known[0],
            "known_for_holdout_query": target_id in known[1],
            "structures": [_structure(row, query_groups) for row in representatives],
        })
    if not all(set(query["known_target_ids"]) <= set(by_sequence) for query in queries):
        raise AssertionError("A query's known target sequence is missing from the candidate panel.")
    return {
        "schema": SCHEMA, "queries": queries, "targets": targets,
        "panel_rule": "all_training_exact_sequences_plus_all_sequences_of_selected_holdout_compound_group",
        "target_identity": "SHA256_of_exact_input_protein_sequence_not_UniProt_gene_identity",
        "receptor_selection": "prefer_native_ligand_unrelated_to_both_queries_then_resolution_then_pdb_id_up_to_three",
        "box_policy": BOX_POLICY, "split_lock_sha256": _sha256(lock_path),
        "target_name_index_sha256": _sha256(NAME_INDEX),
        "experimental_affinity_labels_included": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pdb", required=True)
    parser.add_argument("--holdout-pdb", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new panel path: {args.output}")
    panel = build_panel(args.train_pdb, args.holdout_pdb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(panel, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"panel": str(args.output.resolve()), "targets": len(panel["targets"]),
                      "receptor_candidates": sum(len(target["structures"]) for target in panel["targets"]),
                      "known_targets_per_query": {query["pdb_id"]: len(query["known_target_ids"])
                                                  for query in panel["queries"]}}, indent=2))


if __name__ == "__main__":
    main()
