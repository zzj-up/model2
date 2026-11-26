"""Check atom/residue mapping and window coverage without loading encoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from rdkit import Chem

from node_rank.prepare_features import ROOT, SCHEMA, directed_bonds, load_records, pocket_mapping, reject_quarantine_path, residue_windows, sha256
from prepare_pbcnet_features import crop_pocket, read_pose


def verify(manifest):
    checks, mapping_rows = [], []
    for length in (1, 509, 510, 511, 1020, 1021):
        sequence = ("ACDEFGHIKLMNPQRSTVWY" * ((length + 19) // 20))[:length]
        windows = residue_windows(sequence)
        assert "".join(window for _, window in windows) == sequence
        assert [start for start, _ in windows] == list(range(0, length, 510))
        assert all(1 <= len(window) <= 510 for _, window in windows)
    checks.append("Full residue coverage across six window boundary cases")
    records = load_records(manifest)
    for row_index, record in enumerate(records):
        ligand, coordinates = read_pose(record["ligand_sdf"], record.get("pose_index", 0), record["smiles"])
        with tempfile.TemporaryDirectory(prefix="node_rank_mapping_test_") as directory:
            pocket_path = Path(directory) / "pocket.pdb"
            preparation = crop_pocket(record["protein_pdb"], coordinates, record["protein_sequence"], pocket_path, record, 5000)
            pocket = Chem.RemoveAllHs(Chem.MolFromPDBFile(str(pocket_path)))
            allowed = {row["chain"] for row in preparation["chain_alignment"]}
            mapping, atoms = pocket_mapping(record["protein_pdb"], pocket, record["protein_sequence"], allowed, preparation["kept_hetero_residues"])
            assert mapping.shape == (pocket.GetNumAtoms(),)
            assert mapping.dtype == torch.long and (mapping >= 0).all()
            assert (mapping < len(record["protein_sequence"])).all()
            assert [atom["sequence_index"] for atom in atoms] == mapping.tolist()
            edges, types = directed_bonds(ligand)
            assert edges.shape[1] == 2 * ligand.GetNumBonds() == len(types)
            if edges.numel():
                assert torch.equal(edges[:, 0::2].flip(0), edges[:, 1::2])
            # The atom identity is not accepted simply because the residue ID
            # matches: geometry must still correspond to the source structure.
            changed = Chem.Mol(pocket)
            position = np.asarray(changed.GetConformer().GetAtomPosition(0))
            changed.GetConformer().SetAtomPosition(0, position + np.array([1.0, 0, 0]))
            try:
                pocket_mapping(record["protein_pdb"], changed, record["protein_sequence"], allowed, preparation["kept_hetero_residues"])
            except ValueError as error:
                assert "coordinate" in str(error)
            else:
                raise AssertionError("Altered pocket atom coordinates were accepted.")
            if row_index == 0:
                try:
                    pocket_mapping(record["protein_pdb"], pocket, record["protein_sequence"] * 2, allowed, preparation["kept_hetero_residues"])
                except ValueError as error:
                    assert "Ambiguous optimal" in str(error)
                else:
                    raise AssertionError("A repeated-sequence ambiguous alignment was accepted.")
            mapping_rows.append({"pair_id": record["pair_id"], "pocket_atoms": len(mapping),
                                 "mapped_residues": len(set(mapping.tolist())), "sequence_length": len(record["protein_sequence"])})
    checks.extend(["Five real PDBbind structures map exact RDKit atoms to full-sequence residues",
                   "Altered source coordinates rejected for every structure", "Directed chemical bond identity verified"])
    for path in (ROOT / "quarantine/no_read.json", ROOT / "quarantine/nested/no_read.pdb"):
        try:
            reject_quarantine_path(path)
        except ValueError as error:
            assert "Isolated historical" in str(error)
        else:
            raise AssertionError("A quarantined source path was accepted.")
    try:
        load_records(ROOT / "quarantine/no_read.json")
    except ValueError as error:
        assert "Isolated historical" in str(error)
    else:
        raise AssertionError("A quarantined manifest was read.")
    with tempfile.TemporaryDirectory(prefix="node_rank_path_test_") as temporary:
        path = Path(temporary) / "manifest.json"
        record = dict(records[0], protein_pdb=str(ROOT / "quarantine/no_read.pdb"))
        path.write_text(json.dumps([record]), encoding="utf-8")
        try:
            load_records(path)
        except ValueError as error:
            assert "Isolated historical" in str(error)
        else:
            raise AssertionError("A quarantined structure was read through an external manifest.")
    checks.extend(["Repeated-sequence optimal-alignment ambiguity rejected", "Quarantined manifest and structural source paths rejected before reading"])
    return {"passed": True, "checks": checks, "rows": mapping_rows, "encoders_loaded": False,
            "note": "Mapping/unit checks only; does not establish model predictive performance."}


def verify_extracted(path, manifest, pooled_entities=None, pooled_complex=None):
    """Verify genuine node variation and optional parity with old frozen readouts.

    Old caches are read only for numerical verification; they are never inputs
    to the node feature extractor or the new ranking model.
    """
    from pbcnet_features import structure_key
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    assert artifact["schema"] == SCHEMA
    source_records = load_records(manifest)
    by_id = {record["pair_id"]: record for record in source_records}
    pooled_esm = torch.load(pooled_entities, map_location="cpu", weights_only=True) if pooled_entities else None
    pooled_pbc = torch.load(pooled_complex, map_location="cpu", weights_only=True) if pooled_complex else None
    rows = []
    for sample in artifact["samples"]:
        record = by_id[sample["pair_id"]]
        assert sample["protein_sequence"] == record["protein_sequence"]
        assert sample["sequence_features"].shape == (len(record["protein_sequence"]), 2560)
        for field in ("sequence_features", "ligand_features", "pocket_features"):
            value = sample[field]
            assert value.dtype == torch.float32 and torch.isfinite(value).all()
            if len(value) > 1:
                assert not torch.equal(value, value[:1].expand_as(value)), "Node features were collapsed to identical rows."
        row = {"pair_id": sample["pair_id"], "genuine_node_variation": True}
        if pooled_esm is not None:
            index = pooled_esm["protein_sequences"].index(record["protein_sequence"])
            expected = pooled_esm["protein_features"][index]
            actual = sample["sequence_features"].mean(0)
            row["esm_readout_max_abs_error"] = float((actual - expected).abs().max())
            assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
        if pooled_pbc is not None:
            index = pooled_pbc["keys"].index(structure_key(record["smiles"], record["protein_sequence"]))
            expected = pooled_pbc["embeddings"][index]
            actual = sample["ligand_features"].sum(0)
            row["pbc_readout_max_abs_error"] = float((actual - expected).abs().max())
            assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
        rows.append(row)
    return {"feature_sha256": sha256(path), "passed": True, "rows": rows,
            "note": "Frozen-encoder extraction parity; not predictive evaluation and no task optimization."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "evaluation/pdbbind_unseen5_20261116/dataset/structure_manifest.json")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--pooled-entities", type=Path)
    parser.add_argument("--pooled-complex", type=Path)
    args = parser.parse_args()
    report = verify(args.manifest)
    if args.features:
        report["extracted_feature_verification"] = verify_extracted(args.features, args.manifest, args.pooled_entities, args.pooled_complex)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
