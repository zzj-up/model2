"""Frozen per-residue ESM2 and per-atom PBCNet features for the node ranker.

The all stage launches the two incompatible model environments sequentially.
No pooled legacy feature is expanded or reused as a node representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "node_rank_features_v1"
VERSION = "esm_residue_pbc_atom_aligned_v1"
WINDOW = 510


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def reject_quarantine_path(path):
    path = Path(path).resolve()
    if path.is_relative_to((ROOT / "quarantine").resolve()):
        raise ValueError("Isolated historical data/weights cannot be used by node_rank features.")
    return path


def publish(path, value):
    import torch
    path = reject_quarantine_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_records(manifest, limit=None):
    from pbcnet_features import canonical_smiles, normalize_sequence
    manifest = reject_quarantine_path(manifest)
    records = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must be a nonempty JSON list of posed compound-target pairs.")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        records = records[:limit]
    result, pair_ids, identities = [], set(), set()
    for index, incoming in enumerate(records):
        if not isinstance(incoming, dict) or not {"smiles", "protein_sequence", "protein_pdb", "ligand_sdf"} <= incoming.keys():
            raise ValueError(f"Manifest row {index} lacks structure/sequence/compound information.")
        record = dict(incoming)
        record["smiles"] = canonical_smiles(record["smiles"])
        record["protein_sequence"] = normalize_sequence(record["protein_sequence"])
        sequence_id = hashlib.sha256(record["protein_sequence"].encode("ascii")).hexdigest()
        # Identity rather than a display name defines ranking groups. This also
        # prevents two arbitrary aliases of one compound becoming different groups.
        record["compound_id"] = record["smiles"]
        record["target_id"] = sequence_id
        record["pair_id"] = str(record.get("pair_id") or record.get("sample_id") or digest_json([record["smiles"], sequence_id]))
        identity = (record["compound_id"], sequence_id)
        if record["pair_id"] in pair_ids or identity in identities:
            raise ValueError("Require unique pair_id and one explicitly selected pose per compound-target identity.")
        pair_ids.add(record["pair_id"])
        identities.add(identity)
        for field in ("protein_pdb", "ligand_sdf"):
            path = reject_quarantine_path(manifest.parent / record[field])
            if not path.is_file():
                raise FileNotFoundError(path)
            record[field] = str(path)
            record[field + "_sha256"] = sha256(path)
        result.append(record)
    return result


def request_identity(args, records):
    return {
        "version": VERSION,
        "extractor_sha256": sha256(__file__),
        "base_extractor_sha256": sha256(ROOT / "prepare_pbcnet_features.py"),
        "esm_identity_helper_sha256": sha256(ROOT / "fresh_dti/prepare_entity_features.py"),
        "manifest_sha256": sha256(args.manifest),
        "records_digest": digest_json(records),
        "selected_records": len(records),
        "max_atoms": args.max_atoms,
        "window_residues": WINDOW,
        "esm_dtype": "float16" if args.device == "cuda" else "float32",
        "esm_source": str(args.esm_model.resolve()),
        "pbc_source": str(args.repository.resolve()),
    }


def stage_existing(path, schema, identity):
    import torch
    if not path.exists():
        return None
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema") != schema or value.get("request") != identity:
        raise ValueError(f"Existing cache is incompatible or belongs to changed inputs/code: {path}; choose a fresh output path.")
    return value


def directed_bonds(molecule):
    import torch
    pairs, types = [], []
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pairs.extend([(a, b), (b, a)])
        types.extend([float(bond.GetBondTypeAsDouble())] * 2)
    edges = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()
    return edges, torch.tensor(types, dtype=torch.float32)


def pocket_mapping(protein_path, pocket_molecule, sequence, allowed_chains, kept_hetero):
    """Map exact RDKit atom order through PDB chain/number/insertion identifiers.

    Sequence alignment is performed against each original full structural chain,
    never against the cropped pocket sequence. Amino acids in the pocket must
    map to an identical residue; explicitly retained hetero atoms receive -1.
    """
    import numpy as np
    import torch
    from Bio.Align import PairwiseAligner
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import seq1

    original = next(PDBParser(QUIET=True).get_structure("source", str(protein_path)).get_models())
    residue_lookup = {}
    atom_positions = {}
    for chain in original:
        for residue in chain:
            key = (chain.id, int(residue.id[1]), residue.id[2].strip(), residue.resname.strip())
            for atom in residue.get_atoms():
                atom = atom.selected_child if hasattr(atom, "selected_child") else atom
                atom_positions[(key, atom.name.strip())] = np.asarray(atom.coord, dtype=np.float64)
        if chain.id not in allowed_chains:
            continue
        residues = [residue for residue in chain if is_aa(residue, standard=False)]
        chain_sequence = "".join(seq1(residue.resname, custom_map={"MSE": "M"}) for residue in residues)
        aligner = PairwiseAligner(mode="local", match_score=2, mismatch_score=-3, open_gap_score=-8, extend_gap_score=-0.5)
        alignments = aligner.align(chain_sequence, sequence)
        alignment = alignments[0]
        # Ambiguous optimal alignments must not silently assign atom features to
        # an arbitrary sequence repeat. Checking the next solution is bounded.
        try:
            alternative = alignments[1]
        except IndexError:
            alternative = None
        # Reject any second optimum conservatively. Two initial alignments can
        # share an aligned-block map while a later optimum changes that map;
        # comparing only the first two maps does not establish uniqueness.
        if alternative is not None:
            raise ValueError(f"Ambiguous optimal structure-to-sequence alignment in chain {chain.id}.")
        for (a, b), (c, d) in zip(*alignment.aligned):
            for chain_index, target_index in zip(range(a, b), range(c, d)):
                residue = residues[chain_index]
                if chain_sequence[chain_index] != sequence[target_index]:
                    continue
                key = (chain.id, int(residue.id[1]), residue.id[2].strip(), residue.resname.strip())
                residue_lookup[key] = int(target_index)
    mapping, atom_records = [], []
    positions = pocket_molecule.GetConformer().GetPositions()
    for atom in pocket_molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            raise ValueError("Pocket atom lacks PDB residue identity; cannot construct sequence mapping.")
        key = (info.GetChainId(), info.GetResidueNumber(), info.GetInsertionCode().strip(), info.GetResidueName().strip())
        source_key = (key, info.GetName().strip())
        if source_key not in atom_positions or not np.allclose(positions[atom.GetIdx()], atom_positions[source_key], atol=0.0011, rtol=0):
            raise ValueError(f"Pocket atom identity/coordinate differs from original receptor: {source_key}.")
        mapped = residue_lookup.get(key)
        if mapped is None:
            if key[3] in kept_hetero:
                mapped = -1
            else:
                raise ValueError(f"Pocket amino acid does not map uniquely to identical full-sequence residue: {key}.")
        mapping.append(mapped)
        atom_records.append({"atom_index": atom.GetIdx(), "chain": key[0], "residue_number": key[1],
                             "insertion_code": key[2], "residue_name": key[3], "atom_name": info.GetName().strip(),
                             "sequence_index": mapped, "element": atom.GetSymbol()})
    if not mapping or all(index < 0 for index in mapping):
        raise ValueError("No pocket amino acid is linked to the target sequence.")
    return torch.tensor(mapping, dtype=torch.long), atom_records


def extract_pbc(args, records, identity, work):
    import numpy as np
    import torch
    from rdkit import Chem
    import dgl.function as fn
    from prepare_pbcnet_features import build_complex, load_official_encoder

    output = work / "pbc.pt"
    if stage_existing(output, "node_rank_pbc_atoms_v1", identity) is not None:
        print(json.dumps({"stage": "pbc", "cache": "validated", "output": str(output)}), flush=True)
        return
    torch.set_num_threads(args.threads)
    encoder, graph_module, provenance = load_official_encoder(args.repository)
    rows = []
    row_dir = work / "pbc_rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, record in enumerate(records):
            row_path = row_dir / (digest_json(record) + ".pt")
            cached = stage_existing(row_path, "node_rank_pbc_row_v1", identity)
            if cached is not None:
                rows.append(cached["sample"])
                print(json.dumps({"stage": "pbc", "row": index + 1, "total": len(records), "cache": "validated"}), flush=True)
                continue
            with tempfile.TemporaryDirectory(prefix="node_rank_pbc_") as temporary:
                temporary = Path(temporary)
                graph, source = build_complex(record, args.manifest.resolve().parent, temporary, graph_module, args.max_atoms)
                ligand = Chem.RemoveAllHs(Chem.MolFromMolFile(str(temporary / "pose.sdf")))
                pocket = Chem.RemoveAllHs(Chem.MolFromPDBFile(str(temporary / "pocket.pdb")))
                n_ligand, n_pocket = ligand.GetNumAtoms(), pocket.GetNumAtoms()
                positions = torch.from_numpy(np.concatenate([ligand.GetConformer().GetPositions(), pocket.GetConformer().GetPositions()])).float()
                types = graph.nodes["atom"].data["type"]
                atomic_numbers = torch.tensor([atom.GetAtomicNum() for mol in (ligand, pocket) for atom in mol.GetAtoms()])
                if (len(types) != n_ligand + n_pocket or not (types[:n_ligand] == 1).all()
                        or not (types[n_ligand:] == 0).all()
                        or not torch.equal(graph.nodes["atom"].data["x"], atomic_numbers)
                        or not torch.allclose(graph.nodes["atom"].data["pos"], positions, atol=1e-5, rtol=0)):
                    raise ValueError("Official graph atom order differs from exact RDKit ligand/pocket reread.")
                mapping, atom_map = pocket_mapping(record["protein_pdb"], pocket, record["protein_sequence"],
                                                   {value["chain"] for value in source["chain_alignment"]},
                                                   source["kept_hetero_residues"])
                src, dst = graph.edges(etype="int")
                graph.edges["int"].data["dist"] = (positions[src] - positions[dst]).square().sum(-1).sqrt()
                graph.apply_edges(fn.v_sub_u("pos", "pos", "vec"), etype="int")
                node_features, _ = encoder.encoder(graph, False)
                node_features = node_features.detach().cpu().float().clone().contiguous()
                if node_features.shape != (n_ligand + n_pocket, 96) or not torch.isfinite(node_features).all():
                    raise ValueError("Official PBC encoder did not produce finite per-atom 96-dimensional features.")
                ligand_bonds, ligand_bond_orders = directed_bonds(ligand)
                pocket_bonds, pocket_bond_orders = directed_bonds(pocket)
                sample = {"pair_id": record["pair_id"], "compound_id": record["compound_id"], "target_id": record["target_id"],
                          "smiles": record["smiles"], "protein_sequence": record["protein_sequence"],
                          "ligand_features": node_features[:n_ligand].clone(), "pocket_features": node_features[n_ligand:].clone(),
                          "ligand_pos": positions[:n_ligand].clone(), "pocket_pos": positions[n_ligand:].clone(),
                          "ligand_bonds": ligand_bonds, "pocket_bonds": pocket_bonds,
                          "ligand_bond_orders": ligand_bond_orders, "pocket_bond_orders": pocket_bond_orders,
                          "pocket_to_sequence": mapping, "source": {**source, "manifest_record": record, "pocket_atom_map": atom_map}}
                for field in ("ranking_label", "ranking_group", "ranking_endpoint"):
                    if field in record:
                        sample[field] = record[field]
                publish(row_path, {"schema": "node_rank_pbc_row_v1", "request": identity, "sample": sample})
                rows.append(sample)
                print(json.dumps({"stage": "pbc", "row": index + 1, "total": len(records), "ligand_atoms": n_ligand,
                                  "pocket_atoms": n_pocket, "mapped_pocket_atoms": int((mapping >= 0).sum())}), flush=True)
    provenance.update(encoder_mode="frozen_all_atom_tensornet_output_before_readout", original_pairwise_prediction_head_used=False,
                      node_extractor_sha256=sha256(__file__), atom_mapping_policy="exact_rdkit_order_pdb_identity_full_chain_unique_alignment_v1")
    publish(output, {"schema": "node_rank_pbc_atoms_v1", "request": identity, "samples": rows, "provenance": provenance})


def residue_windows(sequence):
    return [(start, sequence[start:start + WINDOW]) for start in range(0, len(sequence), WINDOW)]


def extract_esm(args, records, identity, work):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from fresh_dti.prepare_entity_features import official_model_identity

    output = work / "esm.pt"
    if stage_existing(output, "node_rank_esm_residues_v1", identity) is not None:
        print(json.dumps({"stage": "esm", "cache": "validated", "output": str(output)}), flush=True)
        return
    torch.set_num_threads(args.threads)
    model_identity = official_model_identity(args.esm_model)
    config = AutoConfig.from_pretrained(str(args.esm_model), local_files_only=True)
    if config.hidden_size != 2560:
        raise ValueError("Expected official ESM2 3B width 2560.")
    provenance = {**model_identity, "config_sha256": sha256(args.esm_model / "config.json"),
                  "output": "last_hidden_state_per_residue_exclude_special_tokens", "window_policy": "nonoverlapping_full_coverage_concatenate_v1",
                  "window_residues": WINDOW, "dtype": identity["esm_dtype"], "device": args.device,
                  "frozen": True, "node_extractor_sha256": sha256(__file__)}
    sequence_map = {record["target_id"]: record["protein_sequence"] for record in records}
    row_dir = work / "esm_sequences"
    row_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = {}, []
    for key, sequence in sequence_map.items():
        cached = stage_existing(row_dir / (key + ".pt"), "node_rank_esm_sequence_v1", identity)
        if cached is None:
            missing.append((key, sequence))
        else:
            tensor = cached["features"]
            if tensor.shape != (len(sequence), 2560) or not torch.isfinite(tensor).all():
                raise ValueError("Invalid cached residue feature tensor.")
            rows[key] = tensor
    if missing:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; explicitly use --device cpu with a fresh output cache.")
        print(json.dumps({"stage": "esm", "loading": "official_3B_frozen", "missing_sequences": len(missing)}), flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(args.esm_model), local_files_only=True)
        model = AutoModel.from_pretrained(str(args.esm_model), local_files_only=True,
                                          torch_dtype=getattr(torch, identity["esm_dtype"]), low_cpu_mem_usage=True)
        model = model.to(args.device).eval().requires_grad_(False)
        with torch.inference_mode():
            for index, (key, sequence) in enumerate(missing):
                chunks = []
                for start, window in residue_windows(sequence):
                    inputs = tokenizer(window, return_tensors="pt", truncation=False, add_special_tokens=True).to(args.device)
                    if inputs["input_ids"].shape != (1, len(window) + 2):
                        raise ValueError("ESM tokenizer does not produce exactly one token per input residue plus BOS/EOS.")
                    outputs = model(**inputs)
                    tensor = outputs.last_hidden_state[0, 1:len(window) + 1].detach().cpu().float().clone().contiguous()
                    if tensor.shape != (len(window), 2560) or not torch.isfinite(tensor).all():
                        raise ValueError("ESM produced invalid residue features.")
                    chunks.append(tensor)
                    del inputs, outputs
                full = torch.cat(chunks, dim=0)
                if full.shape != (len(sequence), 2560):
                    raise ValueError("Full sequence residue coverage mismatch.")
                rows[key] = full
                publish(row_dir / (key + ".pt"), {"schema": "node_rank_esm_sequence_v1", "request": identity,
                                                "sequence_sha256": key, "features": full})
                print(json.dumps({"stage": "esm", "row": index + 1, "total_missing": len(missing), "residues": len(sequence)}), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    publish(output, {"schema": "node_rank_esm_residues_v1", "request": identity, "sequences": rows, "provenance": provenance})


def assemble(args, records, identity, work):
    import torch
    pbc = stage_existing(work / "pbc.pt", "node_rank_pbc_atoms_v1", identity)
    esm = stage_existing(work / "esm.pt", "node_rank_esm_residues_v1", identity)
    if pbc is None or esm is None:
        raise ValueError("Both PBC atom and ESM residue stages must complete before assembly.")
    if [sample["pair_id"] for sample in pbc["samples"]] != [record["pair_id"] for record in records]:
        raise ValueError("PBC sample order differs from selected manifest records.")
    for sample, record in zip(pbc["samples"], records):
        sample["sequence_features"] = esm["sequences"][sample["target_id"]]
        if sample["sequence_features"].shape != (len(record["protein_sequence"]), 2560):
            raise ValueError("Residue feature coverage differs from declared sequence.")
        mapping = sample["pocket_to_sequence"]
        if mapping.numel() != len(sample["pocket_features"]) or (mapping >= len(record["protein_sequence"])).any() or (mapping < -1).any():
            raise ValueError("Invalid pocket-to-sequence mapping.")
        for field in ("sequence_features", "ligand_features", "pocket_features", "ligand_pos", "pocket_pos"):
            if not torch.isfinite(sample[field]).all():
                raise ValueError(f"Nonfinite {field} in feature assembly.")
        for field, source_field in (("protein_pdb", "protein_sha256"), ("ligand_sdf", "ligand_sha256")):
            if sha256(record[field]) != sample["source"][source_field]:
                raise ValueError("A structural input changed since the PBC stage.")
    feature_policy = {
        "version": VERSION,
        "esm_repository": esm["provenance"]["repository"], "esm_revision": esm["provenance"]["revision"],
        "esm_config_sha256": esm["provenance"]["config_sha256"], "esm_window_residues": WINDOW,
        "esm_window_policy": esm["provenance"]["window_policy"], "esm_dtype": esm["provenance"]["dtype"],
        "esm_output": esm["provenance"]["output"], "pbc_commit": pbc["provenance"]["official_commit"],
        "pbc_checkpoint_sha256": pbc["provenance"]["checkpoint_sha256"], "pbc_output": pbc["provenance"]["encoder_mode"],
        "pocket_radius_angstrom": 8.0, "pbc_interaction_cutoff_angstrom": 5.0,
        "atom_mapping_policy": pbc["provenance"]["atom_mapping_policy"],
        "node_extractor_sha256": sha256(__file__), "base_extractor_sha256": identity["base_extractor_sha256"],
    }
    artifact = {"schema": SCHEMA, "feature_policy": feature_policy, "samples": pbc["samples"],
                "provenance": {"request": identity, "manifest_path": str(args.manifest.resolve()),
                               "pbc": pbc["provenance"], "esm": esm["provenance"],
                               "pbc_stage_sha256": sha256(work / "pbc.pt"), "esm_stage_sha256": sha256(work / "esm.pt")}}
    publish(args.output, artifact)
    print(json.dumps({"stage": "assembled", "output": str(args.output.resolve()), "samples": len(artifact["samples"]),
                      "sha256": sha256(args.output), "schema": SCHEMA}), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--stage", choices=("all", "pbc", "esm", "assemble"), default="all")
    result.add_argument("--limit", type=int)
    result.add_argument("--repository", type=Path, default=ROOT / "third_party/PBCNet2.0")
    result.add_argument("--esm-model", type=Path, default=Path("/home/dti/models/esm2_3b"))
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--max-atoms", type=int, default=5000)
    result.add_argument("--pbc-python", type=Path, default=Path("/home/dti/.venvs/pbcnet_compat/bin/python"))
    result.add_argument("--esm-python", type=Path, default=Path("/home/dti/.venvs/dti/bin/python"))
    return result


def main():
    args = parser().parse_args()
    args.manifest = reject_quarantine_path(args.manifest)
    args.output = reject_quarantine_path(args.output)
    if min(args.threads, args.max_atoms) < 1:
        raise ValueError("--threads and --max-atoms must be positive.")
    if args.output.exists():
        raise FileExistsError(f"Existing final features will not be overwritten: {args.output}")
    if args.stage == "all":
        # The parent deliberately does not import torch or either large encoder.
        # Process termination frees all CPU/GPU memory before the next stage.
        remaining = sys.argv[1:]
        if "--stage" in remaining:
            position = remaining.index("--stage")
            remaining = remaining[:position] + remaining[position + 2:]
        elif any(value.startswith("--stage=") for value in remaining):
            remaining = [value for value in remaining if not value.startswith("--stage=")]
        for stage, executable in (("pbc", args.pbc_python), ("esm", args.esm_python), ("assemble", args.esm_python)):
            subprocess.run([str(executable), "-m", "node_rank.prepare_features", *remaining, "--stage", stage], cwd=ROOT, check=True)
        return
    records = load_records(args.manifest, args.limit)
    identity = request_identity(args, records)
    work = args.output.parent / (args.output.stem + ".parts")
    work.mkdir(parents=True, exist_ok=True)
    {"pbc": extract_pbc, "esm": extract_esm, "assemble": assemble}[args.stage](args, records, identity, work)


if __name__ == "__main__":
    main()
