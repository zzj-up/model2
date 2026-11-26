"""Extract frozen official PBCNet2.0 embeddings for explicitly posed DTI pairs."""

import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from Bio.Align import PairwiseAligner
from Bio.PDB import Chain, Model, PDBIO, PDBParser, Residue, Structure
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from rdkit import Chem
from scipy.spatial import cKDTree

from pbcnet_features import (
    FEATURE_DIM,
    GRAPH_PREPROCESSING_VERSION,
    SCHEMA,
    canonical_smiles,
    encoder_identity,
    normalize_sequence,
    sha256_file,
    structure_key,
)


ROOT = Path(__file__).resolve().parent
OFFICIAL_COMMIT = "b3563f119864e51177af363a690419214d624194"


def load_official_encoder(repository):
    repository = repository.resolve()
    provenance_file = repository / "provenance.json"
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    if provenance.get("official_commit") != OFFICIAL_COMMIT:
        raise ValueError("The official PBCNet2.0 source commit differs from the pinned version.")
    source_hashes = provenance.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("The official source provenance must include source_sha256 hashes.")
    required = {"Graph2pickle.py"}
    required.update(str(path.relative_to(repository)).replace(os.sep, "/") for path in (repository / "model_code").rglob("*.py"))
    if not required.issubset(source_hashes):
        raise ValueError("The official source hash inventory is incomplete.")
    for relative in required:
        if sha256_file(repository / relative) != source_hashes[relative]:
            raise ValueError(f"An official PBCNet2.0 source file changed: {relative}")
    checkpoint = repository / "PBCNet2.pth"
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != provenance.get("checkpoint_sha256"):
        raise ValueError("The official PBCNet2.0 checkpoint checksum differs from provenance.")
    sys.path.insert(0, str(repository / "model_code"))
    sys.path.insert(0, str(repository))
    graph_module = importlib.import_module("Graph2pickle")
    importlib.import_module("models.readout")
    # This full-module checkpoint is trusted only after verifying the pinned
    # official source and checkpoint above. Extracted features use safe tensors.
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if model.__class__.__name__ != "PBCNetv2" or not hasattr(model, "_readout"):
        raise ValueError("The official checkpoint does not contain a PBCNetv2 encoder.")
    if model.hidden_channels != FEATURE_DIM:
        raise ValueError("The official PBCNet2.0 embedding width differs from the DTI interface.")
    model.eval().requires_grad_(False)
    return model, graph_module, {
        "official_repository": "https://github.com/YuJie-0202/PBCNet2.0",
        "official_commit": OFFICIAL_COMMIT,
        "checkpoint_sha256": checkpoint_hash,
        "graph_preprocessing_version": GRAPH_PREPROCESSING_VERSION,
        "source_sha256": source_hashes,
        "extractor_sha256": sha256_file(__file__),
        "encoder_mode": "frozen_single_complex_readout",
        "original_pairwise_prediction_head_used": False,
        "pocket_radius_angstrom": 8.0,
        "package_versions": {name: version(name) for name in ("torch", "dgl", "numpy", "scipy", "rdkit", "biopython")},
    }


def read_pose(path, pose_index, smiles):
    if not isinstance(pose_index, int) or isinstance(pose_index, bool) or pose_index < 0:
        raise ValueError("pose_index must be a nonnegative integer.")
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    if pose_index >= len(supplier):
        raise ValueError(f"Pose {pose_index} is absent from {path}.")
    molecule = supplier[pose_index]
    if molecule is None or molecule.GetNumAtoms() == 0 or molecule.GetNumConformers() != 1:
        raise ValueError(f"The requested ligand pose is invalid: {path}")
    molecule = Chem.RemoveHs(molecule)
    actual_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if actual_smiles != canonical_smiles(smiles):
        raise ValueError(
            f"Posed ligand identity/stereochemistry differs from manifest SMILES: {path}. "
            f"Pose contains {actual_smiles}."
        )
    conformer = molecule.GetConformer()
    coordinates = np.asarray(conformer.GetPositions(), dtype=np.float64)
    if not conformer.Is3D() or not np.isfinite(coordinates).all():
        raise ValueError(f"A finite three-dimensional bound pose is required: {path}")
    return molecule, coordinates


def heavy_atoms(residue):
    atoms = []
    for atom in residue.get_atoms():
        atom = atom.selected_child if hasattr(atom, "selected_child") else atom
        if atom.element.upper() not in {"H", "D"}:
            atoms.append(atom)
    return atoms


def chain_alignment(chain, sequence):
    residues = [residue for residue in chain if is_aa(residue, standard=False)]
    chain_sequence = "".join(seq1(residue.resname, custom_map={"MSE": "M"}) for residue in residues)
    if not chain_sequence:
        return None
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -3
    aligner.open_gap_score = -8
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(chain_sequence, sequence)[0]
    matches = 0
    aligned = 0
    for (chain_start, chain_end), (target_start, target_end) in zip(*alignment.aligned):
        chain_part = chain_sequence[chain_start:chain_end]
        target_part = sequence[target_start:target_end]
        matches += sum(a == b for a, b in zip(chain_part, target_part))
        aligned += len(chain_part)
    return {
        "chain": chain.id,
        "identity": matches / max(aligned, 1),
        "chain_coverage": aligned / len(chain_sequence),
        "chain_sequence_sha256": hashlib.sha256(chain_sequence.encode("ascii")).hexdigest(),
        "aligned_chain_segments": alignment.aligned[0].tolist(),
        "aligned_target_segments": alignment.aligned[1].tolist(),
    }


def crop_pocket(protein_path, coordinates, sequence, destination, record, max_atoms):
    structure = PDBParser(QUIET=True).get_structure("receptor", str(protein_path))
    source_model = next(structure.get_models())
    requested_chain = record.get("protein_chain")
    alignments = []
    allowed_chains = set()
    for chain in source_model:
        if requested_chain is not None and chain.id != requested_chain:
            continue
        alignment = chain_alignment(chain, sequence)
        if alignment and alignment["identity"] >= 0.95 and alignment["chain_coverage"] >= 0.8:
            allowed_chains.add(chain.id)
            alignments.append(alignment)
    if not allowed_chains:
        raise ValueError(
            f"No receptor chain sufficiently matches the supplied target sequence: {protein_path}. "
            "Required aligned identity >= 0.95 and chain coverage >= 0.80."
        )
    keep_hetero = record.get("keep_hetero_residues", [])
    if not isinstance(keep_hetero, list) or any(not isinstance(name, str) for name in keep_hetero):
        raise ValueError("keep_hetero_residues must be an explicit list of residue names.")
    tree = cKDTree(coordinates)
    pocket = Structure.Structure("pocket")
    pocket_model = Model.Model(0)
    pocket.add(pocket_model)
    residue_count = 0
    atom_count = 0
    nearest = float("inf")
    for chain in source_model:
        pocket_chain = Chain.Chain(chain.id)
        for residue in chain:
            amino_acid = is_aa(residue, standard=False) and chain.id in allowed_chains
            if not amino_acid and residue.resname not in keep_hetero:
                continue
            atoms = heavy_atoms(residue)
            if not atoms:
                continue
            positions = np.asarray([atom.coord for atom in atoms], dtype=np.float64)
            if not np.isfinite(positions).all():
                raise ValueError(f"Non-finite receptor coordinates: {protein_path}")
            distance = float(tree.query(positions, k=1)[0].min())
            if distance > 8.0:
                continue
            nearest = min(nearest, distance)
            copied_residue = Residue.Residue(residue.id, residue.resname, residue.segid)
            for atom in atoms:
                copied_atom = atom.copy()
                copied_atom.set_altloc(" ")
                copied_atom.disordered_flag = 0
                copied_residue.add(copied_atom)
            pocket_chain.add(copied_residue)
            atom_count += len(atoms)
            residue_count += 1
        if len(pocket_chain):
            pocket_model.add(pocket_chain)
    if residue_count == 0 or nearest > 5.0:
        raise ValueError("The ligand has no receptor contacts within the encoder's 5 Å interaction cutoff.")
    if atom_count + len(coordinates) > max_atoms:
        raise ValueError(f"The complex exceeds the explicit {max_atoms}-atom safety limit.")
    writer = PDBIO()
    writer.set_structure(pocket)
    writer.save(str(destination))
    return {
        "chain_alignment": alignments,
        "pocket_residues": residue_count,
        "pocket_heavy_atoms": atom_count,
        "nearest_protein_ligand_distance_angstrom": nearest,
        "kept_hetero_residues": keep_hetero,
        "pocket_sha256": sha256_file(destination),
        "alternate_location_policy": "selected_atom_conformer_written_with_blank_altloc",
    }


def build_complex(record, manifest_directory, temporary_directory, graph_module, max_atoms):
    sequence = normalize_sequence(record["protein_sequence"])
    smiles = canonical_smiles(record["smiles"])
    protein_path = (manifest_directory / record["protein_pdb"]).resolve()
    ligand_path = (manifest_directory / record["ligand_sdf"]).resolve()
    protein_hash, ligand_hash = sha256_file(protein_path), sha256_file(ligand_path)
    pose_index = record.get("pose_index", 0)
    ligand, coordinates = read_pose(ligand_path, pose_index, smiles)
    pocket_path = temporary_directory / "pocket.pdb"
    pose_path = temporary_directory / "pose.sdf"
    preparation = crop_pocket(protein_path, coordinates, sequence, pocket_path, record, max_atoms)
    with Chem.SDWriter(str(pose_path)) as writer:
        writer.write(ligand)
    graph = graph_module.Graph_Information(str(pose_path), str(pocket_path))
    graph.edges["int"].data["bond_scalar"] = graph.edges["int"].data["bond_scalar"].clamp(0, 4)
    if not graph.num_edges("int") or not torch.isfinite(graph.nodes["atom"].data["pos"]).all():
        raise ValueError("The official graph builder produced an invalid complex graph.")
    source, destination = graph.edges(etype="int")
    positions = graph.nodes["atom"].data["pos"]
    if ((positions[source] - positions[destination]).square().sum(dim=1) <= 0).any():
        raise ValueError("The complex contains an interaction edge with coincident atoms.")
    if (sha256_file(protein_path), sha256_file(ligand_path)) != (protein_hash, ligand_hash):
        raise RuntimeError("A structural input changed during feature extraction.")
    metadata = {
        "key": structure_key(smiles, sequence),
        "smiles": smiles,
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "protein_pdb": str(protein_path),
        "protein_sha256": protein_hash,
        "ligand_sdf": str(ligand_path),
        "ligand_sha256": ligand_hash,
        "pose_index": pose_index,
        "graph_atoms": int(graph.num_nodes("atom")),
        "graph_interaction_edges": int(graph.num_edges("int")),
        **preparation,
    }
    return graph, metadata


def extract(args):
    if args.output.exists():
        raise FileExistsError(f"Choose a new feature output path: {args.output}")
    manifest_hash = sha256_file(args.manifest)
    records = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError("The structure manifest must be a nonempty JSON list.")
    keys = []
    for record in records:
        if not isinstance(record, dict) or not {"smiles", "protein_sequence", "protein_pdb", "ligand_sdf"}.issubset(record):
            raise ValueError("Each structure record needs smiles, protein_sequence, protein_pdb and ligand_sdf.")
        if any(not isinstance(record[field], str) or not record[field] for field in ("protein_pdb", "ligand_sdf")):
            raise ValueError("Structure paths must be nonempty strings.")
        keys.append(structure_key(record["smiles"], record["protein_sequence"]))
    if len(set(keys)) != len(keys):
        raise ValueError("Each compound-target key requires exactly one explicitly selected pose.")
    torch.set_num_threads(args.threads)
    encoder, graph_module, provenance = load_official_encoder(args.repository)
    embeddings = []
    metadata = []
    with torch.inference_mode(), tempfile.TemporaryDirectory(prefix="dti_pbcnet_") as directory:
        temporary_directory = Path(directory)
        for index, record in enumerate(records):
            graph, prepared = build_complex(record, args.manifest.resolve().parent, temporary_directory, graph_module, args.max_atoms)
            embedding, _ = encoder._readout(graph, False)
            embedding = embedding.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            if embedding.shape != (FEATURE_DIM,) or not torch.isfinite(embedding).all():
                raise ValueError(f"PBCNet2.0 produced an invalid embedding for structure record {index}.")
            embeddings.append(embedding)
            metadata.append(prepared)
            print(json.dumps({"record": index + 1, "total": len(records), "atoms": prepared["graph_atoms"], "edges": prepared["graph_interaction_edges"]}), flush=True)
    if sha256_file(args.manifest) != manifest_hash:
        raise RuntimeError("The structure manifest changed during feature extraction.")
    provenance.update(manifest_sha256=manifest_hash, manifest_path=str(args.manifest.resolve()))
    encoder_identity(provenance)
    artifact = {
        "schema": SCHEMA,
        "feature_dim": FEATURE_DIM,
        "keys": keys,
        "embeddings": torch.stack(embeddings),
        "records": metadata,
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".tmp", delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        torch.save(artifact, temporary_path)
        # A hard link publishes the complete artifact without overwriting a
        # concurrently created result. Both files are on the same filesystem.
        os.link(temporary_path, args.output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output.resolve()), "pairs": len(keys), "feature_dim": FEATURE_DIM, "checkpoint_sha256": provenance["checkpoint_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON list of explicitly posed compound-target pairs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=ROOT / "third_party/PBCNet2.0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-atoms", type=int, default=5000)
    args = parser.parse_args()
    if min(args.threads, args.max_atoms) < 1:
        parser.error("--threads and --max-atoms must be positive")
    extract(args)


if __name__ == "__main__":
    main()
