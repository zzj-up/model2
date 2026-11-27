"""Prepare isolated conformers and independently proposed receptor pockets for distance learning."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree
import torch

from .conformer_library import DEFAULT_SEED, _cluster


SCHEMA = "model2_distance_data_v1"
ELEMENTS = (6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14)
HYBRIDIZATIONS = tuple(getattr(Chem.HybridizationType, name) for name in ("SP", "SP2", "SP3", "SP3D", "SP3D2"))
RESIDUES = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
ATOM_NAMES = ("N", "CA", "C", "O", "OXT")
LIGAND_FEATURE_DIM = 25
POCKET_FEATURE_DIM = 39
DEFAULT_MAX_ANCHORS = 128
DEFAULT_BOX_SIZE = 22.0
DEFAULT_PROPOSALS = 64
GRID_SPACING = 2.5
MAX_GRID_POINTS = 250_000


def _one_hot(value, choices):
    return [float(value == item) for item in choices] + [float(value not in choices)]


def _ligand_features(molecule):
    features = []
    for atom in molecule.GetAtoms():
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        features.append(_one_hot(atom.GetAtomicNum(), ELEMENTS) + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS) + [
            atom.GetFormalCharge() / 3.0, float(atom.GetIsAromatic()), float(atom.IsInRing()),
            atom.GetDegree() / 4.0, atom.GetTotalNumHs() / 4.0, float(cip == "R"), float(cip == "S"),
        ])
    return torch.tensor(features, dtype=torch.float32)


def _ligand_graph(molecule, conformer_id=-1):
    molecule = Chem.RemoveAllHs(molecule)
    if molecule.GetNumAtoms() == 0 or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("Expected one connected ligand with heavy atoms.")
    conformer = molecule.GetConformer(conformer_id)
    coordinates = np.asarray(conformer.GetPositions(), dtype=np.float32)
    if not conformer.Is3D() or not np.isfinite(coordinates).all():
        raise ValueError("Ligand conformer requires finite three-dimensional coordinates.")
    pairs = []
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pairs.extend(((first, second), (second, first)))
    return {
        "ligand_pos": torch.from_numpy(coordinates - coordinates.mean(axis=0)),
        "ligand_features": _ligand_features(molecule),
        "ligand_bonds": torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous(),
    }


def _receptor_atoms(path, chain=None):
    """Read heavy protein atoms from the first PDB model, excluding waters and ligands."""
    atoms = {}
    periodic = Chem.GetPeriodicTable()
    with Path(path).open(encoding="ascii", errors="replace") as source:
        for line in source:
            if line.startswith("ENDMDL"):
                break
            if line[:6] != "ATOM  " and not (line[:6] == "HETATM" and line[17:20] == "MSE"):
                continue
            if chain is not None and chain != line[21:22]:
                continue
            atom_name = line[12:16].strip()
            residue = line[17:20].strip()
            element = line[76:78].strip() or ("SE" if residue == "MSE" and atom_name == "SE" else atom_name.lstrip("0123456789")[:1])
            if not element or element.upper() in {"H", "D", "T"}:
                continue
            try:
                number = periodic.GetAtomicNumber(element.title())
            except RuntimeError:
                number = 0
            position = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
            if not all(math.isfinite(value) for value in position):
                raise ValueError(f"Nonfinite receptor atom coordinates in {path}.")
            atom_id = (line[21:22], line[22:27].strip(), residue, atom_name)
            occupancy = float(line[54:60].strip() or "0")
            altloc = line[16:17]
            preference = (occupancy, altloc == " ", altloc == "A")
            if atom_id not in atoms or preference > atoms[atom_id][0]:
                atoms[atom_id] = (preference, position, number)
    if len(atoms) < 4:
        raise ValueError(f"Too few receptor heavy atoms for chain {chain!r} in {path}.")
    coordinates = np.array([record[1] for record in atoms.values()], dtype=np.float32)
    identities = [(record[2], "MET" if key[2] == "MSE" else key[2], key[3]) for key, record in atoms.items()]
    return coordinates, identities


def _pocket_graph(atoms, center, size, max_anchors):
    coordinates, identities = atoms
    center = np.asarray(center, dtype=np.float32)
    size = np.asarray(size, dtype=np.float32)
    if center.shape != (3,) or size.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("Pocket box needs a finite center and three positive side lengths in Å.")
    if max_anchors < 4:
        raise ValueError("At least four receptor anchors must be permitted.")
    offsets = coordinates - center
    eligible = np.flatnonzero(np.all(np.abs(offsets) <= size / 2, axis=1))
    distances = (offsets[eligible] ** 2).sum(axis=1)
    selected = eligible[np.argsort(distances, kind="stable")[:max_anchors]]
    if len(selected) < 4 or np.linalg.matrix_rank(coordinates[selected] - center) < 3:
        raise ValueError("Pocket box does not include a three-dimensional set of receptor anchors.")
    features = []
    for index in selected:
        number, residue, atom_name = identities[index]
        features.append(_one_hot(number, ELEMENTS) + _one_hot(residue, RESIDUES) + _one_hot(atom_name, ATOM_NAMES))
    return {
        "pocket_pos": torch.from_numpy(coordinates[selected]),
        "pocket_features": torch.tensor(features, dtype=torch.float32),
        "pocket_center": torch.from_numpy(center),
        "pocket_size": torch.from_numpy(size),
    }


def make_inference_sample(molecule, protein_pdb, *, box_center, box_size, protein_chain=None,
                          max_anchors=DEFAULT_MAX_ANCHORS, conformer_id=-1):
    """Encode an isolated ligand conformer and a user-supplied receptor pocket box."""
    return _ligand_graph(molecule, conformer_id) | _pocket_graph(
        _receptor_atoms(protein_pdb, protein_chain), box_center, box_size, max_anchors)


def _receptor_proposals(coordinates, limit):
    """Propose spatially distinct cavities using only the receptor heavy atoms."""
    low = coordinates.min(axis=0) - GRID_SPACING
    high = coordinates.max(axis=0) + GRID_SPACING
    spacing = GRID_SPACING
    while np.prod(np.ceil((high - low) / spacing)) > MAX_GRID_POINTS:
        spacing *= 1.25
    axes = [np.arange(start, stop, spacing, dtype=np.float32) for start, stop in zip(low, high)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    tree = cKDTree(coordinates)
    nearest, _ = tree.query(points)
    points = points[(nearest >= 2.3) & (nearest <= 5.5)]
    if not len(points):
        raise ValueError("No receptor-only cavity grid points were found.")
    # Estimate enclosure by counting occupied directions around each unoccupied point.
    directions = np.array([direction for direction in product((-1, 0, 1), repeat=3) if any(direction)], dtype=np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    probes = directions * 5.5
    scores = []
    for start in range(0, len(points), 2048):
        batch = points[start:start + 2048]
        probe_distance, _ = tree.query((batch[:, None, :] + probes).reshape(-1, 3))
        occupied = (probe_distance.reshape(len(batch), -1) <= 2.2).sum(axis=1)
        density = tree.query_ball_point(batch, 7.0, return_length=True)
        scores.append(occupied + density / (density + 100.0))
    ranked = points[np.argsort(-np.concatenate(scores), kind="stable")]
    proposals = []
    available = np.ones(len(ranked), dtype=bool)
    for _ in range(limit):
        remaining = np.flatnonzero(available)
        if not len(remaining):
            break
        center = ranked[remaining[0]]
        proposals.append(center)
        available &= (((ranked - center) ** 2).sum(axis=1) >= 5.5**2)
    return np.array(proposals, dtype=np.float32)


def _conformers(smiles, count, threshold, seed):
    if not isinstance(smiles, str):
        raise ValueError("Manifest SMILES must be a string.")
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None or parsed.GetNumHeavyAtoms() < 1 or len(Chem.GetMolFrags(parsed)) != 1:
        raise ValueError("Manifest requires a valid single-compound SMILES.")
    canonical = Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True)
    ligand = Chem.AddHs(Chem.MolFromSmiles(canonical))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.numThreads = 1
    parameters.enforceChirality = True
    parameters.pruneRmsThresh = -1.0
    conformers = list(AllChem.EmbedMultipleConfs(ligand, count, parameters))
    if not conformers:
        # Random coordinate initialization helps with strained fused rings and macrocycles.
        parameters.useRandomCoords = True
        conformers = list(AllChem.EmbedMultipleConfs(ligand, count, parameters))
    if not conformers:
        raise ValueError("ETKDGv3 could not embed the isolated ligand.")
    clusters = _cluster(ligand, threshold)
    heavy = Chem.RemoveAllHs(ligand)
    graphs = [_ligand_graph(heavy, conformers[index]) for index, _, _ in clusters]
    sample = graphs[0] | {"ligand_conformers": torch.stack([graph["ligand_pos"] for graph in graphs])}
    return heavy, canonical, sample


def _native_coordinates(path, pose_index, ligand, canonical):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    if not isinstance(pose_index, int) or not 0 <= pose_index < len(supplier):
        raise ValueError(f"Invalid reference pose index in {path}.")
    native = supplier[pose_index]
    if native is None or native.GetNumConformers() != 1 or not native.GetConformer().Is3D():
        raise ValueError(f"Expected a valid three-dimensional reference ligand in {path}.")
    for atom in native.GetAtoms():
        atom.SetIntProp("original_sdf_index", atom.GetIdx())
    native = Chem.RemoveAllHs(native)
    if Chem.MolToSmiles(native, canonical=True, isomericSmiles=True) != canonical:
        raise ValueError(f"Reference SDF graph or stereochemistry disagrees with manifest SMILES in {path}.")
    match = native.GetSubstructMatch(ligand, useChirality=True)
    if len(match) != ligand.GetNumAtoms():
        raise ValueError(f"Failed to map all isolated conformer atoms to reference SDF in {path}.")
    coordinates = np.asarray(native.GetConformer().GetPositions(), dtype=np.float32)[list(match)]
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Reference SDF has nonfinite coordinates in {path}.")
    atom_indices = [native.GetAtomWithIdx(index).GetIntProp("original_sdf_index") for index in match]
    return coordinates, torch.tensor(atom_indices, dtype=torch.long)


def _path(value, manifest):
    if os.name == "nt" and value.startswith("/mnt/"):
        parts = PurePosixPath(value).parts
        if len(parts) >= 4 and len(parts[2]) == 1:
            return Path(parts[2].upper() + ":/", *parts[3:])
    path = Path(value)
    return path if path.is_absolute() else Path(manifest).parent / path


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _checked_input(row, name, manifest):
    path = _path(row[name], manifest)
    digest = _sha256(path)
    if name + "_sha256" in row and digest != row[name + "_sha256"]:
        raise ValueError(f"Source changed since the split manifest was written: {path}.")
    return path, digest


def _atomic_write(path, writer):
    with NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, data):
    _atomic_write(path, lambda temporary: temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8"))


def _write_sample(path, sample):
    def save(temporary):
        with temporary.open("wb") as stream:
            torch.save(sample, stream)

    _atomic_write(path, save)


def _prepare_row(task):
    index, row, manifest, output, options = task
    pair_id = row["pair_id"]
    filename = f"sample-{index:06d}.pt"
    destination = Path(output) / filename
    try:
        protein, protein_digest = _checked_input(row, "protein_pdb", manifest)
        sdf, ligand_digest = _checked_input(row, "ligand_sdf", manifest)
        if destination.exists():
            cached = torch.load(destination, map_location="cpu", weights_only=True)
            if cached["source_sha256"] == [protein_digest, ligand_digest] and cached["pair_id"] == pair_id:
                return {"pair_id": pair_id, "file": filename, "proposal_error_angstrom": cached["proposal_error_angstrom"]}
        receptor = _receptor_atoms(protein, row.get("protein_chain"))
        proposals = _receptor_proposals(receptor[0], options["num_proposals"])
        ligand, canonical, inputs = _conformers(row["smiles"], options["num_conformers"], options["rmsd_threshold"], options["seed"])
        native, atom_indices = _native_coordinates(sdf, row.get("pose_index", 0), ligand, canonical)
        # The reference only chooses which precomputed receptor-only proposal is labeled positive.
        # Its coordinates never define the box center, input conformer, anchors, or atom features.
        center_errors = np.linalg.norm(proposals - native.mean(axis=0), axis=1)
        selected = int(center_errors.argmin())
        error = float(center_errors[selected])
        pocket = _pocket_graph(receptor, proposals[selected], [options["box_size"]] * 3, options["max_anchors"])
        label = torch.from_numpy(native)
        distances = torch.linalg.vector_norm(label[:, None] - pocket["pocket_pos"][None], dim=-1)
        affinity = float(row["p_affinity"])
        if not math.isfinite(affinity):
            raise ValueError("Missing finite affinity supervision.")
        permutations = ligand.GetSubstructMatches(ligand, useChirality=True, uniquify=False, maxMatches=256)
        sample = inputs | pocket | {
            "pair_id": pair_id, "compound_group": row["compound_group"], "target_id": row["target_id"],
            "native_ligand_pos": label, "native_atom_indices": atom_indices,
            "ligand_symmetry_permutations": torch.tensor(permutations, dtype=torch.long),
            "target_distances": distances, "affinity": torch.tensor(affinity, dtype=torch.float32),
            "source_sha256": [protein_digest, ligand_digest], "proposal_rank": selected + 1,
            "proposal_error_angstrom": error,
        }
        _write_sample(destination, sample)
        return {"pair_id": pair_id, "file": filename, "proposal_error_angstrom": error}
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        return {"pair_id": pair_id, "error": str(exc)}


class DistanceDataset(torch.utils.data.Dataset):
    """Load cached variable-size samples ordered as in the locked split manifest."""

    def __init__(self, cache_dir):
        self.directory = Path(cache_dir)
        self.manifest = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != SCHEMA or not self.manifest.get("samples"):
            raise ValueError(f"Invalid or empty distance dataset: {self.directory}.")
        self.samples = self.manifest["samples"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return torch.load(self.directory / self.samples[index]["file"], map_location="cpu", weights_only=True)


def prepare_distance_data(manifest, output, *, workers=1, max_anchors=DEFAULT_MAX_ANCHORS,
                          num_conformers=4, num_proposals=DEFAULT_PROPOSALS, box_size=DEFAULT_BOX_SIZE,
                          rmsd_threshold=0.75, seed=DEFAULT_SEED, limit=None):
    if workers < 1 or num_conformers < 1 or num_proposals < 1 or max_anchors < 4 or (limit is not None and limit < 1):
        raise ValueError("Preparation counts must be positive; at least four anchors are required.")
    if not all(math.isfinite(x) and x > 0 for x in (box_size, rmsd_threshold)) or not 0 <= seed < 2**31:
        raise ValueError("Box and RMSD threshold must be positive and seed a nonnegative signed 32-bit value.")
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    rows = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows or len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("Split manifest must be a nonempty list with unique pair IDs.")
    rows = rows[:limit]
    options = dict(max_anchors=max_anchors, box_size=box_size, num_conformers=num_conformers,
                   num_proposals=num_proposals, rmsd_threshold=rmsd_threshold, seed=seed)
    identity = {"schema": SCHEMA, "manifest_sha256": _sha256(manifest), "requested_count": len(rows),
                "feature_source_sha256": _sha256(__file__), "rdkit_version": rdBase.rdkitVersion, **options}
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError(f"Cache settings or manifest changed. Choose a fresh output directory: {output}.")
    _write_json(identity_path, identity)
    tasks = [(index, row, manifest, output, options) for index, row in enumerate(rows)]
    samples, failures = [], []
    print(f"Preparing {len(tasks)} complexes with {workers} workers in {output}", flush=True)

    def collect(results):
        for index, record in enumerate(results, start=1):
            (failures if "error" in record else samples).append(record)
            if index % 25 == 0 or index == len(tasks):
                print(f"{index}/{len(tasks)} complexes processed, {len(samples)} cached and {len(failures)} skipped", flush=True)

    if workers == 1:
        collect(map(_prepare_row, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(_prepare_row, tasks))
    report = identity | {
        "pocket_policy": "protein_only_geometric_proposals_label_matches_nearest_candidate_known_pocket",
        "conformer_policy": "isolated_etkdgv3_symmetry_aware_complete_linkage_rmsd_cluster_medoids",
        "labels": "crystal_heavy_atom_distances_to_candidate_box_receptor_anchors_atom_matched",
        "samples": samples, "failures": failures,
        "proposal_within_4_angstrom": sum(row["proposal_error_angstrom"] <= 4.0 for row in samples),
        "proposal_within_8_angstrom": sum(row["proposal_error_angstrom"] <= 8.0 for row in samples),
    }
    _write_json(output / "index.json", report)
    if not samples:
        raise RuntimeError(f"No distance samples prepared. See {output / 'index.json'} for errors.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--workers", type=int, default=1)
    prepare.add_argument("--max-anchors", type=int, default=DEFAULT_MAX_ANCHORS)
    prepare.add_argument("--num-conformers", type=int, default=4)
    prepare.add_argument("--num-proposals", type=int, default=DEFAULT_PROPOSALS)
    prepare.add_argument("--box-size", type=float, default=DEFAULT_BOX_SIZE)
    prepare.add_argument("--rmsd-threshold", type=float, default=0.75)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--limit", type=int)
    options = vars(parser.parse_args(argv))
    del options["command"]
    prepare_distance_data(**options)


if __name__ == "__main__":
    main()
