"""Prepare reusable rigid Meeko receptors for defined screening pockets."""

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial import cKDTree


BOX_GUARD_ANGSTROM = 5.0
FRAME_TOLERANCE_ANGSTROM = 0.05
HISTIDINE_POLICY = "neutral_his_oxygen_hbond_geometry_v1"
_HISTIDINES = {"HIS", "HID", "HIE", "HIP"}
_METALS = {"CA", "CD", "CO", "CU", "FE", "MG", "MN", "NI", "ZN"}


@dataclass(frozen=True, slots=True)
class _Atom:
    line: str
    name: str
    residue: str
    residue_id: str
    element: str
    altloc: str
    xyz: tuple


def _read_atoms(protein_pdb):
    atoms = []
    models = 0
    for line in protein_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL "):
            models += 1
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name = line[12:16].strip()
        residue = line[17:20].strip()
        chain = line[21:22].strip()
        number = int(line[22:26])
        insertion = line[26:27].strip()
        element = line[76:78].strip().upper()
        xyz = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
        if not element or not all(map(math.isfinite, xyz)):
            raise ValueError(f"Missing element or invalid coordinate in {chain}:{number}{insertion} {name}")
        atoms.append(_Atom(
            line, name, residue, f"{chain}:{number}{insertion}", element,
            line[16:17].strip(), xyz,
        ))
    if models > 1:
        raise ValueError("Select a single experimental receptor model before screening")
    if not atoms:
        raise ValueError("Receptor PDB contains no ATOM or HETATM records")
    return atoms


def _select_altlocs(atoms):
    alternates = {}
    for atom in atoms:
        if atom.altloc:
            occupancy = float(atom.line[54:60].strip() or 0.0)
            scores = alternates.setdefault(atom.residue_id, {})
            scores[atom.altloc] = scores.get(atom.altloc, 0.0) + occupancy
    selected = {
        residue_id: min(scores, key=lambda location: (-scores[location], location))
        for residue_id, scores in alternates.items()
    }
    return [
        atom for atom in atoms
        if not atom.altloc or atom.altloc == selected[atom.residue_id]
    ], selected


@lru_cache(maxsize=1)
def _meeko_preparation():
    from meeko import MoleculePreparation, ResidueChemTemplates

    return ResidueChemTemplates.create_from_defaults(), MoleculePreparation()


def _histidine_hydrogen_score(ring, neighbors, metals, oxygens, oxygen_xyz):
    nitrogen, first, second = (np.asarray(ring[name].xyz) for name in neighbors)
    towards_first, towards_second = first - nitrogen, second - nitrogen
    first_length, second_length = np.linalg.norm(towards_first), np.linalg.norm(towards_second)
    if min(first_length, second_length) < 0.1:
        return 0.0
    external = -(towards_first / first_length + towards_second / second_length)
    external_length = np.linalg.norm(external)
    if external_length < 1e-6:
        return 0.0
    direction = external / external_length
    hydrogen = nitrogen + 1.01 * direction

    # Prefer to leave a nitrogen available for coordination when a nearby metal
    # is present. Otherwise score geometry to neighboring oxygen acceptors.
    if metals is not None and metals.query_ball_point(nitrogen, 2.7):
        return -1.0
    if oxygens is None:
        return 0.0
    score = 0.0
    for index in oxygens.query_ball_point(hydrogen, 2.7):
        acceptor, residue_id = oxygen_xyz[index]
        if residue_id == ring[neighbors[0]].residue_id:
            continue
        offset = acceptor - hydrogen
        distance = float(np.linalg.norm(offset))
        if distance >= 1.2:
            alignment = max(0.0, float(offset @ direction / distance))
            score = max(score, alignment**2 / distance**2)
    return score


def _histidine_templates(atoms):
    by_residue = {}
    for atom in atoms:
        if atom.residue in _HISTIDINES:
            by_residue.setdefault(atom.residue_id, {})[atom.name] = atom
    oxygen_xyz = [(np.asarray(atom.xyz), atom.residue_id) for atom in atoms if atom.element == "O"]
    metal_xyz = [atom.xyz for atom in atoms if atom.element in _METALS]
    oxygens = cKDTree([xyz for xyz, _ in oxygen_xyz]) if oxygen_xyz else None
    metals = cKDTree(metal_xyz) if metal_xyz else None
    templates = {}
    inferred = {}
    for residue_id, ring in by_residue.items():
        residue_name = next(iter(ring.values())).residue
        if residue_name in {"HID", "HIE", "HIP"}:
            templates[residue_id] = residue_name
        elif {"HD1", "HE2"} & ring.keys():
            state = ("HD1" in ring, "HE2" in ring)
            templates[residue_id] = {(True, True): "HIP", (True, False): "HID", (False, True): "HIE"}[state]
        elif {"CG", "ND1", "CE1", "NE2", "CD2"} <= ring.keys():
            hid = _histidine_hydrogen_score(ring, ("ND1", "CG", "CE1"), metals, oxygens, oxygen_xyz)
            hie = _histidine_hydrogen_score(ring, ("NE2", "CE1", "CD2"), metals, oxygens, oxygen_xyz)
            template = "HID" if hid > hie else "HIE"
            templates[residue_id] = template
            inferred[residue_id] = template
        if residue_id in templates:
            if "OXT" in ring:
                templates[residue_id] = "C" + templates[residue_id]
            elif any(
                names <= ring.keys()
                for names in ({"H1", "H2"}, {"HN1", "HN2"}, {"1H", "2H"})
            ):
                templates[residue_id] = "N" + templates[residue_id]
    return templates, inferred


def _outside_box(coordinates, center, size):
    return np.linalg.norm(np.maximum(0.0, np.abs(coordinates - center) - size / 2), axis=1)


def _prepared_coordinates(pdbqt):
    return np.asarray([
        [float(line[start : start + 8]) for start in (30, 38, 46)]
        for line in pdbqt.splitlines()
        if line.startswith(("ATOM  ", "HETATM")) and line.split()[-1] not in {"H", "HD", "HS"}
    ], dtype=float)


def prepare_receptor(protein_pdb: Path, destination: Path, *, center, size, check_at_pose=False) -> dict:
    """Write one receptor PDBQT in the input frame after validating the pocket."""
    from meeko import PDBQTWriterLegacy, Polymer

    protein_pdb, destination = Path(protein_pdb).resolve(strict=True), Path(destination).resolve()
    if protein_pdb == destination:
        raise ValueError("Write the PDBQT to a different path from the input PDB")
    center, size = np.asarray(center, dtype=float), np.asarray(size, dtype=float)
    if (
        center.shape != (3,) or size.shape != (3,)
        or not np.isfinite([center, size]).all() or (size <= 0).any()
    ):
        raise ValueError("Docking box requires 3 finite coordinates and 3 positive finite sizes")

    atoms, altlocs = _select_altlocs(_read_atoms(protein_pdb))
    heavy_atoms = [atom for atom in atoms if atom.element not in {"H", "D"}]
    if not heavy_atoms:
        raise ValueError("Receptor contains no heavy atoms")
    source = np.asarray([atom.xyz for atom in heavy_atoms])
    near_box = _outside_box(source, center, size) <= BOX_GUARD_ANGSTROM
    if not near_box.any():
        raise ValueError("No receptor heavy atoms are near the requested docking box")

    templates, preparer = _meeko_preparation()
    known_names = templates.residue_templates.keys() | templates.ambiguous.keys()
    unsupported = sorted({
        f"{atom.residue_id} {atom.residue}"
        for atom, near in zip(heavy_atoms, near_box)
        if near and atom.residue not in known_names
    })
    if unsupported and not check_at_pose:
        raise ValueError(f"No installed Meeko template for residues near the docking box: {', '.join(unsupported)}")
    known_atoms = [atom for atom in atoms if atom.residue in known_names]
    histidine_templates, inferred = _histidine_templates(known_atoms)
    pdb = "\n".join(atom.line[:16] + " " + atom.line[17:] for atom in known_atoms) + "\n"
    try:
        polymer = Polymer.from_pdb_string(
            pdb, chem_templates=templates, mk_prep=preparer,
            set_template=histidine_templates, allow_bad_res=True,
        )
        pdbqt, flexible = PDBQTWriterLegacy.write_from_polymer(polymer)
    except (KeyError, RuntimeError, ValueError) as error:
        raise ValueError(f"Meeko could not prepare {protein_pdb.name}: {error}") from error
    prepared = _prepared_coordinates(pdbqt)
    if flexible or not prepared.size or not np.isfinite(prepared).all():
        raise ValueError("Meeko produced no valid rigid receptor coordinates")
    if (cKDTree(source).query(prepared)[0] > FRAME_TOLERANCE_ANGSTROM).any():
        raise ValueError("Meeko changed or added heavy atom coordinates relative to the source PDB")
    missing = cKDTree(prepared).query(source)[0] > FRAME_TOLERANCE_ANGSTROM
    if (missing & near_box).any() and not check_at_pose:
        failed = sorted({heavy_atoms[index].residue_id for index in np.flatnonzero(missing & near_box)})
        raise ValueError(f"Meeko omitted receptor heavy atoms near the docking box: {', '.join(failed)}")
    omitted = sorted({heavy_atoms[index].residue_id for index in np.flatnonzero(missing)})

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".receptor_", dir=destination.parent) as temporary:
        staged = Path(temporary) / "receptor.pdbqt"
        staged.write_text(pdbqt, encoding="ascii")
        staged.replace(destination)
    return {
        "receptor_pdbqt": str(destination),
        "receptor_heavy_atoms": len(prepared),
        "histidine_policy": HISTIDINE_POLICY,
        "histidine_tie_break": "HIE",
        "histidine_templates": histidine_templates,
        "inferred_histidine_templates": inferred,
        "alternate_locations": altlocs,
        "omitted_distal_residues": omitted,
        "omitted_heavy_coordinates": source[missing].tolist() if check_at_pose else [],
        "box_guard_angstrom": BOX_GUARD_ANGSTROM,
        "frame_tolerance_angstrom": FRAME_TOLERANCE_ANGSTROM,
    }
