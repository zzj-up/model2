"""Place a library of rigid RDKit conformers in a specified receptor box."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from rdkit import Chem


def _mapped_coordinates(pdbqt):
    return {
        int(line[6:11]): [float(line[start : start + 8]) for start in (30, 38, 46)]
        for line in pdbqt.splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    }


def _rigid_pdbqt(molecule, preparer, writer):
    setups = preparer.prepare(molecule)
    if len(setups) != 1:
        raise ValueError(f"Meeko produced {len(setups)} ligand preparations, expected one")
    pdbqt, ok, error = writer.write_string(
        setups[0], add_index_map=True, remove_smiles=True
    )
    if not ok:
        raise ValueError(f"Meeko ligand preparation failed: {error}")

    # Retain Meeko charges, atom types and mapping while fixing any residual
    # rotatable bonds that Meeko's chemistry rules can enable.
    lines = pdbqt.splitlines()
    remarks = [line for line in lines if line.startswith("REMARK ")]
    atoms = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    rigid = "\n".join([*remarks, "ROOT", *atoms, "ENDROOT", "TORSDOF 0", ""])
    index_prefix = "REMARK INDEX MAP "
    numbers = [
        int(value)
        for line in remarks
        if line.startswith(index_prefix)
        for value in line[len(index_prefix) :].split()
    ]
    if not numbers or len(numbers) % 2:
        raise ValueError("Meeko ligand preparation lacks a valid atom index map")
    atom_map = {index - 1: serial for index, serial in zip(numbers[::2], numbers[1::2])}
    if set(atom_map.values()) != set(_mapped_coordinates(rigid)) or len(atom_map) != len(atoms):
        raise ValueError("Meeko ligand preparation has inconsistent atom serials")
    heavy_atoms = {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1}
    if not heavy_atoms or not heavy_atoms <= atom_map.keys():
        raise ValueError("Meeko ligand preparation omitted mapped heavy atoms")
    return rigid, atom_map


def _restore_pose(molecule, atom_map, vina_pose):
    original = molecule.GetConformer().GetPositions()
    posed_atoms = _mapped_coordinates(vina_pose)
    if set(posed_atoms) != set(atom_map.values()):
        raise ValueError("Vina output atom serials differ from the prepared ligand")
    source = original[list(atom_map)]
    target = np.asarray([posed_atoms[serial] for serial in atom_map.values()])
    if not np.isfinite(target).all():
        raise ValueError("Vina output contains non-finite coordinates")

    # Fit a proper rigid transform and transfer it to every original atom,
    # including nonpolar hydrogens that Meeko merged during preparation.
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd((source - source_center).T @ (target - target_center))
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    rotation = left @ right
    translation = target_center - source_center @ rotation
    residuals = source @ rotation + translation - target
    rmsd = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    if rmsd > 0.01:
        raise ValueError(f"Vina output differs from a rigid transform by {rmsd:.4f} Å")

    result = Chem.Mol(molecule)
    conformation = result.GetConformer()
    for index, coordinates in enumerate(original @ rotation + translation):
        conformation.SetAtomPosition(index, tuple(coordinates))
    return result, rmsd


def _prepare_receptor(protein_pdb, destination):
    command = Path(sys.executable).with_name("mk_prepare_receptor.py")
    if not command.is_file():
        raise FileNotFoundError(
            "mk_prepare_receptor.py is missing from the active Python environment. "
            "Use the docking_pb environment or provide --receptor-pdbqt."
        )
    result = subprocess.run(
        [str(command), "--read_pdb", str(protein_pdb), "-p", str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not destination.is_file():
        diagnostics = (result.stderr or result.stdout).splitlines()
        details = "\n".join(diagnostics[-8:]).strip()
        raise ValueError(
            f"Meeko could not prepare {protein_pdb.name}. Provide a prepared "
            "receptor PDBQT in the same coordinate frame, or correct the receptor PDB."
            f"\n{details}"
        )
    return destination


def _validate_receptor_frame(protein_pdb, receptor_pdbqt):
    from scipy.spatial import cKDTree

    source = _mapped_coordinates(protein_pdb.read_text(encoding="utf-8", errors="replace"))
    prepared_lines = [
        line for line in receptor_pdbqt.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith(("ATOM  ", "HETATM")) and line.split()[-1] not in {"H", "HD", "HS"}
    ]
    prepared = _mapped_coordinates("\n".join(prepared_lines))
    if not source or not prepared:
        raise ValueError("Receptor PDB and PDBQT both need atomic coordinates")
    distances, _ = cKDTree(list(source.values())).query(list(prepared.values()))
    if np.max(distances) > 0.05:
        raise ValueError(
            "Receptor PDBQT has heavy atoms more than 0.05 Å from any source PDB atom. "
            "Provide the matching PDB and PDBQT in the same coordinate frame."
        )


def place_target(
    sdf_path, protein_pdb, output_sdf, *, center, size, receptor_pdbqt=None,
    seed=20261125, exhaustiveness=8, cpu=1, max_evals=0,
):
    """Place one rigid pose per conformer and write the best Vina pose as SDF."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from vina import Vina

    sdf_path = Path(sdf_path).resolve(strict=True)
    protein_pdb = Path(protein_pdb).resolve(strict=True)
    output_sdf = Path(output_sdf).resolve()
    if output_sdf == sdf_path or output_sdf == protein_pdb:
        raise ValueError("Output SDF must differ from ligand and receptor input paths")
    center, size = tuple(center), tuple(size)
    if (
        len(center) != 3 or len(size) != 3
        or not all(math.isfinite(value) for value in (*center, *size))
        or min(size) <= 0
    ):
        raise ValueError("Docking box requires finite center coordinates and positive sizes")
    if seed < 1 or seed > 2**31 - 1 or exhaustiveness < 1 or cpu < 1 or max_evals < 0:
        raise ValueError("Use a positive 32-bit seed, exhaustiveness and CPU count, and nonnegative max_evals")

    preparer = MoleculePreparation(rigid_macrocycles=True)
    best = None
    ligand_identity = None
    searched = set()
    with TemporaryDirectory(prefix="model2_rigid_") as temporary:
        if receptor_pdbqt is None:
            receptor = _prepare_receptor(protein_pdb, Path(temporary) / "receptor.pdbqt")
        else:
            receptor = Path(receptor_pdbqt).resolve(strict=True)
            if receptor == output_sdf:
                raise ValueError("Output SDF must differ from the receptor PDBQT path")
            _validate_receptor_frame(protein_pdb, receptor)

        engine = Vina(sf_name="vina", seed=seed, cpu=cpu, verbosity=0)
        engine.set_receptor(str(receptor))
        engine.compute_vina_maps(
            center=center, box_size=size, force_even_voxels=True,
        )
        for record, molecule in enumerate(Chem.SDMolSupplier(str(sdf_path), removeHs=False), 1):
            if molecule is None or not molecule.HasProp("cluster_id"):
                raise ValueError(f"SDF record {record} has invalid chemistry or no cluster_id")
            cluster_id = molecule.GetIntProp("cluster_id")
            if (
                cluster_id <= 0 or cluster_id in searched or molecule.GetNumConformers() != 1
                or not molecule.GetConformer().Is3D()
                or not np.isfinite(molecule.GetConformer().GetPositions()).all()
            ):
                raise ValueError(f"SDF record {record} has an invalid cluster ID or 3D geometry")
            identity = Chem.MolToSmiles(molecule, isomericSmiles=True)
            if ligand_identity is not None and identity != ligand_identity:
                raise ValueError(f"SDF record {record} does not represent the same ligand")
            ligand_identity = identity
            try:
                ligand, atom_map = _rigid_pdbqt(molecule, preparer, PDBQTWriterLegacy)
                engine.set_ligand_from_string(ligand)
                engine.dock(exhaustiveness=exhaustiveness, n_poses=1, max_evals=max_evals)
                energies = engine.energies(n_poses=1)
                if not energies.size or not math.isfinite(energies[0, 0]):
                    raise ValueError("Vina returned no finite docking energy")
                energy = float(energies[0, 0])
                pose, rmsd = _restore_pose(molecule, atom_map, engine.poses(n_poses=1))
            except (RuntimeError, ValueError) as error:
                raise ValueError(f"Cluster {cluster_id} rigid placement failed: {error}") from error
            searched.add(cluster_id)
            if best is None or (energy, cluster_id) < best[:2]:
                best = (energy, cluster_id, pose, rmsd)

    if best is None:
        raise ValueError("Conformer SDF contains no clusters")
    energy, cluster_id, pose, rmsd = best
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".rigid_pose_", dir=output_sdf.parent) as temporary:
        staged_pose = Path(temporary) / "pose.sdf"
        with Chem.SDWriter(str(staged_pose)) as writer:
            writer.SetForceV3000(True)
            writer.write(pose)
        staged_pose.replace(output_sdf)
    return {
        "cluster_id": cluster_id,
        "searched_conformers": len(searched),
        "vina_energy_kcal_mol": energy,
        "rigid_fit_rmsd_angstrom": rmsd,
        "pose_sdf": str(output_sdf),
        "seed": seed,
        "exhaustiveness_per_conformer": exhaustiveness,
        "max_evaluations_per_conformer": max_evals,
        "cpu": cpu,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conformers", required=True, type=Path)
    parser.add_argument("--protein", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--center", nargs=3, required=True, type=float)
    parser.add_argument("--size", nargs=3, required=True, type=float)
    parser.add_argument("--receptor-pdbqt", type=Path)
    parser.add_argument("--seed", default=20261125, type=int)
    parser.add_argument("--exhaustiveness", default=8, type=int)
    parser.add_argument("--cpu", default=1, type=int)
    args = parser.parse_args()
    try:
        result = place_target(
            args.conformers, args.protein, args.output, center=args.center, size=args.size,
            receptor_pdbqt=args.receptor_pdbqt, seed=args.seed,
            exhaustiveness=args.exhaustiveness, cpu=args.cpu,
        )
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Rigid placement failed: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
