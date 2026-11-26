"""Build a reusable library of representative isolated ligand conformers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolAlign
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


DEFAULT_NUM_CONFORMERS = 32
DEFAULT_RMSD_THRESHOLD = 0.75
DEFAULT_SEED = 20261125
REPRESENTATIVES_FILENAME = 'representatives.sdf'
MANIFEST_FILENAME = 'conformer_manifest.json'


def _cluster(molecule, threshold):
    # RMSD alignment works on a separate molecule. Export the original coordinates.
    heavy = Chem.RemoveAllHs(molecule)
    count = heavy.GetNumConformers()
    if count == 1:
        return [(0, [0], 0.0)]

    # RDKit returns lower-triangular row order; SciPy expects upper-triangular row order.
    lower_triangle = np.tril_indices(count, -1)
    distances = np.zeros((count, count), dtype=np.float64)
    distances[lower_triangle] = rdMolAlign.GetAllConformerBestRMS(heavy, numThreads=1)
    distances += distances.T.copy()
    hierarchy = linkage(squareform(distances), method='complete')
    members_by_label = defaultdict(list)
    for index, label in enumerate(fcluster(hierarchy, threshold, criterion='distance')):
        members_by_label[label].append(index)

    clusters = []
    for members in sorted(members_by_label.values(), key=lambda group: group[0]):
        pairwise = distances[np.ix_(members, members)]
        medoid = members[int(np.argmin(pairwise.sum(axis=1)))]
        clusters.append((medoid, members, float(pairwise.max())))
    return clusters


def build_library(smiles, output_dir, *, num_conformers=DEFAULT_NUM_CONFORMERS,
                  rmsd_threshold=DEFAULT_RMSD_THRESHOLD, seed=DEFAULT_SEED):
    """Write one ETKDGv3 medoid per complete-linkage RMSD cluster and return its manifest."""
    if not isinstance(num_conformers, int) or isinstance(num_conformers, bool) or num_conformers < 1:
        raise ValueError('num_conformers must be a positive integer.')
    if not math.isfinite(rmsd_threshold) or rmsd_threshold <= 0:
        raise ValueError('rmsd_threshold must be finite and greater than zero.')
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**31 - 1:
        raise ValueError('seed must be a nonnegative 32-bit signed integer.')
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Use a fresh conformer library output directory: {output}')

    molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
    if not molecule or not molecule.GetNumHeavyAtoms() or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError('Provide a valid SMILES for one connected ligand.')
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    # Normalize atom order so alternative SMILES for the same ligand reuse the same seed.
    molecule = Chem.AddHs(Chem.MolFromSmiles(canonical_smiles))
    embedding = AllChem.ETKDGv3()
    embedding.randomSeed = seed
    embedding.numThreads = 1
    embedding.enforceChirality = True
    embedding.pruneRmsThresh = -1.0
    conformer_ids = list(AllChem.EmbedMultipleConfs(molecule, int(num_conformers), embedding))
    if not conformer_ids:
        raise ValueError('ETKDGv3 could not generate a 3D conformer for this ligand.')
    cluster_rows = []
    for cluster_id, (representative, members, diameter) in enumerate(_cluster(molecule, rmsd_threshold), start=1):
        cluster_rows.append({
            'cluster_id': cluster_id,
            'representative_conformer_id': conformer_ids[representative],
            'member_conformer_ids': [conformer_ids[index] for index in members],
            'max_heavy_atom_rmsd_angstrom': diameter,
        })
    manifest = {
        'schema': 'model2_ligand_conformer_library_v1',
        'canonical_smiles': canonical_smiles,
        'generator': 'ETKDGv3',
        'rdkit_version': rdBase.rdkitVersion,
        'seed': seed,
        'requested_conformer_count': num_conformers,
        'generated_conformer_count': len(conformer_ids),
        'representative_count': len(cluster_rows),
        'distance_metric': 'symmetry_corrected_aligned_heavy_atom_rmsd_angstrom',
        'clustering': 'complete_linkage',
        'rmsd_threshold_angstrom': rmsd_threshold,
        'representative_selection': 'minimum_sum_of_rmsds_to_cluster_members',
        'representatives_file': REPRESENTATIVES_FILENAME,
        'clusters': cluster_rows,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.conformers_', dir=output.parent) as temporary:
        staged = Path(temporary) / 'library'
        staged.mkdir()
        sdf_path = staged / REPRESENTATIVES_FILENAME
        with Chem.SDWriter(str(sdf_path)) as writer:
            writer.SetForceV3000(True)
            for row in cluster_rows:
                molecule.SetProp('_Name', f'cluster_{row["cluster_id"]:04d}')
                molecule.SetIntProp('cluster_id', row['cluster_id'])
                molecule.SetIntProp('representative_conformer_id', row['representative_conformer_id'])
                writer.write(molecule, confId=row['representative_conformer_id'])

        with sdf_path.open('rb') as stream:
            count = 0
            for representative in Chem.ForwardSDMolSupplier(stream, removeHs=False):
                if representative is None:
                    raise RuntimeError('The representative SDF could not be read back.')
                if Chem.MolToSmiles(Chem.RemoveHs(representative), isomericSmiles=True) != canonical_smiles:
                    raise ValueError('The 3D output changed the ligand identity recorded by SMILES. '
                                     'Specify any ambiguous stereochemistry in the input SMILES.')
                if not np.isfinite(representative.GetConformer().GetPositions()).all():
                    raise RuntimeError('An exported conformer contains invalid coordinates.')
                count += 1
        if count != len(cluster_rows):
            raise RuntimeError('The exported SDF does not contain one record per cluster.')
        (staged / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        if output.exists() or output.is_symlink():
            raise FileExistsError(f'The output directory was created during processing: {output}')
        staged.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smiles', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--num-conformers', default=DEFAULT_NUM_CONFORMERS, type=int)
    parser.add_argument('--rmsd-threshold', default=DEFAULT_RMSD_THRESHOLD, type=float, help='Maximum intra-cluster RMSD in Å.')
    parser.add_argument('--seed', default=DEFAULT_SEED, type=int)
    args = parser.parse_args(argv)
    manifest = build_library(args.smiles, args.output_dir, num_conformers=args.num_conformers,
                             rmsd_threshold=args.rmsd_threshold, seed=args.seed)
    print(json.dumps({
        'generated_conformers': manifest['generated_conformer_count'],
        'representative_conformers': manifest['representative_count'],
        'output_directory': str(args.output_dir.resolve()),
    }), flush=True)


if __name__ == '__main__':
    main()
