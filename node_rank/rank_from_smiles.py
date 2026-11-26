"""Rank posed candidates using one clustered ETKDGv3 library per query compound."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from pbcnet_features import normalize_sequence

from .candidate_features import DEFAULT_CACHE, DEFAULT_MODEL
from .conformer_library import DEFAULT_NUM_CONFORMERS, DEFAULT_RMSD_THRESHOLD, DEFAULT_SEED, build_library
from .prepare_features import ROOT, reject_quarantine_path, sha256
from .train_pdbbind import atomic_json


def read_targets(path):
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError('Provide a JSON array containing at least two receptor records.')
    targets, ids, sequences = [], set(), set()
    for record in payload:
        if not isinstance(record, dict):
            raise ValueError('Each receptor record must be a JSON object.')
        target_id = record.get('id')
        if not isinstance(target_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', target_id):
            raise ValueError('Each receptor needs a filename-safe id, starting with a letter or digit.')
        if target_id.lower() in ids:
            raise ValueError(f'Duplicate receptor id: {target_id}')
        ids.add(target_id.lower())
        sequence = record.get('protein_sequence')
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError(f'{target_id}: provide protein_sequence.')
        sequence = normalize_sequence(sequence)
        if sequence in sequences:
            raise ValueError(f'{target_id}: the same protein sequence already has a receptor in this run.')
        sequences.add(sequence)
        target = dict(record, protein_sequence=sequence)
        target['target_name'] = record.get('target_name', target_id)
        if not isinstance(target['target_name'], str) or not target['target_name'].strip():
            raise ValueError(f'{target_id}: target_name must be nonempty text.')
        for field in ('box_center', 'box_size'):
            values = record.get(field)
            if (not isinstance(values, list) or len(values) != 3
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                    or (field == 'box_size' and min(values) <= 0)):
                raise ValueError(f'{target_id}: invalid {field}, expected three finite numbers and positive sizes.')
        for field, suffix in (('protein_pdb', '.pdb'), ('receptor_pdbqt', '.pdbqt')):
            if field == 'receptor_pdbqt' and field not in record:
                continue
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{target_id}: {field} must contain a structure path.')
            source = reject_quarantine_path(path.parent / value)
            if not source.is_file() or source.suffix.lower() != suffix:
                raise ValueError(f'{target_id}: expected a {suffix} file at {source}.')
            target[field] = str(source)
        targets.append(target)
    return targets


def dock_target(target, conformers, poses, args):
    target_id = target['id']
    pose = poses / f'{target_id}.sdf'
    command = [str(args.docking_python), '-m', 'node_rank.rigid_placement',
               '--conformers', str(conformers), '--protein', target['protein_pdb'],
               '--output', str(pose), '--center', *map(str, target['box_center']),
               '--size', *map(str, target['box_size']), '--seed', str(args.seed),
               '--exhaustiveness', str(args.exhaustiveness), '--cpu', str(args.dock_cpu)]
    if 'receptor_pdbqt' in target:
        command.extend(['--receptor-pdbqt', target['receptor_pdbqt']])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log = poses / f'{target_id}.log'
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode or not pose.is_file():
        lines = (result.stderr.strip() or result.stdout.strip()).splitlines()
        detail = lines[-1] if lines else f'process exited with status {result.returncode}'
        return {'id': target_id, 'status': 'placement_failed', 'error': detail, 'log': f'poses/{log.name}'}
    placement = json.loads(result.stdout.strip().splitlines()[-1])
    return {'id': target_id, 'status': 'pose_available', 'ligand_sdf': f'poses/{pose.name}',
            'ligand_sdf_sha256': sha256(pose), 'placement': placement, 'log': f'poses/{log.name}'}


def run(args):
    target_path = reject_quarantine_path(args.targets)
    destination = reject_quarantine_path(args.output_dir)
    checkpoint = reject_quarantine_path(args.checkpoint)
    if destination.exists():
        raise FileExistsError(f'Choose a new output directory: {destination}')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not args.docking_python.is_file():
        raise FileNotFoundError(f'Pass --docking-python pointing to the installed Vina/Meeko environment: {args.docking_python}')
    targets = read_targets(target_path)
    destination.mkdir(parents=True, exist_ok=False)
    status_path = destination / 'run.json'
    report = {'status': 'generating_conformers', 'target_manifest': str(target_path),
              'target_manifest_sha256': sha256(target_path), 'checkpoint_sha256': sha256(checkpoint),
              'requested_targets': len(targets), 'rdkit_version': version('rdkit'), 'targets': []}
    atomic_json(status_path, report)
    try:
        library_dir = destination / 'conformers'
        library = build_library(args.smiles, library_dir, num_conformers=args.num_conformers,
                                rmsd_threshold=args.rmsd_threshold, seed=args.seed)
        conformer_sdf = library_dir / library['representatives_file']
        report.update(status='placing_conformers', canonical_smiles=library['canonical_smiles'],
                      conformer_clusters=len(library['clusters']),
                      conformers_file=conformer_sdf.relative_to(destination).as_posix(),
                      conformer_sdf_sha256=sha256(conformer_sdf))
        print(f"Generated {len(library['clusters'])} cluster representatives for {len(targets)} receptors.", flush=True)
        atomic_json(status_path, report)
        poses = destination / 'poses'
        poses.mkdir()
        posed_records = []
        for target in targets:
            result = dock_target(target, conformer_sdf, poses, args)
            report['targets'].append(result)
            atomic_json(status_path, report)
            print(f"{target['id']}: {result['status']}", flush=True)
            if result['status'] != 'pose_available':
                continue
            pose_record = {key: target[key] for key in ('target_name', 'protein_sequence', 'protein_pdb',
                                                       'protein_chain', 'keep_hetero_residues') if key in target}
            posed_records.append({**pose_record, 'pair_id': target['id'], 'smiles': library['canonical_smiles'],
                                  'ligand_sdf': result['ligand_sdf'], 'ligand_sdf_sha256': result['ligand_sdf_sha256']})
        manifest = destination / 'posed_candidates.json'
        atomic_json(manifest, posed_records)
        if len(posed_records) < 2:
            raise RuntimeError('Fewer than two receptors have usable placement results. Check the per-target logs.')
        indexed = destination / 'candidate_features.json'
        report['status'] = 'extracting_features'
        atomic_json(status_path, report)
        subprocess.run([sys.executable, '-m', 'node_rank.candidate_features', '--manifest', str(manifest),
                        '--output', str(indexed), '--checkpoint', str(checkpoint), '--cache', str(args.cache),
                        '--esm-model', str(args.esm_model)], cwd=ROOT, check=True)
        report['status'] = 'ranking_targets'
        atomic_json(status_path, report)
        ranking = destination / 'ranking'
        subprocess.run([sys.executable, '-m', 'node_rank.predict', '--features', str(indexed),
                        '--checkpoint', str(checkpoint), '--output-dir', str(ranking), '--device', args.device],
                       cwd=ROOT, check=True)
        report.update(status='complete' if len(posed_records) == len(targets) else 'complete_with_unplaced_targets',
                      ranked_targets=len(posed_records), ranking_file='ranking/scores.csv')
    except (Exception, KeyboardInterrupt) as error:
        report.update(failed_stage=report['status'], status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(status_path, report)
    print(f'Target ranks and raw Model 2 scores: {destination / report["ranking_file"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smiles', required=True)
    parser.add_argument('--targets', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--num-conformers', type=int, default=DEFAULT_NUM_CONFORMERS)
    parser.add_argument('--rmsd-threshold', type=float, default=DEFAULT_RMSD_THRESHOLD,
                        help='Max within-cluster heavy-atom RMSD in angstroms.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--exhaustiveness', type=int, default=8)
    parser.add_argument('--dock-cpu', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--esm-model', type=Path, default=Path('/home/dti/models/esm2_3b'))
    parser.add_argument('--docking-python', type=Path, default=Path.home() / '.venvs/docking_pb/bin/python')
    args = parser.parse_args()
    if min(args.exhaustiveness, args.dock_cpu) < 1:
        parser.error('--exhaustiveness and --dock-cpu must be positive.')
    if not 1 <= args.seed <= 2**31 - 1:
        parser.error('--seed must be a positive 32-bit signed integer for reproducible rigid placement.')
    run(args)


if __name__ == '__main__':
    main()
