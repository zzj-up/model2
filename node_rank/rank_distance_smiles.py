"""Rank the targets of one compound with the distance diffusion model."""

import argparse
import csv
import math
from pathlib import Path

import torch
from rdkit import Chem

from .conformer_library import DEFAULT_NUM_CONFORMERS, DEFAULT_RMSD_THRESHOLD, DEFAULT_SEED, build_library
from .distance_data import make_inference_sample
from .prepare_features import ROOT, reject_quarantine_path, sha256
from .rank_from_smiles import read_targets
from .train_distance_diffusion import load_trained_distance
from .train_pdbbind import atomic_json


RUN_DIR = ROOT / 'node_rank/runs/pdbbind2020_distance_diffusion'
DEFAULT_CHECKPOINT = RUN_DIR / 'best_validation.pt'


def write_ranking(path, rows):
    fields = ('rank', 'target_id', 'target_name', 'score', 'conformers_scored')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        previous_score = None
        for index, row in enumerate(sorted(rows, key=lambda item: (-item['score'], item['target_id'])), 1):
            if row['score'] != previous_score:
                rank = index
            writer.writerow({'rank': rank, **row})
            previous_score = row['score']


@torch.inference_mode()
def run(args):
    destination = reject_quarantine_path(args.output_dir)
    manifest_path = reject_quarantine_path(args.targets)
    checkpoint_path = reject_quarantine_path(args.checkpoint)
    targets = read_targets(manifest_path)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model, checkpoint = load_trained_distance(checkpoint_path, args.device)
    if not checkpoint.get('score_trained'):
        raise ValueError('This checkpoint lacks a trained target score head.')
    model.eval()
    destination.mkdir(parents=True, exist_ok=False)
    status_path = destination / 'run.json'
    report = {'status': 'generating_conformers', 'targets': [], 'target_manifest_sha256': sha256(manifest_path),
              'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha256(checkpoint_path),
              'aggregation': 'arithmetic_mean_of_cluster_representative_scores'}
    atomic_json(status_path, report)
    results = []
    try:
        library_dir = destination / 'conformers'
        library = build_library(args.smiles, library_dir, num_conformers=args.num_conformers,
                                rmsd_threshold=args.rmsd_threshold, seed=args.seed)
        conformers = list(Chem.SDMolSupplier(str(library_dir / library['representatives_file']), removeHs=False))
        if len(conformers) != len(library['clusters']) or any(mol is None for mol in conformers):
            raise ValueError('The clustered SDF is missing usable representatives.')
        report.update(status='scoring', compound_id=library['canonical_smiles'], conformer_clusters=len(conformers))
        atomic_json(status_path, report)
        for target in targets:
            summary = {'id': target['id']}
            try:
                scores = []
                best_score = -math.inf
                representative = None
                for molecule in conformers:
                    sample = make_inference_sample(molecule, Path(target['protein_pdb']),
                                                   box_center=target['box_center'], box_size=target['box_size'],
                                                   protein_chain=target.get('protein_chain'))
                    score = float(model.score_sample(sample))
                    if not math.isfinite(score):
                        raise ValueError('Nonfinite score from the checkpoint.')
                    if score > best_score:
                        best_score = score
                        representative = (molecule.GetIntProp('cluster_id'), sample)
                    scores.append(score)
                if args.export_distances:
                    cluster_id, sample = representative
                    generator = torch.Generator(device=args.device).manual_seed(args.seed)
                    prediction = model.sample_distances(sample, num_steps=args.distance_steps, generator=generator)
                    distance_dir = destination / 'distances'
                    distance_dir.mkdir(exist_ok=True)
                    artifact = distance_dir / f"{target['id']}.pt"
                    torch.save({'distances_angstrom': prediction['distances'].cpu(), 'cluster_id': cluster_id,
                                'ligand_pos': sample['ligand_pos'].cpu(), 'pocket_anchors': sample['pocket_pos'].cpu()},
                               artifact)
                    summary['distance_map'] = artifact.relative_to(destination).as_posix()
                results.append({'target_id': target['id'], 'target_name': target['target_name'],
                                'score': math.fsum(scores) / len(scores), 'conformers_scored': len(scores)})
                summary.update(status='scored', conformers_scored=len(scores))
            except (ValueError, OSError) as error:
                summary.update(status='failed', error=str(error))
            report['targets'].append(summary)
            atomic_json(status_path, report)
            print(f"{target['id']}: {summary['status']}", flush=True)

        if len(results) < 2:
            raise RuntimeError('Fewer than two targets had scores. See run.json for per-target details.')
        write_ranking(destination / 'ranking/scores.csv', results)
        report.update(status='complete' if len(results) == len(targets) else 'complete_with_failed_targets',
                      ranking_file='ranking/scores.csv', ranked_targets=len(results))
    except (Exception, KeyboardInterrupt) as error:
        report.update(failed_stage=report['status'], status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(status_path, report)
    print(destination / report['ranking_file'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smiles', required=True)
    parser.add_argument('--targets', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--num-conformers', type=int, default=DEFAULT_NUM_CONFORMERS)
    parser.add_argument('--rmsd-threshold', type=float, default=DEFAULT_RMSD_THRESHOLD)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--export-distances', action='store_true', help='Export a predicted distance map for each target.')
    parser.add_argument('--distance-steps', type=int, default=25)
    args = parser.parse_args()
    if args.distance_steps < 1:
        parser.error('--distance-steps must be positive.')
    run(args)


if __name__ == '__main__':
    main()
