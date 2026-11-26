"""Dock two compound groups against a shared locked PDBbind target panel."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


ROOT = Path(__file__).resolve().parents[1]
SCORE_MODEL = ROOT / 'node_rank/runs/pdbbind2020_refined_20261118/model/best_validation.pt'
SOURCE_CACHE = ROOT.parent / 'inputs/pdbbind2020_retrain_20261118/feature_cache'


def write_json(path, data):
    stage = path.with_suffix('.json.tmp')
    stage.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    stage.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def dock_one(target, queries, libraries, output, seed, exhaustiveness, max_evals, check_at_pose=False):
    from .rigid_placement import place_target
    from .screen_receptor import prepare_receptor

    target_id = target['target_id']
    attempts = []
    with TemporaryDirectory(prefix='model2_panel_') as temporary:
        receptor = Path(temporary) / 'receptor.pdbqt'
        for structure in target['structures']:
            protein = Path(structure['protein_pdb'])
            box = {'center': structure['box_center'], 'size': structure['box_size']}
            try:
                preparation = prepare_receptor(protein, receptor, check_at_pose=check_at_pose, **box)
            except (ValueError, RuntimeError, OSError, KeyError, AssertionError) as error:
                attempts.append({'pdb_id': structure['pdb_id'], 'error': str(error)})
            else:
                break
        else:
            return {'target_id': target_id, 'status': 'receptor_failed', 'attempts': attempts}
        omitted_coordinates = preparation.pop('omitted_heavy_coordinates', [])
        result = {'target_id': target_id, 'status': 'prepared', 'structure': structure,
                  'preparation': {k: v for k, v in preparation.items() if k != 'receptor_pdbqt'},
                  'attempts': attempts, 'queries': {}, 'checked_omissions_at_pose': check_at_pose}
        for query in queries:
            split = query['split']
            pose_root = 'recovery_poses' if check_at_pose else 'poses'
            pose = output / pose_root / split / f'{target_id}.sdf'
            try:
                placement = place_target(libraries[split], protein, pose, receptor_pdbqt=receptor,
                                         seed=seed, exhaustiveness=exhaustiveness, cpu=1,
                                         max_evals=max_evals, **box)
                if omitted_coordinates:
                    from rdkit import Chem
                    from scipy.spatial import cKDTree

                    ligand = Chem.MolFromMolFile(str(pose))
                    if ligand is None or not ligand.GetNumHeavyAtoms():
                        raise ValueError('Could not validate pose against the omitted heavy atoms.')
                    distances = cKDTree(omitted_coordinates).query(ligand.GetConformer().GetPositions())[0]
                    nearest = float(distances.min())
                    if nearest <= 8:
                        raise ValueError(f'Omitted receptor heavy atoms are {nearest:.2f} Å from the pose (8 Å required).')
                    placement['closest_omitted_heavy_atom_angstrom'] = nearest
            except (ValueError, RuntimeError, OSError, KeyError, AssertionError) as error:
                result['queries'][split] = {'status': 'placement_failed', 'error': str(error)}
            else:
                result['queries'][split] = {'status': 'pose_available', **placement}
        return result


def dock(args):
    from .conformer_library import build_library

    output = args.output.resolve()
    panel_path = output / 'panel.json'
    panel = read_json(panel_path)
    queries, targets = panel['queries'], panel['targets']
    from .prepare_features import sha256
    config = {'seed': args.seed, 'exhaustiveness': args.exhaustiveness, 'num_conformers': args.num_conformers,
              'max_evaluations_per_conformer': args.max_evals, 'panel_sha256': sha256(panel_path)}
    config_path = output / 'dock_config.json'
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError('To change docking settings, use a fresh output directory.')
    write_json(config_path, config)
    libraries = {}
    for query in queries:
        split = query['split']
        library = output / 'conformers' / split
        if not library.exists():
            build_library(query['smiles'], library, seed=args.seed, num_conformers=args.num_conformers)
        libraries[split] = library / 'representatives.sdf'
        (output / 'poses' / split).mkdir(parents=True, exist_ok=True)
        state = read_json(library / 'conformer_manifest.json')
        print(f'{split}: {state["representative_count"]} conformer clusters', flush=True)
    records = output / 'docking_records'
    records.mkdir(exist_ok=True)
    remaining = [target for target in targets if not (records / f'{target["target_id"]}.json').exists()]
    known = {target_id for query in queries for target_id in query['known_target_ids']}
    remaining.sort(key=lambda target: target['target_id'] not in known)
    if args.limit:
        remaining = remaining[:args.limit]
    total = len(targets)
    completed = total - sum(not (records / f'{target["target_id"]}.json').exists() for target in targets)
    print(f'Docking {len(remaining)} targets, {completed}/{total} already attempted.', flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(dock_one, target, queries, libraries, output, args.seed,
                                   args.exhaustiveness, args.max_evals): target
                   for target in remaining}
        for future in as_completed(futures):
            result = future.result()
            write_json(records / f'{result["target_id"]}.json', result)
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f'Docking attempts complete: {completed}/{total}', flush=True)


def recover(args):
    output = args.output.resolve()
    panel = read_json(output / 'panel.json')
    config = read_json(output / 'dock_config.json')
    queries = panel['queries']
    libraries = {query['split']: output / 'conformers' / query['split'] / 'representatives.sdf' for query in queries}
    targets = {target['target_id']: target for target in panel['targets']}
    out = output / 'recovery_records'
    out.mkdir(exist_ok=True)
    for query in queries:
        (output / 'recovery_poses' / query['split']).mkdir(parents=True, exist_ok=True)
    seen = set()
    finished = len(list(out.glob('*.json')))
    rescued = 0
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        while True:
            paths = {path.stem: path for path in (output / 'docking_records').glob('*.json')}
            eligible = []
            for target_id in sorted(paths.keys() - seen):
                result = read_json(paths[target_id])
                if result['status'] == 'receptor_failed' and not (out / f'{target_id}.json').is_file():
                    errors = [item['error'] for item in result['attempts']]
                    if any(error.startswith(('Meeko omitted receptor heavy atoms near', 'No installed Meeko template for residues near'))
                           for error in errors):
                        eligible.append(targets[target_id])
            seen.update(paths)
            batches = [executor.submit(dock_one, target, queries, libraries, output, config['seed'],
                                       config['exhaustiveness'], config['max_evaluations_per_conformer'], True)
                       for target in eligible]
            for future in as_completed(batches):
                result = future.result()
                write_json(out / f'{result["target_id"]}.json', result)
                finished += 1
                rescued += any(pose['status'] == 'pose_available' for pose in result.get('queries', {}).values())
                if finished % 25 == 0:
                    print(f'Omitted-atom checks complete: {finished}, with at least one valid pose: {rescued}', flush=True)
            if len(seen) == len(targets) or not args.follow:
                break
            time.sleep(5)
    print(f'Omitted-atom checks complete: {finished}, with at least one valid pose in this run: {rescued}', flush=True)


def run_recovery(args):
    state = args.output.resolve() / 'recovery_state.json'
    write_json(state, {'status': 'running'})
    try:
        recover(args)
    except BaseException as error:
        write_json(state, {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    write_json(state, {'status': 'complete'})


def docking_record(output, target_id):
    strict = output / 'docking_records' / f'{target_id}.json'
    if not strict.is_file():
        return {}
    result = read_json(strict)
    fallback = output / 'recovery_records' / strict.name
    if not fallback.is_file():
        return result
    recovery = read_json(fallback)
    if recovery['status'] != 'prepared':
        return result
    return recovery


def posed_manifest(panel, output, split):
    from .conformer_library import MANIFEST_FILENAME

    smiles = read_json(output / 'conformers' / split / MANIFEST_FILENAME)['canonical_smiles']
    for target in panel['targets']:
        target_id = target['target_id']
        result = docking_record(output, target_id)
        pose = result.get('queries', {}).get(split, {})
        if pose.get('status') != 'pose_available':
            continue
        structure = result['structure']
        yield {'pair_id': f'{split}_{target_id}', 'target_name': target['target_name'],
               'target_id': target_id, 'protein_sequence': target['protein_sequence'],
               'protein_chain': structure['protein_chain'], 'protein_pdb': structure['protein_pdb'],
               'ligand_sdf': pose['pose_sdf'], 'smiles': smiles}


def pbc(args):
    from .candidate_features import candidate_records
    from .pdbbind_features import pbc_job, pbc_worker_init
    from .train_pdbbind import load_trained_score
    import torch

    torch.set_num_threads(1)
    policy = load_trained_score(SCORE_MODEL, 'cpu')[1]['feature_policy']['pbc']
    records = [row for manifest in args.manifest for row in candidate_records(manifest)]
    cache = args.output / 'pbc_cache'
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=pbc_worker_init,
                             initargs=(ROOT / 'third_party/PBCNet2.0', 1, policy)) as executor:
        tasks = [executor.submit(pbc_job, (row, policy, cache, False)) for row in records]
        for number, future in enumerate(as_completed(tasks), 1):
            future.result()
            if number % 50 == 0 or number == len(tasks):
                print(f'PBC feature attempts complete: {number}/{len(tasks)}', flush=True)


def score(args):
    from .candidate_features import atom_shards, candidate_records, residue_shards
    from .pdbbind_features import INDEX_SCHEMA, pbc_identity
    from .prepare_features import digest_json, sha256
    from .train_pdbbind import load_trained_score
    import torch

    torch.set_num_threads(4)
    output = args.output.resolve()
    panel = read_json(output / 'panel.json')
    recovery_state = output / 'recovery_state.json'
    while recovery_state.exists() and read_json(recovery_state)['status'] == 'running':
        time.sleep(5)
    manifests = {}
    for query in panel['queries']:
        split = query['split']
        path = output / f'{split}_poses.json'
        write_json(path, list(posed_manifest(panel, output, split)))
        if len(read_json(path)) >= 2:
            manifests[split] = path
    if not manifests:
        raise ValueError('No query has enough placed poses to score a ranking.')
    subprocess.run(['bash', str(ROOT / 'pbcnet_runtime.sh'), '-m', 'node_rank.screen_library', 'pbc',
                    '--output', str(output), '--workers', str(args.workers), '--manifest',
                    *map(str, manifests.values())], cwd=ROOT, check=True)
    checkpoint = load_trained_score(SCORE_MODEL, 'cpu')[1]
    policy = checkpoint['feature_policy']
    cache = output / 'pbc_cache'
    matching = {}
    for split, manifest in manifests.items():
        records = candidate_records(manifest)
        atoms = atom_shards(records, policy['pbc'], cache)
        usable = [row for row in records if row['pair_id'] in atoms]
        failures = {}
        for row in records:
            if row['pair_id'] not in atoms:
                key = digest_json(pbc_identity(row, policy['pbc']))
                report = cache / 'pbc_failures' / f'{key}.json'
                failures[row['target_id']] = read_json(report)['failure']['error'] if report.exists() else 'Missing PBC features'
        matching[split] = {'usable': usable, 'atoms': atoms}
        write_json(output / f'{split}_feature_failures.json', failures)
    sequence_rows = [row for item in matching.values() for row in item['usable']]
    residues = residue_shards(sequence_rows, policy['esm'], args.sequence_cache, args.esm_model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rank_paths = {}
    for split, item in matching.items():
        samples = []
        for row in item['usable']:
            atom = item['atoms'][row['pair_id']]
            sequence = residues[row['target_id']]
            sample = {key: row[key] for key in ('pair_id', 'compound_id', 'target_id', 'target_name')}
            for kind, shard in [('sample', atom), ('sequence', sequence)]:
                sample[f'{kind}_path'] = str(shard['payload_path'])
                sample[f'{kind}_sha256'] = shard['payload_sha256']
            samples.append(sample)
        if len(samples) < 2:
            continue
        index = output / f'{split}_features.json'
        write_json(index, {'schema': INDEX_SCHEMA, 'split': 'inference', 'feature_policy': policy,
                           'samples': samples, 'manifest_sha256': sha256(manifests[split])})
        feature_digest = sha256(index)
        destination = output / f'{split}_ranking_{feature_digest[:12]}'
        if not destination.exists():
            with TemporaryDirectory(prefix='.rank_', dir=output) as temporary:
                staged = Path(temporary) / 'ranking'
                subprocess.run([sys.executable, '-m', 'node_rank.predict', '--features', str(index),
                                '--checkpoint', str(SCORE_MODEL), '--output-dir', str(staged),
                                '--device', args.device], cwd=ROOT, check=True)
                staged.rename(destination)
        metadata = read_json(destination / 'summary.json')
        ranks = destination / 'scores.csv'
        with ranks.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if (metadata['features_sha256'] != feature_digest or metadata['checkpoint_sha256'] != sha256(SCORE_MODEL)
                or len(rows) != len(samples) or {row['target_id'] for row in rows} != {row['target_id'] for row in samples}):
            raise ValueError(f'Ranking output does not match all input features: {destination}')
        rank_paths[split] = ranks
    summarize(panel, output, rank_paths)


def summarize(panel, output, rank_paths):
    targets = panel['targets']
    records = {path.stem: docking_record(output, path.stem) for path in (output / 'docking_records').glob('*.json')}
    summary = {'requested_targets': len(targets), 'attempted_docking_targets': len(records), 'queries': []}
    for query in panel['queries']:
        split = query['split']
        known = set(query['known_target_ids'])
        ranks = rank_paths.get(split)
        ranked = {}
        if ranks is not None:
            with ranks.open(newline='', encoding='utf-8') as stream:
                ranked = {row['target_id']: row for row in csv.DictReader(stream)}
        failures_path = output / f'{split}_feature_failures.json'
        failures = read_json(failures_path) if failures_path.exists() else {}
        rows = []
        for target in targets:
            target_id = target['target_id']
            result = records.get(target_id, {})
            pose = result.get('queries', {}).get(split, {})
            rank = ranked.get(target_id, {})
            status = 'ranked' if rank else 'feature_failed' if target_id in failures else pose.get('status', result.get('status', 'not_attempted'))
            error = failures.get(target_id) or pose.get('error') or (' | '.join(item['error'] for item in result.get('attempts', [])) if status == 'receptor_failed' else '')
            rows.append({'status': status, 'rank': rank.get('rank', ''), 'score': rank.get('score', ''),
                         'target_id': target_id, 'target_name': target['target_name'],
                         'accession': target['accession'], 'structure_pdb': result.get('structure', {}).get('pdb_id', ''),
                         'omission_check_at_pose': result.get('checked_omissions_at_pose', False),
                         'known_target': target_id in known, 'error': error})
        rows.sort(key=lambda row: (not row['rank'], int(row['rank']) if row['rank'] else 0, row['target_id']))
        with (output / f'{split}_all_targets.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        statuses = {status: sum(row['status'] == status for row in rows) for status in sorted({row['status'] for row in rows})}
        summary['queries'].append({**query, 'statuses': statuses,
                                   'known_results': [row for row in rows if row['known_target']],
                                   'top_ten': [row for row in rows if row['rank']][:10],
                                   'results': f'{split}_all_targets.csv'})
    write_json(output / 'screening_summary.json', summary)


def predict_verified(args):
    """Run the standard pair scorer in parallel and rank once over all scores."""
    from .io import rank_rows
    from .prepare_features import sha256

    output = args.output.resolve()
    panel = read_json(output / 'panel.json')
    checkpoint_digest = sha256(SCORE_MODEL)
    batches = output / 'prediction_batches'
    batches.mkdir(exist_ok=True)
    tasks, groups = [], {}
    for split in [query['split'] for query in panel['queries']]:
        index = output / f'{split}_verified_features.json'
        features = read_json(index)
        groups[split] = (index, features['samples'], [])
        length = len(features['samples'])
        chunks = min(args.workers, length // 2)
        for chunk in range(chunks):
            subset = features['samples'][chunk * length // chunks : (chunk + 1) * length // chunks]
            path = batches / f'{split}_{chunk:02d}.json'
            payload = {**features, 'samples': subset}
            if path.exists() and read_json(path) != payload:
                raise ValueError(f'Existing inference batch has different inputs: {path}')
            write_json(path, payload)
            tasks.append((split, path, subset))

    def predict_batch(task):
        split, path, subset = task
        ranking = path.with_suffix('')
        if not ranking.exists():
            with TemporaryDirectory(prefix='.predict_', dir=batches) as temporary, path.with_suffix('.log').open('w') as log:
                staged = Path(temporary) / 'ranking'
                subprocess.run([sys.executable, '-m', 'node_rank.predict', '--features', str(path),
                                '--checkpoint', str(SCORE_MODEL), '--output-dir', str(staged),
                                '--device', args.device, '--threads', '1'], cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
                staged.rename(ranking)
        metadata = read_json(ranking / 'summary.json')
        with (ranking / 'scores.csv').open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        keys = ('pair_id', 'compound_id', 'target_id')
        if (metadata['checkpoint_sha256'] != checkpoint_digest or metadata['features_sha256'] != sha256(path)
                or len(rows) != len(subset) or {tuple(row[key] for key in keys) for row in rows}
                != {tuple(row[key] for key in keys) for row in subset}):
            raise ValueError(f'Batch scores do not match their declared feature rows: {ranking}')
        return split, [{**row, 'score': float(row['score'])} for row in rows], metadata

    with ThreadPoolExecutor(args.workers) as executor:
        futures = [executor.submit(predict_batch, task) for task in tasks]
        for number, future in enumerate(as_completed(futures), 1):
            split, rows, metadata = future.result()
            groups[split][2].append((rows, metadata))
            print(f'Inference batches complete: {number}/{len(tasks)}', flush=True)
    rank_paths = {}
    for split, (index, samples, results) in groups.items():
        rows = rank_rows([row for batch_rows, _ in results for row in batch_rows])
        if len(rows) != len(samples):
            raise ValueError(f'Incomplete combined ranking for {split}.')
        destination = output / f'{split}_ranking'
        destination.mkdir(exist_ok=False)
        rank_paths[split] = destination / 'scores.csv'
        with rank_paths[split].open('x', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(destination / 'summary.json', {**results[0][1], 'pairs': len(rows),
                   'features_sha256': sha256(index), 'inference_batches': len(results)})
    summarize(panel, output, rank_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, handler in [('dock', dock), ('recover', run_recovery), ('pbc', pbc), ('score', score),
                             ('predict-verified', predict_verified)]:
        stage = commands.add_parser(command)
        stage.set_defaults(handler=handler)
        stage.add_argument('--output', type=Path, required=True)
        stage.add_argument('--workers', type=int, default=6)
        if command == 'dock':
            stage.add_argument('--seed', type=int, default=20261125)
            stage.add_argument('--num-conformers', type=int, default=32)
            stage.add_argument('--exhaustiveness', type=int, default=2)
            stage.add_argument('--max-evals', type=int, default=0)
            stage.add_argument('--limit', type=int)
        elif command == 'recover':
            stage.add_argument('--follow', action='store_true')
        elif command == 'pbc':
            stage.add_argument('--manifest', nargs='+', type=Path, required=True)
        elif command == 'score':
            stage.add_argument('--sequence-cache', type=Path, default=SOURCE_CACHE)
            stage.add_argument('--esm-model', type=Path, default=Path('/home/dti/models/esm2_3b'))
            stage.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        else:
            stage.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    args.handler(args)


if __name__ == '__main__':
    main()
