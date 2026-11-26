"""Prepare unlabeled, posed candidate targets with Model 3's frozen training encoders."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

from . import pdbbind_features as frozen
from .prepare_features import ROOT, digest_json, load_records, reject_quarantine_path, residue_windows, sha256

DEFAULT_MODEL = ROOT / 'node_rank/runs/pdbbind2020_refined_20261118/model/best_validation.pt'
DEFAULT_CACHE = ROOT / 'inputs/pdbbind2020_retrain_20261118/feature_cache'


def candidate_records(path):
    originals = json.loads(path.read_text(encoding='utf-8-sig'))
    records = load_records(path)
    if any(count < 2 for count in Counter(row['compound_id'] for row in records).values()):
        raise ValueError('Provide at least two distinct target sequences for each exact compound SMILES.')
    for original, row in zip(originals, records):
        for key in ('target_id', 'ligand_sdf_sha256', 'protein_pdb_sha256'):
            if original.get(key) is not None and original[key] != row[key]:
                raise ValueError(f'{row["pair_id"]}: declared {key} differs from the input.')
        row['compound_group'] = row['compound_id']
    return records


def atom_shards(records, policy, cache):
    result = {}
    for row in records:
        identity = frozen.pbc_identity(row, policy)
        shard = frozen.cache_load(cache / 'pbc' / digest_json(identity), identity, payload=False)
        if shard:
            result[row['pair_id']] = shard
    return result


def compute_missing_atoms(records, known, policy, cache):
    repo = ROOT / 'third_party/PBCNet2.0'
    frozen.pbc_worker_init(repo, 1, policy)
    for row in records:
        if row['pair_id'] not in known:
            shard = frozen.pbc_job((row, policy, cache, False))
            if shard['status'] != 'success':
                raise ValueError(f'{row["pair_id"]}: {shard.get("error", "structural quality check failed")}')


def residue_shards(records, policy, cache, esm_model):
    import torch
    from transformers import AutoModel, AutoTokenizer
    namespace = digest_json(policy)
    sequences = {row['target_id']: row['protein_sequence'] for row in records}
    result, missing = {}, []
    for target, sequence in sequences.items():
        identity = frozen.sequence_identity(sequence, policy)
        directory = cache / 'esm_sequences' / namespace / target
        shard = frozen.cache_load(directory, identity, payload=False)
        if shard is None:
            missing.append((target, sequence, identity, directory))
        else:
            result[target] = shard
    if not missing:
        return result
    print(json.dumps({'stage': 'extract_missing_residue_features', 'targets': len(missing)}), flush=True)
    if frozen.esm_policy(esm_model, 'cuda') != policy or not torch.cuda.is_available():
        raise ValueError('Uncached targets require the frozen CUDA ESM setup used during training.')
    tokenizer = AutoTokenizer.from_pretrained(str(esm_model), local_files_only=True)
    model = AutoModel.from_pretrained(str(esm_model), local_files_only=True, add_pooling_layer=False,
                                      torch_dtype=getattr(torch, policy['dtype']), low_cpu_mem_usage=True)
    model = model.to('cuda').eval().requires_grad_(False)
    with torch.inference_mode():
        for target, sequence, identity, directory in missing:
            chunks = []
            for _, window in residue_windows(sequence):
                window_identity = frozen.sequence_identity(window, policy)
                window_dir = cache / 'esm_windows' / namespace / window_identity['sequence_sha256']
                shard = frozen.cache_load(window_dir, window_identity)
                if shard is None:
                    tokens = tokenizer(window, return_tensors='pt', truncation=False, add_special_tokens=True).to('cuda')
                    if tokens['input_ids'].shape != (1, len(window) + 2):
                        raise ValueError('ESM tokenization changed.')
                    features = model(**tokens).last_hidden_state[0, 1:-1].detach().cpu().float().clone().contiguous()
                    frozen.validate_sequence(features, window)
                    frozen.cache_save(window_dir, features, window_identity, {'residues': len(window)})
                else:
                    features = shard['payload']
                    frozen.validate_sequence(features, window)
                chunks.append(features)
            features = torch.cat(chunks)
            frozen.validate_sequence(features, sequence)
            result[target] = frozen.cache_save(directory, features, identity, {'residues': len(sequence)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--esm-model', type=Path, default=Path('/home/dti/models/esm2_3b'))
    parser.add_argument('--pbc-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    from .train_pdbbind import load_trained_score
    import torch
    torch.set_num_threads(2)
    output, manifest, cache = map(reject_quarantine_path, (args.output, args.manifest, args.cache))
    if output.exists():
        raise FileExistsError(f'Choose a new candidate index path: {output}')
    manifest_digest = sha256(manifest)
    records = candidate_records(manifest)
    _, checkpoint = load_trained_score(args.checkpoint, 'cpu')
    policy = checkpoint['feature_policy']
    atoms = atom_shards(records, policy['pbc'], cache)
    if args.pbc_only:
        compute_missing_atoms(records, atoms, policy['pbc'], cache)
        return
    if len(atoms) < len(records):
        print(json.dumps({'stage': 'extract_missing_atom_features', 'complexes': len(records) - len(atoms)}), flush=True)
        subprocess.run(['bash', str(ROOT / 'pbcnet_runtime.sh'), '-m', 'node_rank.candidate_features',
                        '--pbc-only', '--manifest', str(manifest), '--output', str(output),
                        '--cache', str(cache), '--checkpoint', str(args.checkpoint.resolve())], check=True, cwd=ROOT)
        atoms = atom_shards(records, policy['pbc'], cache)
    residues = residue_shards(records, policy['esm'], cache, args.esm_model)
    samples = []
    for row in records:
        sample = {key: row[key] for key in ('pair_id', 'compound_id', 'target_id')}
        if row.get('target_name'):
            sample['target_name'] = row['target_name']
        for name, shard in [('sample', atoms[row['pair_id']]), ('sequence', residues[row['target_id']])]:
            sample[name + '_path'] = os.path.relpath(shard['payload_path'], output.parent)
            sample[name + '_sha256'] = shard['payload_sha256']
        samples.append(sample)
    if sha256(manifest) != manifest_digest:
        raise RuntimeError('Candidate manifest changed during preprocessing.')
    frozen.write_json(output, {'schema': frozen.INDEX_SCHEMA, 'split': 'inference', 'feature_policy': policy,
                               'samples': samples, 'manifest_sha256': manifest_digest})
    print(json.dumps({'candidate_features': str(output), 'pairs': len(samples)}), flush=True)


if __name__ == '__main__':
    main()
