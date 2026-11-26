"""Cold-cache integration check against two previously prepared real candidates."""
import csv
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import torch

from .prepare_features import ROOT
from .target_ranking import DEFAULT_RANK_CHECKPOINT, load_for_ranking, ranked_inputs


def main():
    example = ROOT / 'evaluation/model3_target_ranking_20261118'
    selected = json.loads((example / 'structure_manifest.json').read_text(encoding='utf-8-sig'))[:2]
    with TemporaryDirectory(prefix='node_rank_cold_candidates_') as directory:
        work = Path(directory)
        manifest = work / 'two.json'
        manifest.write_text(json.dumps(selected))
        index = work / 'features.json'
        subprocess.run(['bash', 'dti.sh', 'rank-candidate-features', '--manifest', str(manifest),
                        '--output', str(index), '--cache', str(work / 'empty_cache')], cwd=ROOT, check=True)
        subprocess.run(['bash', 'dti.sh', 'rank-predict', '--features', str(index), '--device', 'cpu',
                        '--threads', '2', '--output-dir', str(work / 'results')], cwd=ROOT, check=True)
        with (work / 'results/scores.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2 and [row['rank'] for row in rows] == ['1', '2']
        assert {row['pair_id'] for row in rows} == {row['pair_id'] for row in selected}
        assert float(rows[0]['score']) > float(rows[1]['score'])
        assert set(rows[0]) == {'compound_id', 'rank', 'target_id', 'target_name', 'pair_id', 'score', 'candidate_count', 'status'}
        torch.set_num_threads(2)
        model, checkpoint = load_for_ranking(DEFAULT_RANK_CHECKPOINT)
        model.eval()
        predictions = {row['pair_id']: float(row['score']) for row in rows}
        errors = []
        with torch.inference_mode():
            for identity, sample in ranked_inputs(example / 'candidate_features.json', checkpoint):
                if identity['pair_id'] in predictions:
                    errors.append(abs(float(model(sample)['score']) - predictions[identity['pair_id']]))
        assert len(errors) == 2 and max(errors) < 1e-5, errors
        print(json.dumps({'cold_cache_parity': 'passed', 'candidates': 2, 'max_raw_score_difference': max(errors)}))


if __name__ == '__main__':
    main()
