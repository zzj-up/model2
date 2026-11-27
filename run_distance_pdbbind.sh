#!/usr/bin/env bash
set -euo pipefail

project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${PDBBIND_SPLITS_DIR:-$project/../inputs/pdbbind2020_retrain_20261118/dataset/splits_v2}"
cache="${DISTANCE_CACHE_DIR:-$project/inputs/pdbbind_distance_diffusion}"
run="${DISTANCE_RUN_DIR:-$project/node_rank/runs/pdbbind2020_distance_diffusion}"
workers="${DISTANCE_PREP_WORKERS:-8}"

cd "$project"
bash dti.sh python - "$source_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
metadata = json.loads((root / 'SPLIT_LOCK.json').read_text())
if metadata['schema'] != 'pdbbind_compound_group_split_v2':
    raise ValueError('Expected the locked compound-group split.')
groups = {}
for split in ('train', 'validation'):
    record = metadata['partitions'][split]
    path = root / f'{split}_manifest.json'
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record['sha256']:
        raise ValueError(f'{split} manifest failed its SHA-256 check.')
    rows = json.loads(content)
    if len(rows) != record['pairs']:
        raise ValueError(f'{split} manifest has an unexpected number of examples.')
    groups[split] = {row['compound_group'] for row in rows}
    print(f'{split}: {len(rows)} locked examples', flush=True)
if groups['train'] & groups['validation']:
    raise ValueError('Train and validation contain common compound groups.')
PY

mkdir -p "$cache"
for split in train validation; do
  bash dti.sh distance-data prepare --manifest "$source_dir/${split}_manifest.json" \
    --output "$cache/$split" --workers "$workers"
done

exec bash dti.sh distance-train --data "$cache" --split-dir "$source_dir" --run-dir "$run" "$@"
