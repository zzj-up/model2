#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${DTI_VENV:-$HOME/.venvs/dti}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Environment missing: $VENV_DIR. Run bash setup_wsl.sh first." >&2
  exit 1
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT_DIR"
case "${1:-help}" in
  status|rank-status)
    shift
    status_file="${1:-node_rank/runs/pdbbind2020_distance_diffusion/training_status.json}"
    if [[ -f "$status_file" ]]; then
      exec "$VENV_DIR/bin/python" -m json.tool "$status_file"
    fi
    echo 'No status file for the new Model 2 distance diffusion run.'
    echo "Expected: $status_file"
    ;;
  fresh-status)
    shift
    exec "$VENV_DIR/bin/python" -m fresh_dti.status "$@"
    ;;
  init|rank-init)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.cli init "$@"
    ;;
  fresh-init)
    shift
    exec "$VENV_DIR/bin/python" -m fresh_dti.initialize "$@"
    ;;
  verify|rank-verify)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.verify_model "$@"
    ;;
  rank-verify-workflow)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.verify_workflow "$@"
    ;;
  rank-features)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.prepare_features "$@"
    ;;
  conformer-library)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.conformer_library "$@"
    ;;
  rank-from-smiles|predict|rank-predict)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.rank_distance_smiles "$@"
    ;;
  legacy-rank-from-smiles)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.rank_from_smiles "$@"
    ;;
  rank-candidate-features)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.candidate_features "$@"
    ;;
  rank-audit)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.cli audit "$@"
    ;;
  legacy-rank-train)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.train "$@"
    ;;
  legacy-rank-predict|evaluate-pdbbind)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.predict "$@"
    ;;
  fresh-verify)
    shift
    exec "$VENV_DIR/bin/python" -m fresh_dti.verify "$@"
    ;;
  features-pbcnet)
    shift
    exec bash "$PROJECT_DIR/pbcnet_runtime.sh" "$PROJECT_DIR/prepare_pbcnet_features.py" "$@"
    ;;
  features-entities)
    shift
    exec "$VENV_DIR/bin/python" -m fresh_dti.prepare_entity_features "$@"
    ;;
  features-complex)
    shift
    exec bash "$PROJECT_DIR/pbcnet_runtime.sh" -m fresh_dti.prepare_complex_features "$@"
    ;;
  distance-data)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.distance_data "$@"
    ;;
  distance-train|rank-train|train-pdbbind)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.train_distance_diffusion "$@"
    ;;
  legacy-train-pdbbind)
    shift
    exec "$VENV_DIR/bin/python" -m node_rank.train_pdbbind "$@"
    ;;
  python)
    shift
    exec "$VENV_DIR/bin/python" "$@"
    ;;
  train|train-hybrid|hybrid|train-gin|gin|train-pbcnet|pbcnet)
    echo 'Legacy DTI entry points are isolated. Use rank-predict for current target ranking.' >&2
    echo 'Use bash dti.sh status or bash dti.sh verify. See README.md for the fresh model API.' >&2
    exit 2
    ;;
  help|-h|--help)
    echo 'Model 2 uses 2/4-layer geometry attention for conditional distance diffusion and compound-to-target scoring.'
    echo 'Usage: bash dti.sh {status|conformer-library|distance-data|distance-train|rank-from-smiles|python} [arguments]'
    echo 'Aliases: rank-train / train-pdbbind = distance-train; predict / rank-predict = rank-from-smiles.'
    echo 'Archived architecture tools: legacy-train-pdbbind, legacy-rank-from-smiles, legacy-rank-predict.'
    ;;
  *)
    echo "Unknown action: $1. Use bash dti.sh help." >&2
    exit 2
    ;;
esac
