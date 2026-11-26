#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PBCNET_VENV:-$HOME/.venvs/pbcnet_compat}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "PBCNet2.0 environment missing: $VENV_DIR. Run bash setup_pbcnet_wsl.sh." >&2
  exit 1
fi

export DGLBACKEND=pytorch
export DGL_FFI=ctypes
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
cd "$PROJECT_DIR"
exec "$VENV_DIR/bin/python" "$@"
