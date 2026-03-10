#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-uv}"
CONDA_ENV=""
FORWARD_ARGS=()

activate_conda() {
  if [[ -z "$CONDA_ENV" ]]; then
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    return
  fi
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    return
  fi
  if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    return
  fi
  echo "Warning: cannot activate conda env '$CONDA_ENV' automatically." >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

activate_conda
exec "$UV_BIN" run python -m cs336_basics.resume_training "${FORWARD_ARGS[@]}"
