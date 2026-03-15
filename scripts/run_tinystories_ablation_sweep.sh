#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_tinystories_ablation_sweep.sh [sweep options] [run_tinystories_ablation options...]

Sweep options:
  --sweep-root DIR            (default: runs/tinystories_ablation_sweep)
  --run-prefix NAME           (default: ablation)
  --wandb-name-prefix NAME    (default: same as --run-prefix)
  --fail-fast
  -h, --help

Behavior:
  - Enumerates all 16 combinations of:
      RMSNorm keep/remove
      norm order pre/post
      RoPE keep/remove
      FFN SwiGLU/SiLU
  - All other arguments are forwarded to scripts/run_tinystories_ablation.sh
  - This script always overrides per-run:
      --runs-dir
      --rmsnorm
      --norm-order
      --rope
      --ffn-activation
      --wandb-run-name (when --use-wandb is forwarded)

Examples:
  bash scripts/run_tinystories_ablation_sweep.sh --skip-bpe --skip-tokenize --train-bin data/tinystories_train.bin
  bash scripts/run_tinystories_ablation_sweep.sh --use-wandb --skip-bpe --skip-tokenize --conda-env C:/Software/Miniconda/envs/cs336
EOF
}

SWEEP_ROOT="runs/tinystories_ablation_sweep"
RUN_PREFIX="ablation"
WANDB_NAME_PREFIX=""
FAIL_FAST=0
USE_WANDB=0
FORWARDED_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep-root) SWEEP_ROOT="$2"; shift 2 ;;
    --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
    --wandb-name-prefix) WANDB_NAME_PREFIX="$2"; shift 2 ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --use-wandb)
      USE_WANDB=1
      FORWARDED_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$WANDB_NAME_PREFIX" ]]; then
  WANDB_NAME_PREFIX="$RUN_PREFIX"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
sweep_root="${SWEEP_ROOT}/sweep_${timestamp}"
mkdir -p "$sweep_root"
results_csv="${sweep_root}/results.csv"
echo "run_name,rmsnorm,norm_order,rope,ffn_activation,save_dir,exit_code,start_time,end_time,duration_sec" > "$results_csv"

rmsnorm_values=(remove keep)
norm_order_values=(pre post)
rope_values=(remove keep)
ffn_values=(swiglu silu)

echo "Sweep root: $sweep_root"
echo "Enumerating 16 ablation combinations"

for rmsnorm in "${rmsnorm_values[@]}"; do
  for norm_order in "${norm_order_values[@]}"; do
    for rope in "${rope_values[@]}"; do
      for ffn in "${ffn_values[@]}"; do
        run_name="${RUN_PREFIX}_rmsnorm-${rmsnorm}_norm-${norm_order}_rope-${rope}_ffn-${ffn}"
        run_dir="${sweep_root}/${run_name}"
        mkdir -p "$run_dir"

        start_time="$(date +%Y-%m-%dT%H:%M:%S)"
        start_ts="$(date +%s)"

        echo "============================================================"
        echo "Starting run: $run_name"

        wrapper_args=(
          bash "${SCRIPT_DIR}/run_tinystories_ablation.sh"
          "${FORWARDED_ARGS[@]}"
          --runs-dir "$run_dir"
          --rmsnorm "$rmsnorm"
          --norm-order "$norm_order"
          --rope "$rope"
          --ffn-activation "$ffn"
        )

        if [[ "$USE_WANDB" -eq 1 ]]; then
          wrapper_args+=(--wandb-run-name "${WANDB_NAME_PREFIX}_${run_name}")
        fi

        if "${wrapper_args[@]}"; then
          exit_code=0
        else
          exit_code=$?
          echo "Warning: run failed ($run_name), exit_code=$exit_code" >&2
          if [[ "$FAIL_FAST" -eq 1 ]]; then
            end_ts="$(date +%s)"
            end_time="$(date +%Y-%m-%dT%H:%M:%S)"
            duration_sec=$((end_ts - start_ts))
            echo "${run_name},${rmsnorm},${norm_order},${rope},${ffn},${run_dir},${exit_code},${start_time},${end_time},${duration_sec}" >> "$results_csv"
            exit "$exit_code"
          fi
        fi

        end_ts="$(date +%s)"
        end_time="$(date +%Y-%m-%dT%H:%M:%S)"
        duration_sec=$((end_ts - start_ts))
        echo "${run_name},${rmsnorm},${norm_order},${rope},${ffn},${run_dir},${exit_code},${start_time},${end_time},${duration_sec}" >> "$results_csv"
      done
    done
  done
done

echo "Sweep complete. Summary: $results_csv"
