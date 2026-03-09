#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/lr_sweep.sh --train-data PATH [options]

Required:
  --train-data PATH

Optional:
  --val-data PATH
  --data-dtype uint16|uint32|int32|int64      (default: uint16)
  --vocab-size INT                             (default: 10000)
  --context-length INT                         (default: 256)
  --d-model INT                                (default: 512)
  --num-heads INT                              (default: 16)
  --d-ff INT                                   (default: 1344)
  --num-layers INT                             (default: 4)
  --rope-theta FLOAT                           (default: 10000)
  --batch-size INT                             (default: 64)
  --token-budget INT                           (default: 327680000)
  --learning-rates CSV                         (default: 1e-4,2e-4,3e-4,6e-4,1e-3,2e-3)
  --min-lr-ratio FLOAT                         (default: 0.1)
  --warmup-fraction FLOAT                      (default: 0.02)
  --beta1 FLOAT                                (default: 0.9)
  --beta2 FLOAT                                (default: 0.95)
  --eps FLOAT                                  (default: 1e-8)
  --weight-decay FLOAT                         (default: 0.1)
  --grad-clip FLOAT                            (default: 1.0)
  --eval-interval INT                          (default: 500)
  --eval-iters INT                             (default: 50)
  --log-interval INT                           (default: 50)
  --save-interval INT                          (default: 1000)
  --device auto|cpu|cuda                       (default: auto)
  --seed INT                                   (default: 1337)
  --save-root DIR                              (default: runs/lr_sweep)
  --conda-env NAME_OR_PATH
  --use-wandb
  --wandb-project NAME                         (default: cs336-assignment1)
  --wandb-entity NAME
  --wandb-mode online|offline|disabled         (default: online)
  -h, --help
EOF
}

TRAIN_DATA=""
VAL_DATA=""
DATA_DTYPE="uint16"
VOCAB_SIZE=10000
CONTEXT_LENGTH=256
D_MODEL=512
NUM_HEADS=16
D_FF=1344
NUM_LAYERS=4
ROPE_THETA=10000.0
BATCH_SIZE=64
TOKEN_BUDGET=327680000
LEARNING_RATES="1e-4,2e-4,3e-4,6e-4,1e-3,2e-3"
MIN_LR_RATIO=0.1
WARMUP_FRACTION=0.02
BETA1=0.9
BETA2=0.95
EPS=1e-8
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0
EVAL_INTERVAL=500
EVAL_ITERS=50
LOG_INTERVAL=50
SAVE_INTERVAL=1000
DEVICE="auto"
SEED=1337
SAVE_ROOT="runs/lr_sweep"
CONDA_ENV=""
USE_WANDB=0
WANDB_PROJECT="cs336-assignment1"
WANDB_ENTITY=""
WANDB_MODE="online"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-data) TRAIN_DATA="$2"; shift 2 ;;
    --val-data) VAL_DATA="$2"; shift 2 ;;
    --data-dtype) DATA_DTYPE="$2"; shift 2 ;;
    --vocab-size) VOCAB_SIZE="$2"; shift 2 ;;
    --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
    --d-model) D_MODEL="$2"; shift 2 ;;
    --num-heads) NUM_HEADS="$2"; shift 2 ;;
    --d-ff) D_FF="$2"; shift 2 ;;
    --num-layers) NUM_LAYERS="$2"; shift 2 ;;
    --rope-theta) ROPE_THETA="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --token-budget) TOKEN_BUDGET="$2"; shift 2 ;;
    --learning-rates) LEARNING_RATES="$2"; shift 2 ;;
    --min-lr-ratio) MIN_LR_RATIO="$2"; shift 2 ;;
    --warmup-fraction) WARMUP_FRACTION="$2"; shift 2 ;;
    --beta1) BETA1="$2"; shift 2 ;;
    --beta2) BETA2="$2"; shift 2 ;;
    --eps) EPS="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --grad-clip) GRAD_CLIP="$2"; shift 2 ;;
    --eval-interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --eval-iters) EVAL_ITERS="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --save-root) SAVE_ROOT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --use-wandb) USE_WANDB=1; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$TRAIN_DATA" ]]; then
  echo "--train-data is required" >&2
  usage
  exit 1
fi
if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "Train data not found: $TRAIN_DATA" >&2
  exit 1
fi
if [[ -n "$VAL_DATA" && ! -f "$VAL_DATA" ]]; then
  echo "Val data not found: $VAL_DATA" >&2
  exit 1
fi

if [[ -n "$CONDA_ENV" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
  fi
fi

IFS=',' read -r -a lr_values <<< "$LEARNING_RATES"
if [[ "${#lr_values[@]}" -eq 0 ]]; then
  echo "No learning rates provided." >&2
  exit 1
fi

tokens_per_step=$((BATCH_SIZE * CONTEXT_LENGTH))
steps=$(((TOKEN_BUDGET + tokens_per_step - 1) / tokens_per_step))
warmup_iters="$(awk -v s="$steps" -v f="$WARMUP_FRACTION" 'BEGIN { v=int(s*f + 0.5); if (v < 1) v=1; print v }')"

timestamp="$(date +%Y%m%d_%H%M%S)"
sweep_root="${SAVE_ROOT}/lr_${timestamp}"
mkdir -p "$sweep_root"

results_csv="${sweep_root}/results.csv"
echo "run_name,learning_rate,min_learning_rate,batch_size,steps,warmup_iters,save_dir,exit_code,start_time,end_time,duration_sec" > "$results_csv"

echo "LR sweep root: $sweep_root"
echo "Computed steps=$steps from token_budget=$TOKEN_BUDGET batch_size=$BATCH_SIZE context_length=$CONTEXT_LENGTH"

for lr in "${lr_values[@]}"; do
  safe_lr="$(echo "$lr" | tr '.' 'p' | tr '-' 'm')"
  run_name="lr_${safe_lr}_bs${BATCH_SIZE}_ctx${CONTEXT_LENGTH}"
  run_dir="${sweep_root}/${run_name}"
  mkdir -p "$run_dir"

  min_lr="$(awk -v a="$lr" -v r="$MIN_LR_RATIO" 'BEGIN { printf "%.12g", (a*r) }')"
  start_time="$(date +%Y-%m-%dT%H:%M:%S)"
  start_ts="$(date +%s)"

  echo "============================================================"
  echo "Starting run: $run_name"
  echo "  lr=$lr min_lr=$min_lr steps=$steps warmup=$warmup_iters"

  train_args=(
    python -m cs336_basics.train
    --train-data "$TRAIN_DATA"
    --data-dtype "$DATA_DTYPE"
    --vocab-size "$VOCAB_SIZE"
    --context-length "$CONTEXT_LENGTH"
    --d-model "$D_MODEL"
    --num-heads "$NUM_HEADS"
    --d-ff "$D_FF"
    --num-layers "$NUM_LAYERS"
    --rope-theta "$ROPE_THETA"
    --batch-size "$BATCH_SIZE"
    --total-iters "$steps"
    --learning-rate "$lr"
    --min-learning-rate "$min_lr"
    --warmup-iters "$warmup_iters"
    --beta1 "$BETA1"
    --beta2 "$BETA2"
    --eps "$EPS"
    --weight-decay "$WEIGHT_DECAY"
    --grad-clip "$GRAD_CLIP"
    --eval-interval "$EVAL_INTERVAL"
    --eval-iters "$EVAL_ITERS"
    --log-interval "$LOG_INTERVAL"
    --save-interval "$SAVE_INTERVAL"
    --save-dir "$run_dir"
    --device "$DEVICE"
    --seed "$SEED"
  )
  if [[ -n "$VAL_DATA" ]]; then
    train_args+=(--val-data "$VAL_DATA")
  fi
  if [[ "$USE_WANDB" -eq 1 ]]; then
    train_args+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$run_name" --wandb-mode "$WANDB_MODE")
    if [[ -n "$WANDB_ENTITY" ]]; then
      train_args+=(--wandb-entity "$WANDB_ENTITY")
    fi
  fi

  if uv run "${train_args[@]}"; then
    exit_code=0
  else
    exit_code=$?
    echo "Warning: run failed ($run_name), exit_code=$exit_code. Continuing." >&2
  fi

  end_ts="$(date +%s)"
  end_time="$(date +%Y-%m-%dT%H:%M:%S)"
  duration_sec=$((end_ts - start_ts))
  echo "${run_name},${lr},${min_lr},${BATCH_SIZE},${steps},${warmup_iters},${run_dir},${exit_code},${start_time},${end_time},${duration_sec}" >> "$results_csv"
done

echo "LR sweep complete. Summary: $results_csv"
