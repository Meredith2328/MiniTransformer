#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/batch_sweep.sh --train-data PATH [options]

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
  --batch-sizes CSV                            (default: 1,8,16,32,64,128)
  --token-budget INT                           (default: 327680000)
  --base-learning-rate FLOAT                   (default: 6e-4)
  --base-batch-size INT                        (default: 64)
  --min-lr-ratio FLOAT                         (default: 0.1)
  --warmup-fraction FLOAT                      (default: 0.02)
  --disable-linear-lr-scaling
  --lr-overrides CSV                           (example: 64=6e-4,128=8e-4)
  --beta1 FLOAT                                (default: 0.9)
  --beta2 FLOAT                                (default: 0.95)
  --eps FLOAT                                  (default: 1e-8)
  --weight-decay FLOAT                         (default: 0.1)
  --grad-clip FLOAT                            (default: 1.0)
  --eval-interval INT                          (default: 500)
  --eval-iters INT                             (default: 50)
  --log-interval INT                           (default: 50)
  --save-interval INT                          (default: 1000)
  --keep-last-checkpoints INT                  (default: 3)
  --device auto|cpu|cuda                       (default: auto)
  --seed INT                                   (default: 1337)
  --save-root DIR                              (default: runs/batch_sweep)
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
BATCH_SIZES="1,8,16,32,64,128"
TOKEN_BUDGET=327680000
BASE_LR=6e-4
BASE_BATCH_SIZE=64
MIN_LR_RATIO=0.1
WARMUP_FRACTION=0.02
DISABLE_LINEAR_LR_SCALING=0
LR_OVERRIDES=""
BETA1=0.9
BETA2=0.95
EPS=1e-8
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0
EVAL_INTERVAL=500
EVAL_ITERS=50
LOG_INTERVAL=50
SAVE_INTERVAL=1000
KEEP_LAST_CHECKPOINTS=3
DEVICE="auto"
SEED=1337
SAVE_ROOT="runs/batch_sweep"
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
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    --token-budget) TOKEN_BUDGET="$2"; shift 2 ;;
    --base-learning-rate) BASE_LR="$2"; shift 2 ;;
    --base-batch-size) BASE_BATCH_SIZE="$2"; shift 2 ;;
    --min-lr-ratio) MIN_LR_RATIO="$2"; shift 2 ;;
    --warmup-fraction) WARMUP_FRACTION="$2"; shift 2 ;;
    --disable-linear-lr-scaling) DISABLE_LINEAR_LR_SCALING=1; shift ;;
    --lr-overrides) LR_OVERRIDES="$2"; shift 2 ;;
    --beta1) BETA1="$2"; shift 2 ;;
    --beta2) BETA2="$2"; shift 2 ;;
    --eps) EPS="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --grad-clip) GRAD_CLIP="$2"; shift 2 ;;
    --eval-interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --eval-iters) EVAL_ITERS="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --keep-last-checkpoints) KEEP_LAST_CHECKPOINTS="$2"; shift 2 ;;
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

IFS=',' read -r -a batch_values <<< "$BATCH_SIZES"
if [[ "${#batch_values[@]}" -eq 0 ]]; then
  echo "No batch sizes provided." >&2
  exit 1
fi

declare -A override_map
if [[ -n "$LR_OVERRIDES" ]]; then
  IFS=',' read -r -a pairs <<< "$LR_OVERRIDES"
  for pair in "${pairs[@]}"; do
    key="${pair%%=*}"
    value="${pair#*=}"
    if [[ "$key" == "$value" ]]; then
      echo "Invalid --lr-overrides entry: $pair" >&2
      exit 1
    fi
    override_map["$key"]="$value"
  done
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
sweep_root="${SAVE_ROOT}/batch_${timestamp}"
mkdir -p "$sweep_root"
results_csv="${sweep_root}/results.csv"
echo "run_name,batch_size,learning_rate,min_learning_rate,steps,warmup_iters,save_dir,exit_code,start_time,end_time,duration_sec" > "$results_csv"

echo "Batch sweep root: $sweep_root"
echo "Token budget: $TOKEN_BUDGET"

for batch_size in "${batch_values[@]}"; do
  if [[ -n "${override_map[$batch_size]+x}" ]]; then
    lr="${override_map[$batch_size]}"
  elif [[ "$DISABLE_LINEAR_LR_SCALING" -eq 1 ]]; then
    lr="$BASE_LR"
  else
    lr="$(awk -v a="$BASE_LR" -v bs="$batch_size" -v b0="$BASE_BATCH_SIZE" 'BEGIN { printf "%.12g", (a*bs/b0) }')"
  fi

  min_lr="$(awk -v a="$lr" -v r="$MIN_LR_RATIO" 'BEGIN { printf "%.12g", (a*r) }')"
  tokens_per_step=$((batch_size * CONTEXT_LENGTH))
  steps=$(((TOKEN_BUDGET + tokens_per_step - 1) / tokens_per_step))
  warmup_iters="$(awk -v s="$steps" -v f="$WARMUP_FRACTION" 'BEGIN { v=int(s*f + 0.5); if (v < 1) v=1; print v }')"

  safe_lr="$(echo "$lr" | tr '.' 'p' | tr '-' 'm')"
  run_name="bs${batch_size}_lr${safe_lr}_ctx${CONTEXT_LENGTH}"
  run_dir="${sweep_root}/${run_name}"
  mkdir -p "$run_dir"

  start_time="$(date +%Y-%m-%dT%H:%M:%S)"
  start_ts="$(date +%s)"

  echo "============================================================"
  echo "Starting run: $run_name"
  echo "  batch_size=$batch_size lr=$lr min_lr=$min_lr steps=$steps warmup=$warmup_iters"

  wrapper_args=(
    bash "${SCRIPT_DIR}/run_tinystories_train.sh"
    --skip-bpe
    --skip-tokenize
    --train-bin "$TRAIN_DATA"
    --data-dtype "$DATA_DTYPE"
    --runs-dir "$run_dir"
    --vocab-size "$VOCAB_SIZE"
    --context-length "$CONTEXT_LENGTH"
    --d-model "$D_MODEL"
    --num-heads "$NUM_HEADS"
    --d-ff "$D_FF"
    --num-layers "$NUM_LAYERS"
    --rope-theta "$ROPE_THETA"
    --batch-size "$batch_size"
    --token-budget "$TOKEN_BUDGET"
    --learning-rate "$lr"
    --min-learning-rate "$min_lr"
    --warmup-fraction "$WARMUP_FRACTION"
    --beta1 "$BETA1"
    --beta2 "$BETA2"
    --eps "$EPS"
    --weight-decay "$WEIGHT_DECAY"
    --grad-clip "$GRAD_CLIP"
    --eval-interval "$EVAL_INTERVAL"
    --eval-iters "$EVAL_ITERS"
    --log-interval "$LOG_INTERVAL"
    --save-interval "$SAVE_INTERVAL"
    --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS"
    --device "$DEVICE"
    --seed "$SEED"
  )

  if [[ -n "$CONDA_ENV" ]]; then
    wrapper_args+=(--conda-env "$CONDA_ENV")
  fi
  if [[ -n "$VAL_DATA" ]]; then
    wrapper_args+=(--val-bin "$VAL_DATA")
  else
    wrapper_args+=(--val-bin "" --val-txt "")
  fi
  if [[ "$USE_WANDB" -eq 1 ]]; then
    wrapper_args+=(--use-wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$run_name" --wandb-mode "$WANDB_MODE")
    if [[ -n "$WANDB_ENTITY" ]]; then
      wrapper_args+=(--wandb-entity "$WANDB_ENTITY")
    fi
  fi

  if "${wrapper_args[@]}"; then
    exit_code=0
  else
    exit_code=$?
    echo "Warning: run failed ($run_name), exit_code=$exit_code. Continuing." >&2
  fi

  end_ts="$(date +%s)"
  end_time="$(date +%Y-%m-%dT%H:%M:%S)"
  duration_sec=$((end_ts - start_ts))
  echo "${run_name},${batch_size},${lr},${min_lr},${steps},${warmup_iters},${run_dir},${exit_code},${start_time},${end_time},${duration_sec}" >> "$results_csv"
done

echo "Batch sweep complete. Summary: $results_csv"