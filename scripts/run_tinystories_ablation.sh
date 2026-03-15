#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-uv}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_tinystories_ablation.sh [options]

Core options:
  --conda-env NAME_OR_PATH
  --data-dir DIR                         (default: data)
  --tokenizer-dir DIR                    (default: tokenizer)
  --runs-dir DIR                         (default: runs/tinystories_ablation)
  --train-txt FILE                       (default: TinyStoriesV2-GPT4-train.txt)
  --val-txt FILE                         (default: TinyStoriesV2-GPT4-valid.txt; empty disables validation)
  --train-bin FILE                       (default: data/tinystories_train.bin)
  --val-bin FILE                         (default: data/tinystories_val.bin; empty disables validation)
  --vocab-size INT                       (default: 10000)
  --special-tokens CSV                   (default: <|endoftext|>)
  --data-dtype uint16|uint32|int32|int64 (default: uint16)

Model and training options:
  --context-length INT                   (default: 256)
  --d-model INT                          (default: 512)
  --num-heads INT                        (default: 16)
  --d-ff INT                             (default: 1344)
  --num-layers INT                       (default: 4)
  --rope-theta FLOAT                     (default: 10000)
  --batch-size INT                       (default: 64)
  --token-budget INT                     (default: 327680000)
  --learning-rate FLOAT                  (default: 6e-4)
  --min-learning-rate FLOAT              (default: 6e-5)
  --warmup-fraction FLOAT                (default: 0.02)
  --beta1 FLOAT                          (default: 0.9)
  --beta2 FLOAT                          (default: 0.95)
  --eps FLOAT                            (default: 1e-8)
  --weight-decay FLOAT                   (default: 0.1)
  --grad-clip FLOAT                      (default: 1.0)
  --optimizer custom_adamw|torch_adamw   (default: custom_adamw)
  --eval-interval INT                    (default: 500)
  --eval-iters INT                       (default: 50)
  --log-interval INT                     (default: 50)
  --save-interval INT                    (default: 1000)
  --keep-last-checkpoints INT            (default: 3)
  --device auto|cpu|cuda                 (default: auto)
  --seed INT                             (default: 1337)

Ablation switches:
  --keep-rmsnorm                         (default: off; removes RMSNorm)
  --remove-rmsnorm
  --post-norm                            (default: pre-norm)
  --pre-norm
  --keep-rope                            (default: off; removes RoPE)
  --remove-rope
  --use-silu                             (default: SwiGLU)
  --use-swiglu

Raw ablation values:
  --rmsnorm keep|remove
  --norm-order pre|post
  --rope keep|remove
  --ffn-activation swiglu|silu

W&B:
  --use-wandb
  --wandb-project NAME                   (default: cs336-assignment1)
  --wandb-entity NAME                    (default: empty)
  --wandb-run-name NAME                  (default: auto)
  --wandb-mode online|offline|disabled   (default: online)

Pipeline control:
  --skip-bpe
  --skip-tokenize
  --bpe-progress-every INT               (default: 100)
  --bpe-heartbeat-seconds INT            (default: 15)
  --tokenize-progress-every-lines INT    (default: 10000)
  -h, --help
EOF
}

resolve_path_arg() {
  local base_dir="$1"
  local raw_path="$2"

  if [[ -z "$raw_path" ]]; then
    printf ''
    return
  fi
  if [[ "$raw_path" = /* || "$raw_path" == ./* || "$raw_path" == ../* || "$raw_path" == *"/"* ]]; then
    printf '%s' "$raw_path"
    return
  fi
  printf '%s/%s' "$base_dir" "$raw_path"
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ -z "$path" || ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

activate_conda() {
  if [[ -z "$CONDA_ENV" ]]; then
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if conda activate "$CONDA_ENV"; then
      return
    fi
    echo "Warning: failed to activate conda env '$CONDA_ENV' via current conda." >&2
  fi
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    if conda activate "$CONDA_ENV"; then
      return
    fi
    echo "Warning: failed to activate conda env '$CONDA_ENV' via \$HOME/miniconda3." >&2
  fi
  if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    if conda activate "$CONDA_ENV"; then
      return
    fi
    echo "Warning: failed to activate conda env '$CONDA_ENV' via \$HOME/anaconda3." >&2
  fi
  echo "Warning: cannot activate conda env '$CONDA_ENV' automatically." >&2
}

normalize_path_for_bash() {
  local raw_path="$1"

  if [[ -z "$raw_path" ]]; then
    printf ''
    return
  fi
  if [[ -e "$raw_path" ]]; then
    printf '%s' "$raw_path"
    return
  fi
  if command -v cygpath >/dev/null 2>&1; then
    local converted
    converted="$(cygpath -u "$raw_path" 2>/dev/null || true)"
    if [[ -n "$converted" ]]; then
      printf '%s' "$converted"
      return
    fi
  fi
  if [[ "$raw_path" =~ ^([A-Za-z]):[\\/](.*)$ ]]; then
    local drive_letter="${BASH_REMATCH[1],,}"
    local rest_path="${BASH_REMATCH[2]}"
    rest_path="${rest_path//\\//}"
    if [[ -d "/mnt/${drive_letter}" ]]; then
      printf '/mnt/%s/%s' "$drive_letter" "$rest_path"
      return
    fi
    printf '/%s/%s' "$drive_letter" "$rest_path"
    return
  fi
  printf '%s' "$raw_path"
}

resolve_uv_bin() {
  if [[ "$UV_BIN" == *"/"* || "$UV_BIN" == *"\\"* ]]; then
    local normalized
    normalized="$(normalize_path_for_bash "$UV_BIN")"
    if [[ -f "$normalized" ]]; then
      UV_BIN="$normalized"
      return
    fi
  elif command -v "$UV_BIN" >/dev/null 2>&1; then
    return
  fi

  local candidate normalized_candidate
  for candidate in "${CONDA_PREFIX:-}" "${CONDA_ENV}" "${VIRTUAL_ENV:-}"; do
    if [[ -z "$candidate" ]]; then
      continue
    fi
    normalized_candidate="$(normalize_path_for_bash "${candidate}/Scripts/uv.exe")"
    if [[ -f "$normalized_candidate" ]]; then
      UV_BIN="$normalized_candidate"
      return
    fi
  done

  echo "Could not find uv. Set UV_BIN explicitly or activate an environment that provides uv." >&2
  exit 1
}

validate_choice() {
  local label="$1"
  local value="$2"
  shift 2

  for candidate in "$@"; do
    if [[ "$value" == "$candidate" ]]; then
      return
    fi
  done

  echo "Invalid ${label}: ${value}" >&2
  exit 1
}

CONDA_ENV=""
DATA_DIR="data"
TOKENIZER_DIR="tokenizer"
RUNS_DIR="runs/tinystories_ablation"

TRAIN_TXT="TinyStoriesV2-GPT4-train.txt"
VAL_TXT="TinyStoriesV2-GPT4-valid.txt"
TRAIN_BIN="tinystories_train.bin"
VAL_BIN="tinystories_val.bin"
TRAIN_BIN_SET=0
VAL_BIN_SET=0
VAL_TXT_SET=0
VOCAB_SIZE=10000
SPECIAL_TOKENS="<|endoftext|>"
DATA_DTYPE="uint16"

CONTEXT_LENGTH=256
D_MODEL=512
NUM_HEADS=16
D_FF=1344
NUM_LAYERS=4
ROPE_THETA=10000.0

BATCH_SIZE=64
TOKEN_BUDGET=327680000
LEARNING_RATE=0.002
MIN_LEARNING_RATE=0.0002
WARMUP_FRACTION=0.02
BETA1=0.9
BETA2=0.95
EPS=1e-8
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0
OPTIMIZER="custom_adamw"

EVAL_INTERVAL=500
EVAL_ITERS=50
LOG_INTERVAL=50
SAVE_INTERVAL=1000
KEEP_LAST_CHECKPOINTS=3
DEVICE="auto"
SEED=1337

RMSNORM="remove"
NORM_ORDER="pre"
ROPE="remove"
FFN_ACTIVATION="swiglu"

USE_WANDB=0
WANDB_PROJECT="cs336-assignment1"
WANDB_ENTITY=""
WANDB_RUN_NAME=""
WANDB_MODE="online"

SKIP_BPE=0
SKIP_TOKENIZE=0
BPE_PROGRESS_EVERY=100
BPE_HEARTBEAT_SECONDS=15
TOKENIZE_PROGRESS_EVERY_LINES=10000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --tokenizer-dir) TOKENIZER_DIR="$2"; shift 2 ;;
    --runs-dir) RUNS_DIR="$2"; shift 2 ;;
    --train-txt) TRAIN_TXT="$2"; shift 2 ;;
    --val-txt) VAL_TXT="$2"; VAL_TXT_SET=1; shift 2 ;;
    --train-bin) TRAIN_BIN="$2"; TRAIN_BIN_SET=1; shift 2 ;;
    --val-bin) VAL_BIN="$2"; VAL_BIN_SET=1; shift 2 ;;
    --vocab-size) VOCAB_SIZE="$2"; shift 2 ;;
    --special-tokens) SPECIAL_TOKENS="$2"; shift 2 ;;
    --data-dtype) DATA_DTYPE="$2"; shift 2 ;;
    --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
    --d-model) D_MODEL="$2"; shift 2 ;;
    --num-heads) NUM_HEADS="$2"; shift 2 ;;
    --d-ff) D_FF="$2"; shift 2 ;;
    --num-layers) NUM_LAYERS="$2"; shift 2 ;;
    --rope-theta) ROPE_THETA="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --token-budget) TOKEN_BUDGET="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --min-learning-rate) MIN_LEARNING_RATE="$2"; shift 2 ;;
    --warmup-fraction) WARMUP_FRACTION="$2"; shift 2 ;;
    --beta1) BETA1="$2"; shift 2 ;;
    --beta2) BETA2="$2"; shift 2 ;;
    --eps) EPS="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --grad-clip) GRAD_CLIP="$2"; shift 2 ;;
    --optimizer) OPTIMIZER="$2"; shift 2 ;;
    --eval-interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --eval-iters) EVAL_ITERS="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --keep-last-checkpoints) KEEP_LAST_CHECKPOINTS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --keep-rmsnorm) RMSNORM="keep"; shift ;;
    --remove-rmsnorm) RMSNORM="remove"; shift ;;
    --rmsnorm) RMSNORM="$2"; shift 2 ;;
    --post-norm) NORM_ORDER="post"; shift ;;
    --pre-norm) NORM_ORDER="pre"; shift ;;
    --norm-order) NORM_ORDER="$2"; shift 2 ;;
    --keep-rope) ROPE="keep"; shift ;;
    --remove-rope) ROPE="remove"; shift ;;
    --rope) ROPE="$2"; shift 2 ;;
    --use-silu) FFN_ACTIVATION="silu"; shift ;;
    --use-swiglu) FFN_ACTIVATION="swiglu"; shift ;;
    --ffn-activation) FFN_ACTIVATION="$2"; shift 2 ;;
    --use-wandb) USE_WANDB=1; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-run-name) WANDB_RUN_NAME="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
    --skip-bpe) SKIP_BPE=1; shift ;;
    --skip-tokenize) SKIP_TOKENIZE=1; shift ;;
    --bpe-progress-every) BPE_PROGRESS_EVERY="$2"; shift 2 ;;
    --bpe-heartbeat-seconds) BPE_HEARTBEAT_SECONDS="$2"; shift 2 ;;
    --tokenize-progress-every-lines) TOKENIZE_PROGRESS_EVERY_LINES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

validate_choice "rmsnorm" "$RMSNORM" keep remove
validate_choice "norm-order" "$NORM_ORDER" pre post
validate_choice "rope" "$ROPE" keep remove
validate_choice "ffn-activation" "$FFN_ACTIVATION" swiglu silu

activate_conda
resolve_uv_bin

TRAIN_TXT_PATH="$(resolve_path_arg "$DATA_DIR" "$TRAIN_TXT")"
VAL_TXT_PATH="$(resolve_path_arg "$DATA_DIR" "$VAL_TXT")"
TRAIN_BIN="$(resolve_path_arg "$DATA_DIR" "$TRAIN_BIN")"
VAL_BIN="$(resolve_path_arg "$DATA_DIR" "$VAL_BIN")"

VOCAB_PKL="${TOKENIZER_DIR}/tinystories_bpe_vocab.pkl"
MERGES_PKL="${TOKENIZER_DIR}/tinystories_bpe_merges.pkl"
TRAIN_META="${TRAIN_BIN}.meta.json"
VAL_META="${VAL_BIN}.meta.json"

USE_VAL=1
if [[ "$VAL_BIN_SET" -eq 1 && -z "$VAL_BIN" ]]; then
  USE_VAL=0
fi
if [[ "$VAL_TXT_SET" -eq 1 && -z "$VAL_TXT_PATH" ]]; then
  USE_VAL=0
fi

mkdir -p "$TOKENIZER_DIR" "$RUNS_DIR"
mkdir -p "$(dirname "$TRAIN_BIN")"
if [[ "$USE_VAL" -eq 1 && -n "$VAL_BIN" ]]; then
  mkdir -p "$(dirname "$VAL_BIN")"
fi
if [[ -n "$DATA_DIR" ]]; then
  mkdir -p "$DATA_DIR"
fi

if [[ "$SKIP_BPE" -eq 0 ]]; then
  if [[ -z "$TRAIN_TXT_PATH" ]]; then
    echo "BPE training requires --train-txt." >&2
    exit 1
  fi
  require_file "train txt" "$TRAIN_TXT_PATH"

  echo "============================================================"
  echo "Step 1/3: train BPE tokenizer"
  "$UV_BIN" run python -m cs336_basics.train_bpe \
    --input-path "$TRAIN_TXT_PATH" \
    --vocab-size "$VOCAB_SIZE" \
    --special-tokens "$SPECIAL_TOKENS" \
    --progress-every "$BPE_PROGRESS_EVERY" \
    --heartbeat-seconds "$BPE_HEARTBEAT_SECONDS" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --vocab-out "$VOCAB_PKL" \
    --merges-out "$MERGES_PKL"
fi

if [[ "$SKIP_TOKENIZE" -eq 0 ]]; then
  if [[ -z "$TRAIN_TXT_PATH" ]]; then
    echo "Tokenization requires --train-txt." >&2
    exit 1
  fi
  require_file "train txt" "$TRAIN_TXT_PATH"
  require_file "tokenizer vocab" "$VOCAB_PKL"
  require_file "tokenizer merges" "$MERGES_PKL"

  echo "============================================================"
  echo "Step 2/3: tokenize txt -> bin"
  "$UV_BIN" run python scripts/tokenize_to_bin.py \
    --input-text "$TRAIN_TXT_PATH" \
    --vocab-pkl "$VOCAB_PKL" \
    --merges-pkl "$MERGES_PKL" \
    --output-bin "$TRAIN_BIN" \
    --output-meta "$TRAIN_META" \
    --special-tokens "$SPECIAL_TOKENS" \
    --dtype "$DATA_DTYPE" \
    --progress-every-lines "$TOKENIZE_PROGRESS_EVERY_LINES"

  if [[ "$USE_VAL" -eq 1 ]]; then
    if [[ -z "$VAL_TXT_PATH" ]]; then
      echo "Validation tokenization requires --val-txt, or disable validation with --val-bin ''." >&2
      exit 1
    fi
    require_file "val txt" "$VAL_TXT_PATH"

    "$UV_BIN" run python scripts/tokenize_to_bin.py \
      --input-text "$VAL_TXT_PATH" \
      --vocab-pkl "$VOCAB_PKL" \
      --merges-pkl "$MERGES_PKL" \
      --output-bin "$VAL_BIN" \
      --output-meta "$VAL_META" \
      --special-tokens "$SPECIAL_TOKENS" \
      --dtype "$DATA_DTYPE" \
      --progress-every-lines "$TOKENIZE_PROGRESS_EVERY_LINES"
  fi
else
  require_file "train bin" "$TRAIN_BIN"
  if [[ "$USE_VAL" -eq 1 ]]; then
    if [[ -f "$VAL_BIN" ]]; then
      :
    elif [[ "$VAL_BIN_SET" -eq 1 ]]; then
      echo "Missing val bin: $VAL_BIN" >&2
      exit 1
    else
      echo "Validation bin not found at $VAL_BIN. Continuing without validation."
      USE_VAL=0
      VAL_BIN=""
    fi
  fi
fi

require_file "train bin" "$TRAIN_BIN"
if [[ "$USE_VAL" -eq 1 ]]; then
  require_file "val bin" "$VAL_BIN"
fi

tokens_per_step=$((BATCH_SIZE * CONTEXT_LENGTH))
steps=$(((TOKEN_BUDGET + tokens_per_step - 1) / tokens_per_step))
warmup_iters="$(awk -v s="$steps" -v f="$WARMUP_FRACTION" 'BEGIN { v=int(s*f + 0.5); if (v < 1) v=1; print v }')"

ablation_tag="rmsnorm-${RMSNORM}_norm-${NORM_ORDER}_rope-${ROPE}_ffn-${FFN_ACTIVATION}"
if [[ -z "$WANDB_RUN_NAME" ]]; then
  WANDB_RUN_NAME="tinystories_${ablation_tag}_bs${BATCH_SIZE}_ctx${CONTEXT_LENGTH}_$(date +%Y%m%d_%H%M%S)"
fi

val_display="(disabled)"
if [[ "$USE_VAL" -eq 1 ]]; then
  val_display="$VAL_BIN"
fi

echo "============================================================"
echo "Step 3/3: train ablation model"
echo "train_bin: $TRAIN_BIN"
echo "val_bin:   $val_display"
echo "steps:     $steps"
echo "warmup:    $warmup_iters"
echo "save_dir:  $RUNS_DIR"
echo "ablation:  $ablation_tag"

train_args=(
  python -m cs336_basics.train_ablation
  --train-data "$TRAIN_BIN"
  --data-dtype "$DATA_DTYPE"
  --vocab-size "$VOCAB_SIZE"
  --context-length "$CONTEXT_LENGTH"
  --d-model "$D_MODEL"
  --num-heads "$NUM_HEADS"
  --d-ff "$D_FF"
  --num-layers "$NUM_LAYERS"
  --rope-theta "$ROPE_THETA"
  --rmsnorm "$RMSNORM"
  --norm-order "$NORM_ORDER"
  --rope "$ROPE"
  --ffn-activation "$FFN_ACTIVATION"
  --optimizer "$OPTIMIZER"
  --batch-size "$BATCH_SIZE"
  --total-iters "$steps"
  --learning-rate "$LEARNING_RATE"
  --min-learning-rate "$MIN_LEARNING_RATE"
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
  --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS"
  --save-dir "$RUNS_DIR"
  --device "$DEVICE"
  --seed "$SEED"
)

if [[ "$USE_VAL" -eq 1 ]]; then
  train_args+=(--val-data "$VAL_BIN")
fi

if [[ "$USE_WANDB" -eq 1 ]]; then
  train_args+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME" --wandb-mode "$WANDB_MODE")
  if [[ -n "$WANDB_ENTITY" ]]; then
    train_args+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

"$UV_BIN" run "${train_args[@]}"

echo "============================================================"
echo "Done."
echo "Tokenizer:"
echo "  $VOCAB_PKL"
echo "  $MERGES_PKL"
echo "Tokenized data:"
echo "  $TRAIN_BIN"
echo "  $val_display"
echo "Checkpoints:"
echo "  $RUNS_DIR"
