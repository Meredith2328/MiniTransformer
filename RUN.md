# RUN Guide (TinyStories)

This document is Linux-first and includes the shortest commands to run end-to-end.

## Why `.bin` instead of `.txt` for training

The training script `cs336_basics/train.py` reads token IDs with `np.memmap`.
That requires a contiguous numeric array on disk (`.bin`), not raw text.

Using `.txt` directly would force tokenization during every training step, which is much slower and less reproducible for sweeps.

Standard pipeline:

1. `txt -> train BPE tokenizer`
2. `txt -> token ids .bin`
3. `train on .bin`

## Linux: one command for full pipeline

From repo root:

```bash
bash scripts/run_tinystories_train.sh \
  --conda-env cs336 \
  --use-wandb
```

What this does:

1. Trains BPE tokenizer (unless `--skip-bpe`)
2. Tokenizes train/valid text into `data/tinystories_train.bin` and `data/tinystories_val.bin` (unless `--skip-tokenize`)
3. Starts model training

Default output paths:

- tokenizer: `tokenizer/tinystories_bpe_vocab.pkl`, `tokenizer/tinystories_bpe_merges.pkl`
- tokenized data: `data/tinystories_train.bin`, `data/tinystories_val.bin`
- checkpoints/logs: `runs/tinystories_base`

Checkpoint retention:

- periodic `step_*.pt` checkpoints now keep only the newest `3` by default
- `best.pt`, `latest.pt`, `final.pt`, and `interrupted_step_*.pt` are kept separately
- override with `--keep-last-checkpoints N` on `scripts/run_tinystories_train.sh` or `scripts/run_tinystories_train.ps1`

## Resume training from a checkpoint

If a run was interrupted, the most convenient resume path is:

```bash
uv run python scripts/resume_training.py \
  --checkpoint runs/tinystories_base/latest.pt
```

By default this script looks for `run_config.json` in the same directory as the checkpoint, rebuilds the original training command, and adds `--resume <checkpoint>`.

Recommended checkpoint choice:

- `latest.pt`: normal resume target
- `interrupted_step_XXXXXXXX.pt`: when you stopped with `Ctrl+C`
- `step_XXXXXXXX.pt`: manual fallback if needed

Common override example: extend total training length while resuming

```bash
uv run python scripts/resume_training.py \
  --checkpoint runs/tinystories_base/latest.pt \
  --set total_iters=40000
```

You can override any saved config entry with repeated `--set key=value`, for example:

- `--set device=cuda`
- `--set wandb_mode=disabled`
- `--set keep_last_checkpoints=3`

If your config file is not in the checkpoint directory, pass it explicitly:

```bash
uv run python scripts/resume_training.py \
  --checkpoint runs/tinystories_base/latest.pt \
  --config runs/tinystories_base/run_config.json
```

## Linux: baseline run for assignment settings

```bash
bash scripts/run_tinystories_train.sh \
  --conda-env cs336 \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-heads 16 \
  --d-ff 1344 \
  --num-layers 4 \
  --rope-theta 10000 \
  --batch-size 64 \
  --token-budget 327680000 \
  --learning-rate 6e-4 \
  --min-learning-rate 6e-5 \
  --use-wandb
```

The script computes:

- `total_iters = ceil(token_budget / (batch_size * context_length))`
- `warmup_iters` from `--warmup-fraction` (default `0.02`)

## Linux: tokenize only

```bash
uv run python scripts/tokenize_to_bin.py \
  --input-text data/TinyStoriesV2-GPT4-train.txt \
  --vocab-pkl tokenizer/tinystories_bpe_vocab.pkl \
  --merges-pkl tokenizer/tinystories_bpe_merges.pkl \
  --output-bin data/tinystories_train.bin \
  --dtype uint16
```

Run again for valid set with `--input-text data/TinyStoriesV2-GPT4-valid.txt` and `--output-bin data/tinystories_val.bin`.

## Linux: single-minibatch overfit sanity check

```bash
uv run python scripts/overfit_single_batch.py \
  --train-data data/tinystories_train.bin \
  --data-dtype uint16 \
  --vocab-size 10000
```

If loss drops clearly, forward/backward/optimizer path is healthy.

## Linux: sweep scripts

Learning-rate sweep:

```bash
bash scripts/lr_sweep.sh \
  --conda-env cs336 \
  --train-data data/tinystories_train.bin \
  --val-data data/tinystories_val.bin \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-heads 16 \
  --d-ff 1344 \
  --num-layers 4 \
  --batch-size 64 \
  --use-wandb
```

Batch-size sweep:

```bash
bash scripts/batch_sweep.sh \
  --conda-env cs336 \
  --train-data data/tinystories_train.bin \
  --val-data data/tinystories_val.bin \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-heads 16 \
  --d-ff 1344 \
  --num-layers 4 \
  --batch-sizes 1,8,16,32,64,128 \
  --use-wandb
```

Both scripts write a `results.csv` in their run root.

## Windows scripts still available

If you also use a Windows workstation, these remain available:

- `scripts/run_tinystories_train.ps1`
- `scripts/lr_sweep.ps1`
- `scripts/batch_sweep.ps1`
