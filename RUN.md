# RUN Guide (TinyStories)

This document is Linux-first and focuses on the shortest commands that cover the full TinyStories workflow.

## Why `.bin` instead of `.txt` for training

`cs336_basics/train.py` loads token IDs through `np.memmap`, so the training loop expects a contiguous numeric array on disk.
That is exactly what the tokenized `.bin` files provide.
Using raw `.txt` for training would force repeated tokenization inside the training loop, which is slower and makes sweeps less reproducible.

Standard pipeline:

1. `txt -> train BPE tokenizer`
2. `txt -> token ids .bin`
3. `train on .bin`

## Full pipeline: one command

From repo root:

```bash
bash scripts/run_tinystories_train.sh \
  --conda-env cs336 \
  --use-wandb
```

This wrapper handles three stages:

1. train BPE unless `--skip-bpe`
2. tokenize train/val text unless `--skip-tokenize`
3. launch `cs336_basics.train`

If `uv` is not on your shell `PATH`, the Linux shell wrappers also accept `UV_BIN=/absolute/path/to/uv`.

Default outputs:

- tokenizer: `tokenizer/tinystories_bpe_vocab.pkl`, `tokenizer/tinystories_bpe_merges.pkl`
- tokenized data: `data/tinystories_train.bin`, `data/tinystories_val.bin`
- checkpoints/logs: `runs/tinystories_base`

## Train directly from existing `.bin`

If BPE and tokenization are already done:

```bash
bash scripts/run_tinystories_train.sh \
  --conda-env cs336 \
  --skip-bpe \
  --skip-tokenize \
  --train-bin data/tinystories_train.bin \
  --val-bin data/tinystories_val.bin \
  --data-dtype uint16
```

If you want train-only without validation, disable validation explicitly:

```bash
bash scripts/run_tinystories_train.sh \
  --conda-env cs336 \
  --skip-bpe \
  --skip-tokenize \
  --train-bin data/tinystories_train.bin \
  --val-bin "" \
  --val-txt ""
```

## Assignment baseline configuration

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

The wrapper computes:

- `total_iters = ceil(token_budget / (batch_size * context_length))`
- `warmup_iters` from `--warmup-fraction` (default `0.02`)

## Checkpoint retention

Periodic checkpoints now keep only the newest `3` `step_*.pt` files by default.
These special files are retained separately:

- `best.pt`
- `latest.pt`
- `final.pt`
- `interrupted_step_*.pt`

Override retention with `--keep-last-checkpoints N`.
Because `lr_sweep` and `batch_sweep` now reuse `run_tinystories_train`, they inherit the same checkpoint policy automatically.

## Resume training

Preferred Linux entrypoint:

```bash
bash scripts/resume_training.sh \
  --conda-env cs336
```

This defaults to `runs/tinystories_base` and auto-selects checkpoints in this order:

1. `latest.pt`
2. newest `interrupted_step_XXXXXXXX.pt`
3. newest `step_XXXXXXXX.pt`
4. `final.pt`

Point it at another run directory when needed:

```bash
bash scripts/resume_training.sh \
  --conda-env cs336 \
  --run-dir runs/my_experiment
```

Explicit checkpoint also works:

```bash
bash scripts/resume_training.sh \
  --conda-env cs336 \
  --checkpoint runs/tinystories_base/latest.pt
```

The underlying module is:

```bash
uv run python -m cs336_basics.resume_training --dry-run
```

Resume reads `run_config.json`, reconstructs the original training command, appends `--resume <checkpoint>`, and restores saved best-validation metadata so `best.pt` is not overwritten after restart by a worse validation score.

Useful overrides:

```bash
bash scripts/resume_training.sh \
  --conda-env cs336 \
  --checkpoint runs/tinystories_base/latest.pt \
  --set total_iters=40000 \
  --set wandb_mode=disabled
```

## Sweep scripts

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

Both sweeps write a `results.csv` under their run root and now delegate actual training to `run_tinystories_train`.

## Tokenize only

```bash
uv run python scripts/tokenize_to_bin.py \
  --input-text data/TinyStoriesV2-GPT4-train.txt \
  --vocab-pkl tokenizer/tinystories_bpe_vocab.pkl \
  --merges-pkl tokenizer/tinystories_bpe_merges.pkl \
  --output-bin data/tinystories_train.bin \
  --dtype uint16
```

Run the same command for validation with `--input-text data/TinyStoriesV2-GPT4-valid.txt` and `--output-bin data/tinystories_val.bin`.

## Single-minibatch sanity check

```bash
uv run python scripts/overfit_single_batch.py \
  --train-data data/tinystories_train.bin \
  --data-dtype uint16 \
  --vocab-size 10000
```

## Windows scripts

The Windows entrypoints are still available:

- `scripts/run_tinystories_train.ps1`
- `scripts/lr_sweep.ps1`
- `scripts/batch_sweep.ps1`
