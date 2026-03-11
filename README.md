# MiniTransformer From Scratch

This repository is a personal implementation project built on top of Stanford CS336 Assignment 1. The goal is to assemble a small but complete Transformer language model training stack from scratch, with enough engineering support to actually run experiments instead of only passing unit tests.

The repository now covers the full path from raw text to model training:

- byte-level BPE tokenizer training and text encoding
- Transformer language model implementation in PyTorch
- custom AdamW, learning rate scheduling, and training loop
- `np.memmap`-based loading for large token datasets
- checkpointing, automatic resume, and periodic checkpoint retention
- W&B logging, learning rate sweep, and batch size sweep
- tokenization profiling utilities

For the shortest end-to-end training commands, see [RUN.md](./RUN.md).

## What Is Implemented

### 1. Tokenizer

- [`cs336_basics/bpe.py`](./cs336_basics/bpe.py): BPE training and encoding logic
- [`cs336_basics/train_bpe.py`](./cs336_basics/train_bpe.py): tokenizer training entrypoint
- [`scripts/tokenize_to_bin.py`](./scripts/tokenize_to_bin.py): convert raw `.txt` text into training-ready `.bin`

### 2. Model

[`cs336_basics/model.py`](./cs336_basics/model.py) includes:

- `Embedding`
- `RMSNorm`
- `RoPE`
- `MultiHeadSelfAttention`
- `SwiGLU`
- `TransformerBlock`
- `TransformerLM`
- custom `AdamW`

### 3. Training System

[`cs336_basics/train.py`](./cs336_basics/train.py) includes:

- `np.memmap` dataset loading
- train / validation loop
- cosine learning rate schedule
- gradient clipping
- checkpoint saving
- `best.pt` / `latest.pt` / `final.pt`
- resume training
- W&B logging

### 4. Experiment Workflow

[`scripts/`](./scripts/) includes:

- [`run_tinystories_train.sh`](./scripts/run_tinystories_train.sh) / [`run_tinystories_train.ps1`](./scripts/run_tinystories_train.ps1)
  - unified training entrypoint
  - supports full pipeline from raw `.txt`
  - also supports direct training from prebuilt `.bin`
- [`resume_training.sh`](./scripts/resume_training.sh)
  - auto-select checkpoint and resume training
- [`lr_sweep.sh`](./scripts/lr_sweep.sh) / [`lr_sweep.ps1`](./scripts/lr_sweep.ps1)
  - learning rate sweep
- [`batch_sweep.sh`](./scripts/batch_sweep.sh) / [`batch_sweep.ps1`](./scripts/batch_sweep.ps1)
  - batch size sweep
- [`overfit_single_batch.py`](./scripts/overfit_single_batch.py)
  - single-minibatch overfit sanity check

### 5. Profiling Practice

If you want to practice tokenizer profiling without touching the main training loop, see:

- [`scripts/README.md`](./scripts/README.md)
- [`scripts/profile_tokenization.py`](./scripts/profile_tokenization.py)
- [`scripts/inspect_profile.py`](./scripts/inspect_profile.py)
- [`scripts/TOKENIZATION_PROFILING_EXPERIMENT.md`](./scripts/TOKENIZATION_PROFILING_EXPERIMENT.md)

## Repository Layout

```text
assignment1-basics/
|- cs336_basics/
|  |- bpe.py
|  |- model.py
|  |- train.py
|  |- train_bpe.py
|  |- resume_training.py
|  `- utils.py
|- scripts/
|  |- run_tinystories_train.sh
|  |- run_tinystories_train.ps1
|  |- resume_training.sh
|  |- lr_sweep.sh
|  |- batch_sweep.sh
|  |- tokenize_to_bin.py
|  |- overfit_single_batch.py
|  `- README.md
|- docs/
|  `- figures/
|- data/
|- runs/
|- tokenizer/
|- tests/
|- RUN.md
`- README.md
```

## Environment Setup

### Conda + uv

If you are using the `cs336` conda environment for this project:

PowerShell:

```powershell
& C:\Software\Miniconda\shell\condabin\conda-hook.ps1
conda activate C:\Software\Miniconda\envs\cs336
uv sync
```

Linux:

```bash
conda activate cs336
uv sync
```

After that, prefer running commands through `uv run` or the provided scripts.

## Download Data

Download TinyStories and the OpenWebText sample:

```bash
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Quick Start

### 1. Run tests

```bash
uv run pytest
```

### 2. Full TinyStories pipeline

Linux:

```bash
bash scripts/run_tinystories_train.sh \
  --conda-env cs336 \
  --use-wandb
```

Windows PowerShell:

```powershell
.\scripts\run_tinystories_train.ps1 -UseWandb
```

### 3. Train directly from `.bin`

```bash
bash scripts/run_tinystories_train.sh \
  --conda-env cs336 \
  --skip-bpe \
  --skip-tokenize \
  --train-bin data/tinystories_train.bin \
  --val-bin data/tinystories_val.bin \
  --data-dtype uint16
```

### 4. Single-minibatch overfit check

```bash
uv run python scripts/overfit_single_batch.py \
  --train-data data/tinystories_train.bin \
  --data-dtype uint16 \
  --vocab-size 10000
```

### 5. Resume training

```bash
bash scripts/resume_training.sh \
  --conda-env cs336 \
  --run-dir runs/tinystories_base
```

### 6. Hyperparameter sweeps

Learning rate sweep:

```bash
bash scripts/lr_sweep.sh \
  --conda-env cs336 \
  --train-data data/tinystories_train.bin \
  --val-data data/tinystories_val.bin \
  --use-wandb
```

Batch size sweep:

```bash
bash scripts/batch_sweep.sh \
  --conda-env cs336 \
  --train-data data/tinystories_train.bin \
  --val-data data/tinystories_val.bin \
  --use-wandb
```

For full command examples and baseline settings, see [RUN.md](./RUN.md).

## Training Outputs

A typical training run directory looks like:

```text
runs/<experiment_name>/
|- best.pt
|- latest.pt
|- final.pt
|- step_00001000.pt
|- run_config.json
`- wandb/
```

Checkpoint policy:

- periodic `step_*.pt` files keep only the latest `3` by default
- `best.pt`, `latest.pt`, `final.pt`, and `interrupted_step_*.pt` are retained separately
- resume prefers `latest.pt` first

## Suggested Metrics To Report

You can fill this section with your final experiment results later. A good project README should usually report at least:

- TinyStories train / validation loss curves
- learning rate sweep comparison
- batch size sweep comparison
- throughput or profiling summary

Suggested table:

| Metric | Value |
| --- | --- |
| Best validation per-token loss | `TODO` |
| Best hyperparameter setting | `TODO` |
| Peak training throughput | `TODO` |
| Tokenizer training / encoding speed | `TODO` |
| W&B run URL | `TODO` |

## Figure Placeholders

The figures below are placeholders. Once you have final results, you can:

1. put your exported figures under `docs/figures/`
2. update the image paths below
3. or replace the placeholder files directly

### 1. TinyStories Train / Val Loss Curve

![TinyStories Loss Placeholder](./docs/figures/tinystories_loss_placeholder.svg)

Suggested replacement:

- `docs/figures/tinystories_loss_curve.png`
- or an exported W&B loss plot

### 2. Learning Rate Sweep

![Learning Rate Sweep Placeholder](./docs/figures/lr_sweep_placeholder.svg)

Suggested replacement:

- `docs/figures/lr_sweep.png`
- a plot showing stable, best, and divergent learning rates

### 3. Batch Size Sweep

![Batch Size Sweep Placeholder](./docs/figures/batch_sweep_placeholder.svg)

Suggested replacement:

- `docs/figures/batch_sweep.png`
- a plot comparing final loss or loss curves across batch sizes

### 4. Throughput / Profiling Summary

![Throughput Placeholder](./docs/figures/throughput_placeholder.svg)

Suggested replacement:

- `docs/figures/throughput_profile.png`
- tokenizer throughput, training tok/s, or stage-wise runtime breakdown

## Additional Notes

- the original course handout is in [`cs336_spring2025_assignment1_basics.pdf`](./cs336_spring2025_assignment1_basics.pdf)
- tokenization profiling notes are in [`scripts/README.md`](./scripts/README.md)
- if your immediate goal is to run TinyStories experiments, start from [RUN.md](./RUN.md)

## License

This repository keeps the original course repository license. See [`LICENSE`](./LICENSE).
