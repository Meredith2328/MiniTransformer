# MiniTransformer From Scratch

[CS336 Assignment 1 前半段记录：架构 | 十派的玩具箱](https://meredith2328.github.io/post/CS336-hw1-1/)

[CS336 Assignment 1 后半段记录：实验 | 十派的玩具箱](https://meredith2328.github.io/post/CS336-hw1-2/)

This repository is a personal implementation project built on top of [Stanford CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics/) . It turns the assignment components into a runnable small-scale language model training stack, from raw text to experiment tracking.

Current scope:

- byte-level BPE tokenizer training and text encoding
- Transformer language model in PyTorch
- custom AdamW, learning rate scheduling, and training loop
- `np.memmap`-based token dataset loading
- checkpointing, automatic resume, and checkpoint retention
- W&B logging, learning rate sweep, and batch size sweep

For full training commands, see [RUN.md](./RUN.md).

## Setup

PowerShell:

```powershell
conda create -n cs336 python==3.12
conda activate cs336
uv sync
```

Linux:

```bash
conda create -n cs336 python==3.12
conda activate cs336
uv sync
```

Download TinyStories:

```bash
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

## Quick Start

Run tests:

```bash
uv run pytest
```

Full TinyStories pipeline:

```bash
bash scripts/run_tinystories_train.sh \
  --conda-env cs336 \
  --use-wandb
```

Train directly from tokenized `.bin`:

```bash
bash scripts/run_tinystories_train.sh \
  --conda-env cs336 \
  --skip-bpe \
  --skip-tokenize \
  --train-bin data/tinystories_train.bin \
  --val-bin data/tinystories_val.bin \
  --data-dtype uint16
```

Resume training:

```bash
bash scripts/resume_training.sh \
  --conda-env cs336 \
  --run-dir runs/tinystories_base
```

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

Single-minibatch sanity check:

```bash
uv run python scripts/overfit_single_batch.py \
  --train-data data/tinystories_train.bin \
  --data-dtype uint16 \
  --vocab-size 10000
```

Prompt-based generation:

```bash
uv run python scripts/generate_prompt.py \
  --run-dir runs/tinystories_base \
  --prompt "Once upon a time"
```

Local web UI:

```bash
uv run python ask.py \
  --run-dir runs/tinystories_base
```

## Training Notes

A typical run directory contains:

```text
runs/<experiment_name>/
|- best.pt
|- latest.pt
|- final.pt
|- step_XXXXXXXX.pt
|- run_config.json
`- wandb/
```

Checkpoint policy:

- periodic `step_*.pt` files keep only the latest `3` by default
- `best.pt`, `latest.pt`, `final.pt`, and `interrupted_step_*.pt` are kept separately
- resume prefers `latest.pt` first

## Results

| Metric | Value |
| --- | --- |
| Best validation per-token loss | 1.37 |
| Best hyperparameter setting | batch size=128, lr = 0.0012, context_length = 256 |
| Peak training throughput | 260,000 tok/s |
| Tokenizer training / encoding speed | 3MB/s (one thread) |

### TinyStories Val Loss

![TinyStories Loss Placeholder](./docs/figures/tinystories_val_loss.png)

### Learning Rate Sweep

![Batch Size Sweep Placeholder](./docs/figures/lr_sweep.png)

### Batch Size Sweep

![Batch Size Sweep Placeholder](./docs/figures/bs_sweep.png)

## license

MIT
