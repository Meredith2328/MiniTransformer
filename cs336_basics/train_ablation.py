from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.model import ConfigurableTransformerLM
from cs336_basics.train import (
    build_checkpoint_extra_state,
    build_optimizer,
    estimate_loss,
    load_memmap_dataset,
    maybe_init_wandb,
    prune_old_step_checkpoints,
    resolve_device,
    validate_args as validate_base_args,
)
from cs336_basics.utils import (
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the ablation-friendly TransformerLM on tokenized binary data."
    )

    # Data
    parser.add_argument("--train-data", type=str, required=True, help="Path to training .bin file.")
    parser.add_argument("--val-data", type=str, default=None, help="Optional path to validation .bin file.")
    parser.add_argument(
        "--data-dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32", "int32", "int64"],
        help="Token dtype used in train/val .bin files.",
    )

    # Model
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size.")
    parser.add_argument("--context-length", type=int, default=1024, help="Maximum context length.")
    parser.add_argument("--d-model", type=int, default=768, help="Model hidden size.")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--d-ff", type=int, default=3072, help="Feedforward hidden size.")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta parameter.")
    parser.add_argument(
        "--rmsnorm",
        type=str,
        default="remove",
        choices=["keep", "remove"],
        help="Keep or remove RMSNorm. Default removes RMSNorm.",
    )
    parser.add_argument(
        "--norm-order",
        type=str,
        default="pre",
        choices=["pre", "post"],
        help="Norm placement. Default is pre-norm.",
    )
    parser.add_argument(
        "--rope",
        type=str,
        default="remove",
        choices=["keep", "remove"],
        help="Keep or remove RoPE. Default removes RoPE.",
    )
    parser.add_argument(
        "--ffn-activation",
        type=str,
        default="swiglu",
        choices=["swiglu", "silu"],
        help="Feed-forward module type. Default is SwiGLU.",
    )

    # Optimizer + schedule
    parser.add_argument(
        "--optimizer",
        type=str,
        default="custom_adamw",
        choices=["custom_adamw", "torch_adamw"],
        help="Optimizer implementation.",
    )
    parser.add_argument("--learning-rate", type=float, default=6e-4, help="Peak learning rate.")
    parser.add_argument("--min-learning-rate", type=float, default=6e-5, help="Final learning rate.")
    parser.add_argument("--warmup-iters", type=int, default=2000, help="Linear warmup iterations.")
    parser.add_argument("--total-iters", type=int, default=100000, help="Total training iterations.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Global grad clip max L2 norm.")

    # Training loop
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda/cuda:0/mps.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")

    # Logging / eval / checkpoints
    parser.add_argument("--log-interval", type=int, default=50, help="Log every N iterations.")
    parser.add_argument("--eval-interval", type=int, default=500, help="Run validation every N iterations.")
    parser.add_argument("--eval-iters", type=int, default=50, help="Validation minibatches per evaluation.")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save checkpoint every N iterations.")
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=3,
        help="How many periodic step_*.pt checkpoints to retain. Special checkpoints are always kept.",
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint output directory.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")

    # Optional W&B
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="cs336-assignment1", help="W&B project.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode.",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    validate_base_args(args)


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading train data from: {args.train_data}")
    train_dataset = load_memmap_dataset(args.train_data, args.data_dtype, args.context_length)
    val_dataset = None
    if args.val_data:
        print(f"Loading val data from: {args.val_data}")
        val_dataset = load_memmap_dataset(args.val_data, args.data_dtype, args.context_length)

    model = ConfigurableTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        theta=args.rope_theta,
        use_rmsnorm=args.rmsnorm == "keep",
        norm_order=args.norm_order,
        use_rope=args.rope == "keep",
        ffn_activation=args.ffn_activation,
    ).to(device)
    optimizer = build_optimizer(args, model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(
        "Ablations: "
        f"rmsnorm={args.rmsnorm}, norm_order={args.norm_order}, rope={args.rope}, ffn={args.ffn_activation}"
    )

    start_step = 1
    best_val_loss = float("inf")
    best_step = 0
    if args.resume:
        resumed_checkpoint = load_checkpoint(args.resume, model, optimizer, return_checkpoint=True)
        resumed_iter = int(resumed_checkpoint["iteration"])
        start_step = int(resumed_iter) + 1
        extra_state = resumed_checkpoint.get("extra_state", {})
        if "best_val_loss" in extra_state:
            best_val_loss = float(extra_state["best_val_loss"])
        if "best_step" in extra_state:
            best_step = int(extra_state["best_step"])
        print(f"Resumed from checkpoint {args.resume}, continuing at step {start_step}.")
        if best_step > 0 and np.isfinite(best_val_loss):
            print(f"Recovered best validation loss {best_val_loss:.4f} from step {best_step}.")

    run_config = vars(args).copy()
    run_config["resolved_device"] = device
    with (save_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    wandb_run = maybe_init_wandb(args, run_config, save_dir)

    model.train()
    last_completed_step = start_step - 1

    window_losses: list[float] = []
    tokens_since_log = 0
    last_log_time = time.perf_counter()
    train_start_time = time.perf_counter()

    print("Starting training...")
    try:
        for step in range(start_step, args.total_iters + 1):
            lr = get_lr_cosine_schedule(
                step,
                args.learning_rate,
                args.min_learning_rate,
                args.warmup_iters,
                args.total_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            inputs, targets = get_batch(train_dataset, args.batch_size, args.context_length, device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                gradient_clipping(model.parameters(), max_l2_norm=args.grad_clip)

            optimizer.step()

            loss_value = float(loss.item())
            window_losses.append(loss_value)
            last_completed_step = step
            tokens_since_log += args.batch_size * args.context_length

            if step % args.log_interval == 0:
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-8)
                avg_train_loss = float(np.mean(window_losses[-args.log_interval :]))
                tokens_per_second = tokens_since_log / elapsed
                print(
                    f"step {step:7d} | train_loss {avg_train_loss:.4f} | "
                    f"lr {lr:.3e} | tok/s {tokens_per_second:,.0f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "step": step,
                            "train/loss": avg_train_loss,
                            "train/lr": lr,
                            "train/tokens_per_second": tokens_per_second,
                        },
                        step=step,
                    )
                last_log_time = now
                tokens_since_log = 0

            if val_dataset is not None and step % args.eval_interval == 0:
                val_loss = estimate_loss(
                    model=model,
                    dataset=val_dataset,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    eval_iters=args.eval_iters,
                    device=device,
                )
                print(f"step {step:7d} | val_loss {val_loss:.4f}")
                if wandb_run is not None:
                    wandb_run.log({"step": step, "val/loss": val_loss}, step=step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    best_path = save_dir / "best.pt"
                    save_checkpoint(
                        model,
                        optimizer,
                        step,
                        best_path,
                        extra_state=build_checkpoint_extra_state(best_val_loss, best_step),
                    )
                    print(f"  saved new best checkpoint to {best_path} (val_loss={val_loss:.4f})")

            if step % args.save_interval == 0:
                ckpt_path = save_dir / f"step_{step:08d}.pt"
                extra_state = build_checkpoint_extra_state(best_val_loss, best_step)
                save_checkpoint(model, optimizer, step, ckpt_path, extra_state=extra_state)
                save_checkpoint(model, optimizer, step, save_dir / "latest.pt", extra_state=extra_state)
                print(f"  checkpoint saved: {ckpt_path}")
                removed = prune_old_step_checkpoints(save_dir, args.keep_last_checkpoints)
                if removed:
                    removed_names = ", ".join(path.name for path in removed)
                    print(f"  pruned old checkpoints: {removed_names}")

    except KeyboardInterrupt:
        interrupted_path = save_dir / f"interrupted_step_{last_completed_step:08d}.pt"
        save_checkpoint(
            model,
            optimizer,
            last_completed_step,
            interrupted_path,
            extra_state=build_checkpoint_extra_state(best_val_loss, best_step),
        )
        print(f"\nInterrupted. Saved checkpoint to {interrupted_path}")
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    final_path = save_dir / "final.pt"
    save_checkpoint(
        model,
        optimizer,
        last_completed_step,
        final_path,
        extra_state=build_checkpoint_extra_state(best_val_loss, best_step),
    )
    total_seconds = time.perf_counter() - train_start_time
    print(f"Training complete in {total_seconds:.1f}s. Final checkpoint: {final_path}")
    if val_dataset is not None and best_step > 0:
        print(f"Best validation loss {best_val_loss:.4f} at step {best_step}.")


def main() -> None:
    args = parse_args()
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    train(args)


if __name__ == "__main__":
    main()
