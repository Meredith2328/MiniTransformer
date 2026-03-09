from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW as TorchAdamW

from cs336_basics.model import AdamW as CustomAdamW
from cs336_basics.model import TransformerLM
from cs336_basics.utils import get_batch, gradient_clipping


NP_DTYPES = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit one fixed minibatch to verify training path.")
    parser.add_argument("--train-data", type=str, required=True, help="Path to tokenized train .bin")
    parser.add_argument(
        "--data-dtype",
        type=str,
        choices=sorted(NP_DTYPES.keys()),
        default="uint16",
        help="Token dtype in .bin",
    )
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--optimizer", choices=["custom_adamw", "torch_adamw"], default="custom_adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    train_path = Path(args.train_data)
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")
    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    dataset = np.memmap(train_path, dtype=NP_DTYPES[args.data_dtype], mode="r")
    if dataset.shape[0] < args.context_length + 1:
        raise ValueError("Dataset is too small for requested context length.")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        theta=args.rope_theta,
    ).to(device)

    opt_cls = CustomAdamW if args.optimizer == "custom_adamw" else TorchAdamW
    optimizer = opt_cls(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    model.train()
    fixed_inputs, fixed_targets = get_batch(dataset, args.batch_size, args.context_length, device)
    first_loss = None
    last_loss = None

    print("Overfit-one-batch configuration:")
    print(f"  train_data: {train_path}")
    print(f"  device: {device}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  context_length: {args.context_length}")
    print(f"  steps: {args.steps}")
    print(f"  lr: {args.learning_rate}")

    for step in range(1, args.steps + 1):
        logits = model(fixed_inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), fixed_targets.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), max_l2_norm=args.grad_clip)
        optimizer.step()

        loss_value = float(loss.item())
        if first_loss is None:
            first_loss = loss_value
        last_loss = loss_value

        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            print(f"step {step:4d} | loss {loss_value:.6f}")

    assert first_loss is not None and last_loss is not None
    improvement = first_loss - last_loss
    print("Overfit result:")
    print(f"  first_loss: {first_loss:.6f}")
    print(f"  final_loss: {last_loss:.6f}")
    print(f"  improvement: {improvement:.6f}")

    if last_loss >= first_loss:
        raise RuntimeError("Single-batch overfit check failed: loss did not decrease.")


if __name__ == "__main__":
    main()
