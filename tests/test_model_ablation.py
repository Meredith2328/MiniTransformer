from __future__ import annotations

import sys

import torch

from cs336_basics.model import (
    ConfigurableTransformerBlock,
    ConfigurableTransformerLM,
    IdentityNorm,
    RMSNorm,
    SiLUFeedForward,
    SwiGLU,
)
from cs336_basics.train_ablation import parse_args


def test_configurable_transformer_lm_forward_without_norm_or_rope():
    model = ConfigurableTransformerLM(
        vocab_size=32,
        context_length=16,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        use_rmsnorm=False,
        norm_order="pre",
        use_rope=False,
        ffn_activation="swiglu",
    )
    token_ids = torch.randint(0, 32, (2, 8))

    logits = model(token_ids)

    assert logits.shape == (2, 8, 32)
    assert isinstance(model.final_norm, IdentityNorm)
    assert all(isinstance(layer.norm1, IdentityNorm) for layer in model.layers)
    assert all(isinstance(layer.norm2, IdentityNorm) for layer in model.layers)
    assert all(layer.attention.use_rope is False for layer in model.layers)
    assert all(isinstance(layer.ffn, SwiGLU) for layer in model.layers)


def test_configurable_transformer_block_post_norm_with_silu():
    block = ConfigurableTransformerBlock(
        d_model=32,
        num_heads=4,
        d_ff=64,
        max_seq_len=16,
        use_rmsnorm=True,
        norm_order="post",
        use_rope=True,
        ffn_activation="silu",
    )
    x = torch.randn(2, 8, 32)
    token_positions = torch.arange(8).unsqueeze(0).expand(2, -1)

    output = block(x, token_positions)

    assert output.shape == x.shape
    assert isinstance(block.norm1, RMSNorm)
    assert isinstance(block.norm2, RMSNorm)
    assert block.attention.use_rope is True
    assert isinstance(block.ffn, SiLUFeedForward)


def test_train_ablation_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_ablation", "--train-data", "train.bin", "--vocab-size", "128"])

    args = parse_args()

    assert args.rmsnorm == "remove"
    assert args.norm_order == "pre"
    assert args.rope == "remove"
    assert args.ffn_activation == "swiglu"


def test_train_ablation_toggles(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ablation",
            "--train-data",
            "train.bin",
            "--vocab-size",
            "128",
            "--rmsnorm",
            "keep",
            "--norm-order",
            "post",
            "--rope",
            "keep",
            "--ffn-activation",
            "silu",
        ],
    )

    args = parse_args()

    assert args.rmsnorm == "keep"
    assert args.norm_order == "post"
    assert args.rope == "keep"
    assert args.ffn_activation == "silu"
