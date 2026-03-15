from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .bpe import ByteLevelBPE
    from .model import TransformerLM
except ImportError:
    from bpe import ByteLevelBPE
    from model import TransformerLM


DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_BPE_VOCAB_PATH = Path(__file__).resolve().parent.parent / "tokenizer" / "tinystories_bpe_vocab.pkl"
DEFAULT_BPE_MERGES_PATH = Path(__file__).resolve().parent.parent / "tokenizer" / "tinystories_bpe_merges.pkl"


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: list[int]) -> str:
        ...


class GPT2TokenizerAdapter:
    def __init__(self) -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding("gpt2")
        self.eos_token_id = self._encoding.eot_token

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, allowed_special={DEFAULT_EOS_TOKEN})

    def decode(self, token_ids: list[int]) -> str:
        return self._encoding.decode(token_ids)


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    theta: float = 10000.0


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0 when sampling.")
    return logits / temperature


def top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1].")
    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = cumulative_probs > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    if temperature < 0:
        raise ValueError("temperature must be >= 0.")
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1].")

    if temperature == 0:
        return int(torch.argmax(logits, dim=-1).item())

    scaled_logits = apply_temperature(logits, temperature)
    probs = F.softmax(scaled_logits, dim=-1)
    probs = top_p_filtering(probs, top_p)
    return int(torch.multinomial(probs, num_samples=1).item())


def _encode_prompt(prompt: str | list[int], tokenizer: TokenizerLike | None) -> list[int]:
    if isinstance(prompt, str):
        if tokenizer is None:
            raise ValueError("String prompt requires a tokenizer.")
        return tokenizer.encode(prompt)
    return list(prompt)


def _infer_eos_token_id(tokenizer: TokenizerLike | None, explicit_eos_token_id: int | None) -> int | None:
    if explicit_eos_token_id is not None:
        return explicit_eos_token_id
    if tokenizer is None:
        return None

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)

    if isinstance(tokenizer, ByteLevelBPE):
        target = DEFAULT_EOS_TOKEN.encode("utf-8")
        for token_id, token in tokenizer.vocab.items():
            if token == target:
                return int(token_id)
    return None


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = str(key)

        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module."):]
        if normalized_key.startswith("_orig_mod."):
            normalized_key = normalized_key[len("_orig_mod."):]

        normalized_key = normalized_key.replace("token_embeddings.weight", "token_embedding.weight")
        normalized_key = normalized_key.replace(".attn.q_proj.weight", ".attention.W_q.weight")
        normalized_key = normalized_key.replace(".attn.k_proj.weight", ".attention.W_k.weight")
        normalized_key = normalized_key.replace(".attn.v_proj.weight", ".attention.W_v.weight")
        normalized_key = normalized_key.replace(".attn.output_proj.weight", ".attention.W_o.weight")
        normalized_key = normalized_key.replace(".ln1.weight", ".norm1.weight")
        normalized_key = normalized_key.replace(".ln2.weight", ".norm2.weight")
        normalized_key = normalized_key.replace("ln_final.weight", "final_norm.weight")

        normalized[normalized_key] = value
    return normalized


def _extract_state_dict(checkpoint_obj: Any) -> dict[str, Any]:
    if isinstance(checkpoint_obj, dict):
        if isinstance(checkpoint_obj.get("model_state_dict"), dict):
            return checkpoint_obj["model_state_dict"]
        if isinstance(checkpoint_obj.get("state_dict"), dict):
            return checkpoint_obj["state_dict"]
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError("Unsupported checkpoint format. Expected state_dict or {'model_state_dict': ...}.")


def _load_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    return loaded


def _resolve_model_config(args: argparse.Namespace, checkpoint_obj: Any) -> ModelConfig:
    checkpoint_config: dict[str, Any] = {}
    if isinstance(checkpoint_obj, dict):
        if isinstance(checkpoint_obj.get("config"), dict):
            checkpoint_config.update(checkpoint_obj["config"])
        for key in ("vocab_size", "context_length", "d_model", "num_heads", "d_ff", "num_layers", "theta", "rope_theta"):
            if key in checkpoint_obj:
                checkpoint_config[key] = checkpoint_obj[key]

    json_config = _load_optional_json(args.config_json)

    cli_overrides = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "theta": args.theta,
    }

    merged: dict[str, Any] = {}
    merged.update(checkpoint_config)
    merged.update(json_config)
    merged.update({k: v for k, v in cli_overrides.items() if v is not None})

    if "theta" not in merged and "rope_theta" in merged:
        merged["theta"] = merged["rope_theta"]

    required_keys = ("vocab_size", "context_length", "d_model", "num_heads", "d_ff", "num_layers")
    missing = [k for k in required_keys if merged.get(k) is None]
    if missing:
        raise ValueError(
            "Missing model hyperparameters: "
            + ", ".join(missing)
            + ". Provide them via checkpoint config, --config-json, or CLI flags."
        )

    return ModelConfig(
        vocab_size=int(merged["vocab_size"]),
        context_length=int(merged["context_length"]),
        d_model=int(merged["d_model"]),
        num_heads=int(merged["num_heads"]),
        d_ff=int(merged["d_ff"]),
        num_layers=int(merged["num_layers"]),
        theta=float(merged.get("theta", 10000.0)),
    )


def load_bpe_tokenizer(
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
) -> ByteLevelBPE:
    tokenizer = ByteLevelBPE()
    with vocab_path.open("rb") as f:
        tokenizer.vocab = pickle.load(f)
    with merges_path.open("rb") as f:
        tokenizer.merges = pickle.load(f)
    tokenizer._special_tokens = list(special_tokens)
    return tokenizer


def generate(
    model: nn.Module,
    prompt: str | list[int],
    tokenizer: TokenizerLike | None = None,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str | torch.device = "cpu",
) -> list[int]:
    model.eval()
    prompt_ids = _encode_prompt(prompt, tokenizer)

    if not prompt_ids:
        if eos_token_id is None:
            raise ValueError("Prompt is empty and eos_token_id is unknown. Provide non-empty prompt or --eos-token-id.")
        prompt_ids = [int(eos_token_id)]

    generated_ids = list(prompt_ids)
    context_length = int(getattr(model, "context_length", len(generated_ids) + max_new_tokens))

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            window_ids = generated_ids[-context_length:]

            input_ids = torch.tensor([window_ids], dtype=torch.long, device=device)
            token_positions = torch.arange(
                len(window_ids),
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)

            logits = model(input_ids, token_positions=token_positions)
            next_token_logits = logits[0, -1, :]
            next_token_id = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_p=top_p,
            )

            generated_ids.append(next_token_id)
            if eos_token_id is not None and next_token_id == eos_token_id:
                break

    return generated_ids


def run_generation(
    model: nn.Module,
    prompt: str | list[int],
    tokenizer: TokenizerLike | None = None,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str | torch.device = "cpu",
) -> str | list[int]:
    generated_ids = generate(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        device=device,
    )

    if tokenizer is not None and isinstance(prompt, str):
        return tokenizer.decode(generated_ids)
    return generated_ids


def _parse_prompt_ids(text: str) -> list[int]:
    stripped = text.strip()
    if not stripped:
        return []

    if stripped.startswith("[") and stripped.endswith("]"):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("--prompt-ids JSON must be a list.")
        return [int(x) for x in parsed]

    return [int(piece) for piece in stripped.replace(",", " ").split()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode text from a trained Transformer language model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint/state_dict.")
    parser.add_argument("--config-json", type=Path, default=None, help="Optional JSON file with model hyperparameters.")

    parser.add_argument("--prompt", type=str, default="", help="Text prompt.")
    parser.add_argument("--prompt-ids", type=str, default=None, help='Prompt token ids, e.g. "12,42,5" or "[12,42,5]".')
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. Use 0 for greedy decode.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) sampling threshold in (0, 1].")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch device.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")

    parser.add_argument(
        "--tokenizer-mode",
        choices=("gpt2", "bpe", "none"),
        default="gpt2",
        help="Tokenizer backend. Use 'none' to decode with token ids only.",
    )
    parser.add_argument("--bpe-vocab-path", type=Path, default=DEFAULT_BPE_VOCAB_PATH, help="Vocab path for bpe mode.")
    parser.add_argument("--bpe-merges-path", type=Path, default=DEFAULT_BPE_MERGES_PATH, help="Merges path for bpe mode.")
    parser.add_argument(
        "--special-tokens",
        type=str,
        default=DEFAULT_EOS_TOKEN,
        help="Comma-separated special tokens for bpe mode.",
    )
    parser.add_argument("--eos-token-id", type=int, default=None, help="Override eos token id used to stop generation.")
    parser.add_argument("--completion-only", action="store_true", help="Print only generated continuation, not prompt+continuation.")
    parser.add_argument("--print-token-ids", action="store_true", help="Also print generated token ids to stderr.")
    parser.add_argument("--strict-load", action="store_true", help="Fail if checkpoint has missing/unexpected keys.")

    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--theta", type=float, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be >= 0.")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0.")
    if not (0 < args.top_p <= 1):
        raise ValueError("--top-p must be in (0, 1].")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    checkpoint_obj = torch.load(args.checkpoint, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint_obj)
    normalized_state_dict = _normalize_state_dict_keys(state_dict)
    model_config = _resolve_model_config(args, checkpoint_obj)

    model = TransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        num_layers=model_config.num_layers,
        theta=model_config.theta,
    ).to(args.device)

    try:
        missing_keys, unexpected_keys = model.load_state_dict(normalized_state_dict, strict=args.strict_load)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load checkpoint weights into model: {exc}") from exc

    if not args.strict_load:
        if missing_keys:
            print(f"[warn] Missing keys while loading checkpoint: {missing_keys}", file=sys.stderr)
        if unexpected_keys:
            print(f"[warn] Unexpected keys while loading checkpoint: {unexpected_keys}", file=sys.stderr)

    tokenizer: TokenizerLike | None
    if args.tokenizer_mode == "none":
        tokenizer = None
    elif args.tokenizer_mode == "gpt2":
        tokenizer = GPT2TokenizerAdapter()
    else:
        special_tokens = [token.strip() for token in args.special_tokens.split(",") if token.strip()]
        tokenizer = load_bpe_tokenizer(args.bpe_vocab_path, args.bpe_merges_path, special_tokens)

    eos_token_id = _infer_eos_token_id(tokenizer, args.eos_token_id)

    if args.prompt_ids is not None:
        prompt_ids = _parse_prompt_ids(args.prompt_ids)
        prompt_for_generate: str | list[int] = prompt_ids
    else:
        prompt_for_generate = args.prompt

    generated_ids = generate(
        model=model,
        prompt=prompt_for_generate,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        device=args.device,
    )

    if args.prompt_ids is not None:
        prompt_ids = _parse_prompt_ids(args.prompt_ids)
    else:
        prompt_ids = _encode_prompt(args.prompt, tokenizer) if tokenizer is not None else []

    completion_ids = generated_ids[len(prompt_ids) :]
    output_ids = completion_ids if args.completion_only else generated_ids

    if tokenizer is None:
        print(json.dumps(output_ids))
    else:
        print(tokenizer.decode(output_ids))

    if args.print_token_ids:
        print(f"token_ids={generated_ids}", file=sys.stderr)


if __name__ == "__main__":
    main()
