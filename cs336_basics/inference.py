from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .generate import (
    DEFAULT_BPE_MERGES_PATH,
    DEFAULT_BPE_VOCAB_PATH,
    DEFAULT_EOS_TOKEN,
    GPT2TokenizerAdapter,
    TokenizerLike,
    _encode_prompt,
    _extract_state_dict,
    _infer_eos_token_id,
    _normalize_state_dict_keys,
    _resolve_model_config,
    generate,
    load_bpe_tokenizer,
)
from .model import TransformerLM


DEFAULT_RUN_DIR = Path("runs") / "tinystories_base"
CHECKPOINT_PRIORITY = ("best.pt", "latest.pt", "final.pt")
INTERRUPTED_PREFIX = "interrupted_step_"
STEP_PREFIX = "step_"


@dataclass(frozen=True)
class GenerationSpec:
    checkpoint: Path
    config_json: Path | None
    device: str
    tokenizer_mode: str
    bpe_vocab_path: Path
    bpe_merges_path: Path
    special_tokens: tuple[str, ...]
    eos_token_id: int | None = None
    strict_load: bool = False


@dataclass
class GenerationResult:
    text: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    completion_token_ids: list[int]


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def parse_special_tokens(raw_value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if raw_value is None:
        return (DEFAULT_EOS_TOKEN,)
    if isinstance(raw_value, str):
        tokens = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
        return tuple(tokens or [DEFAULT_EOS_TOKEN])
    return tuple(token for token in raw_value if token)


def _parse_step_from_path(path: Path, prefix: str) -> int:
    stem = path.stem
    if not stem.startswith(prefix):
        return -1
    try:
        return int(stem[len(prefix) :])
    except ValueError:
        return -1


def select_generation_checkpoint(run_dir: Path) -> Path:
    for name in CHECKPOINT_PRIORITY:
        candidate = run_dir / name
        if candidate.exists():
            return candidate

    interrupted_paths = sorted(
        run_dir.glob("interrupted_step_*.pt"),
        key=lambda path: _parse_step_from_path(path, INTERRUPTED_PREFIX),
    )
    if interrupted_paths:
        return interrupted_paths[-1]

    step_paths = sorted(
        run_dir.glob("step_*.pt"),
        key=lambda path: _parse_step_from_path(path, STEP_PREFIX),
    )
    if step_paths:
        return step_paths[-1]

    raise FileNotFoundError(
        f"No checkpoint found in {run_dir}. "
        "Looked for best.pt, latest.pt, final.pt, interrupted_step_*.pt, and step_*.pt."
    )


def build_generation_spec(
    *,
    checkpoint: Path | None = None,
    run_dir: Path | None = None,
    config_json: Path | None = None,
    tokenizer_mode: str = "bpe",
    bpe_vocab_path: Path | None = DEFAULT_BPE_VOCAB_PATH,
    bpe_merges_path: Path | None = DEFAULT_BPE_MERGES_PATH,
    special_tokens: str | list[str] | tuple[str, ...] | None = None,
    eos_token_id: int | None = None,
    device: str = "auto",
    strict_load: bool = False,
) -> GenerationSpec:
    selected_checkpoint = checkpoint
    if selected_checkpoint is None:
        resolved_run_dir = run_dir if run_dir is not None else DEFAULT_RUN_DIR
        selected_checkpoint = select_generation_checkpoint(Path(resolved_run_dir))
    selected_checkpoint = Path(selected_checkpoint)
    if not selected_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {selected_checkpoint}")

    resolved_config = Path(config_json) if config_json is not None else None
    if resolved_config is None:
        candidate = selected_checkpoint.parent / "run_config.json"
        if candidate.exists():
            resolved_config = candidate

    return GenerationSpec(
        checkpoint=selected_checkpoint,
        config_json=resolved_config,
        device=resolve_device(device),
        tokenizer_mode=tokenizer_mode,
        bpe_vocab_path=Path(bpe_vocab_path) if bpe_vocab_path is not None else DEFAULT_BPE_VOCAB_PATH,
        bpe_merges_path=Path(bpe_merges_path) if bpe_merges_path is not None else DEFAULT_BPE_MERGES_PATH,
        special_tokens=parse_special_tokens(special_tokens),
        eos_token_id=eos_token_id,
        strict_load=strict_load,
    )


class GenerationBackend:
    def __init__(self, spec: GenerationSpec) -> None:
        self.spec = spec
        self._model: TransformerLM | None = None
        self._tokenizer: TokenizerLike | None = None
        self._eos_token_id: int | None = None
        self._missing_keys: list[str] = []
        self._unexpected_keys: list[str] = []
        self._model_config: Any = None

    def _build_tokenizer(self) -> TokenizerLike | None:
        if self.spec.tokenizer_mode == "none":
            return None
        if self.spec.tokenizer_mode == "gpt2":
            return GPT2TokenizerAdapter()
        if not self.spec.bpe_vocab_path.exists():
            raise FileNotFoundError(f"BPE vocab path not found: {self.spec.bpe_vocab_path}")
        if not self.spec.bpe_merges_path.exists():
            raise FileNotFoundError(f"BPE merges path not found: {self.spec.bpe_merges_path}")
        return load_bpe_tokenizer(
            vocab_path=self.spec.bpe_vocab_path,
            merges_path=self.spec.bpe_merges_path,
            special_tokens=list(self.spec.special_tokens),
        )

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return

        checkpoint_obj = torch.load(self.spec.checkpoint, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint_obj)
        normalized_state_dict = _normalize_state_dict_keys(state_dict)

        args = SimpleNamespace(
            config_json=self.spec.config_json,
            vocab_size=None,
            context_length=None,
            d_model=None,
            num_heads=None,
            d_ff=None,
            num_layers=None,
            theta=None,
        )
        model_config = _resolve_model_config(args, checkpoint_obj)

        model = TransformerLM(
            vocab_size=model_config.vocab_size,
            context_length=model_config.context_length,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            d_ff=model_config.d_ff,
            num_layers=model_config.num_layers,
            theta=model_config.theta,
        ).to(self.spec.device)

        missing_keys, unexpected_keys = model.load_state_dict(
            normalized_state_dict,
            strict=self.spec.strict_load,
        )

        tokenizer = self._build_tokenizer()
        eos_token_id = _infer_eos_token_id(tokenizer, self.spec.eos_token_id)

        self._model = model
        self._tokenizer = tokenizer
        self._eos_token_id = eos_token_id
        self._missing_keys = list(missing_keys)
        self._unexpected_keys = list(unexpected_keys)
        self._model_config = model_config

    def generate_text(
        self,
        prompt: str | list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        completion_only: bool,
    ) -> GenerationResult:
        self.ensure_loaded()
        assert self._model is not None

        prompt_token_ids = _encode_prompt(prompt, self._tokenizer)
        generated_token_ids = generate(
            model=self._model,
            prompt=prompt,
            tokenizer=self._tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self._eos_token_id,
            device=self.spec.device,
        )
        completion_token_ids = generated_token_ids[len(prompt_token_ids) :]
        output_token_ids = completion_token_ids if completion_only else generated_token_ids

        if self._tokenizer is None:
            text = json.dumps(output_token_ids)
        else:
            text = self._tokenizer.decode(output_token_ids)

        return GenerationResult(
            text=text,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            completion_token_ids=completion_token_ids,
        )

    def describe(self) -> dict[str, Any]:
        self.ensure_loaded()
        assert self._model_config is not None

        return {
            "checkpoint": str(self.spec.checkpoint),
            "config_json": str(self.spec.config_json) if self.spec.config_json is not None else None,
            "device": self.spec.device,
            "tokenizer_mode": self.spec.tokenizer_mode,
            "bpe_vocab_path": str(self.spec.bpe_vocab_path),
            "bpe_merges_path": str(self.spec.bpe_merges_path),
            "special_tokens": list(self.spec.special_tokens),
            "eos_token_id": self._eos_token_id,
            "model_config": {
                "vocab_size": int(self._model_config.vocab_size),
                "context_length": int(self._model_config.context_length),
                "d_model": int(self._model_config.d_model),
                "num_heads": int(self._model_config.num_heads),
                "d_ff": int(self._model_config.d_ff),
                "num_layers": int(self._model_config.num_layers),
                "theta": float(self._model_config.theta),
            },
            "missing_keys": list(self._missing_keys),
            "unexpected_keys": list(self._unexpected_keys),
        }
