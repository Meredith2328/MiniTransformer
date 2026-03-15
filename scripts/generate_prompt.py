from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from cs336_basics.inference import (
    DEFAULT_RUN_DIR,
    GenerationBackend,
    build_generation_spec,
)


def parse_prompt_ids(raw_value: str) -> list[int]:
    stripped = raw_value.strip()
    if not stripped:
        return []
    return [int(piece) for piece in stripped.replace(",", " ").split()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convenient prompt-based generation wrapper around cs336_basics.generate.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Run directory used to auto-select a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Explicit checkpoint path. Overrides --run-dir.")
    parser.add_argument("--config-json", type=Path, default=None, help="Optional model config JSON.")

    parser.add_argument("--prompt", type=str, default=None, help="Prompt text.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Read prompt text from a UTF-8 file.")
    parser.add_argument("--prompt-ids", type=str, default=None, help='Prompt token ids, e.g. "12,42,5".')
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt loop. Enabled automatically if no prompt is provided.")
    parser.add_argument("--output-file", type=Path, default=None, help="Optional output file for generated text.")

    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--completion-only", action="store_true", help="Print only the generated continuation.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--tokenizer-mode", choices=("bpe", "gpt2", "none"), default="bpe")
    parser.add_argument("--bpe-vocab-path", type=Path, default=None)
    parser.add_argument("--bpe-merges-path", type=Path, default=None)
    parser.add_argument("--special-tokens", type=str, default=None, help="Comma-separated special tokens for bpe mode.")
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--print-token-counts", action="store_true", help="Print prompt/completion token counts to stderr.")
    return parser


def resolve_prompt(args: argparse.Namespace) -> str | list[int] | None:
    if args.prompt_ids is not None:
        return parse_prompt_ids(args.prompt_ids)
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    return None


def run_single_prompt(
    backend: GenerationBackend,
    prompt: str | list[int],
    args: argparse.Namespace,
) -> str:
    start_time = time.perf_counter()
    result = backend.generate_text(
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        completion_only=args.completion_only,
    )
    elapsed = time.perf_counter() - start_time

    if args.print_token_counts:
        print(
            (
                f"[meta] prompt_tokens={len(result.prompt_token_ids)} "
                f"completion_tokens={len(result.completion_token_ids)} "
                f"elapsed={elapsed:.2f}s"
            ),
            file=sys.stderr,
        )
    return result.text


def write_output(text: str, output_file: Path | None) -> None:
    if output_file is not None:
        output_file.write_text(text, encoding="utf-8")
    print(text)


def run_interactive_loop(backend: GenerationBackend, args: argparse.Namespace) -> None:
    print("Interactive generation mode. Type /exit to quit.", file=sys.stderr)
    print(f"[model] checkpoint={backend.describe()['checkpoint']}", file=sys.stderr)
    while True:
        try:
            prompt = input("prompt> ").strip()
        except EOFError:
            print(file=sys.stderr)
            return

        if prompt in {"", "/exit", "exit", "quit"}:
            return

        output = run_single_prompt(backend, prompt, args)
        print("\n=== completion ===")
        print(output)
        print()


def main() -> None:
    parser = build_arg_parser()
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

    spec = build_generation_spec(
        checkpoint=args.checkpoint,
        run_dir=args.run_dir,
        config_json=args.config_json,
        tokenizer_mode=args.tokenizer_mode,
        bpe_vocab_path=args.bpe_vocab_path if args.bpe_vocab_path is not None else None,
        bpe_merges_path=args.bpe_merges_path if args.bpe_merges_path is not None else None,
        special_tokens=args.special_tokens,
        eos_token_id=args.eos_token_id,
        device=args.device,
        strict_load=args.strict_load,
    )
    backend = GenerationBackend(spec)

    prompt = resolve_prompt(args)
    if args.interactive or prompt is None:
        run_interactive_loop(backend, args)
        return

    output = run_single_prompt(backend, prompt, args)
    write_output(output, args.output_file)


if __name__ == "__main__":
    main()
