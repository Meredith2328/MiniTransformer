from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from cs336_basics.bpe import Tokenizer


DTYPE_MAP = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize txt corpus into contiguous token-id .bin")
    parser.add_argument("--input-text", type=str, required=True, help="Input plain-text corpus path.")
    parser.add_argument("--vocab-pkl", type=str, required=True, help="Pickle path of vocab dict[int, bytes].")
    parser.add_argument("--merges-pkl", type=str, required=True, help="Pickle path of merges list[tuple[bytes, bytes]].")
    parser.add_argument("--output-bin", type=str, required=True, help="Output token id binary path.")
    parser.add_argument("--output-meta", type=str, default="", help="Optional JSON metadata path.")
    parser.add_argument(
        "--special-tokens",
        type=str,
        default="<|endoftext|>",
        help="Comma-separated special tokens used when building tokenizer runtime.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "uint16", "uint32", "int32", "int64"],
        default="auto",
        help="Token dtype stored in output binary.",
    )
    parser.add_argument(
        "--progress-every-lines",
        type=int,
        default=10000,
        help="Progress print interval by input lines during both passes.",
    )
    return parser.parse_args()


def parse_special_tokens(raw: str) -> list[str]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return tokens or ["<|endoftext|>"]


def choose_dtype(dtype_arg: str, tokenizer: Tokenizer) -> np.dtype:
    if dtype_arg != "auto":
        return np.dtype(DTYPE_MAP[dtype_arg])
    max_token_id = max(tokenizer.vocab.keys()) if tokenizer.vocab else 0
    if max_token_id <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def load_tokenizer(vocab_pkl: Path, merges_pkl: Path, special_tokens: list[str]) -> Tokenizer:
    with vocab_pkl.open("rb") as f:
        vocab = pickle.load(f)
    with merges_pkl.open("rb") as f:
        merges = pickle.load(f)
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def count_tokens(
    input_text: Path,
    tokenizer: Tokenizer,
    progress_every_lines: int,
) -> tuple[int, int]:
    total_tokens = 0
    total_lines = 0
    start = time.perf_counter()
    with input_text.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            total_lines = line_idx
            total_tokens += len(tokenizer.encode(line))
            if line_idx % progress_every_lines == 0:
                elapsed = max(time.perf_counter() - start, 1e-8)
                print(
                    f"[count] lines={line_idx:,} tokens={total_tokens:,} "
                    f"rate={total_tokens / elapsed:,.0f} tok/s"
                )
    return total_lines, total_tokens


def write_tokens(
    input_text: Path,
    tokenizer: Tokenizer,
    output_bin: Path,
    dtype: np.dtype,
    total_tokens: int,
    progress_every_lines: int,
) -> None:
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.memmap(output_bin, mode="w+", dtype=dtype, shape=(total_tokens,))
    cursor = 0
    start = time.perf_counter()
    with input_text.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            ids = tokenizer.encode(line)
            n = len(ids)
            if n > 0:
                mmap[cursor : cursor + n] = np.asarray(ids, dtype=dtype)
                cursor += n
            if line_idx % progress_every_lines == 0:
                elapsed = max(time.perf_counter() - start, 1e-8)
                print(
                    f"[write] lines={line_idx:,} tokens_written={cursor:,}/{total_tokens:,} "
                    f"rate={cursor / elapsed:,.0f} tok/s"
                )
    mmap.flush()
    if cursor != total_tokens:
        raise RuntimeError(f"Token count mismatch: expected {total_tokens}, wrote {cursor}.")


def main() -> None:
    args = parse_args()
    input_text = Path(args.input_text)
    vocab_pkl = Path(args.vocab_pkl)
    merges_pkl = Path(args.merges_pkl)
    output_bin = Path(args.output_bin)
    output_meta = Path(args.output_meta) if args.output_meta else Path(str(output_bin) + ".meta.json")

    if not input_text.exists():
        raise FileNotFoundError(f"Input text not found: {input_text}")
    if not vocab_pkl.exists():
        raise FileNotFoundError(f"vocab pkl not found: {vocab_pkl}")
    if not merges_pkl.exists():
        raise FileNotFoundError(f"merges pkl not found: {merges_pkl}")
    if args.progress_every_lines <= 0:
        raise ValueError("--progress-every-lines must be positive.")

    special_tokens = parse_special_tokens(args.special_tokens)
    print("Tokenization config:")
    print(f"  input_text: {input_text}")
    print(f"  vocab_pkl: {vocab_pkl}")
    print(f"  merges_pkl: {merges_pkl}")
    print(f"  output_bin: {output_bin}")
    print(f"  output_meta: {output_meta}")
    print(f"  special_tokens: {special_tokens}")

    load_start = time.perf_counter()
    tokenizer = load_tokenizer(vocab_pkl, merges_pkl, special_tokens)
    dtype = choose_dtype(args.dtype, tokenizer)
    print(f"Loaded tokenizer in {time.perf_counter() - load_start:.2f}s, selected dtype={dtype}")

    count_start = time.perf_counter()
    total_lines, total_tokens = count_tokens(input_text, tokenizer, args.progress_every_lines)
    count_elapsed = time.perf_counter() - count_start
    print(f"Count pass done: lines={total_lines:,}, tokens={total_tokens:,}, elapsed={count_elapsed:.2f}s")

    write_start = time.perf_counter()
    write_tokens(
        input_text=input_text,
        tokenizer=tokenizer,
        output_bin=output_bin,
        dtype=dtype,
        total_tokens=total_tokens,
        progress_every_lines=args.progress_every_lines,
    )
    write_elapsed = time.perf_counter() - write_start
    print(f"Write pass done: output={output_bin}, elapsed={write_elapsed:.2f}s")

    meta = {
        "input_text": str(input_text),
        "output_bin": str(output_bin),
        "dtype": str(dtype),
        "num_lines": total_lines,
        "num_tokens": int(total_tokens),
        "vocab_size": int(len(tokenizer.vocab)),
        "special_tokens": special_tokens,
        "created_at_unix": time.time(),
        "count_pass_seconds": count_elapsed,
        "write_pass_seconds": write_elapsed,
    }
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    with output_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Metadata written: {output_meta}")


if __name__ == "__main__":
    main()
