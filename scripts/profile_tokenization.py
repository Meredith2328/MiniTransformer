from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tokenize_to_bin as tokbin


DEFAULT_INPUT_TEXT = Path("data/TinyStoriesV2-GPT4-train.txt")
DEFAULT_VOCAB_PKL = Path("tokenizer/tinystories_bpe_vocab.pkl")
DEFAULT_MERGES_PKL = Path("tokenizer/tinystories_bpe_merges.pkl")
DEFAULT_OUTPUT_DIR = Path("runs/profiles/tokenization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile tokenization workloads with cProfile.")
    parser.add_argument(
        "--stage",
        choices=["encode-sample", "count-pass", "write-pass", "full-pipeline"],
        default="encode-sample",
        help="Which tokenization workload to profile.",
    )
    parser.add_argument("--input-text", type=str, default=str(DEFAULT_INPUT_TEXT), help="Input text corpus path.")
    parser.add_argument("--vocab-pkl", type=str, default=str(DEFAULT_VOCAB_PKL), help="Tokenizer vocab pickle.")
    parser.add_argument("--merges-pkl", type=str, default=str(DEFAULT_MERGES_PKL), help="Tokenizer merges pickle.")
    parser.add_argument(
        "--special-tokens",
        type=str,
        default="<|endoftext|>",
        help="Comma-separated special tokens used by the tokenizer runtime.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "uint16", "uint32", "int32", "int64"],
        default="auto",
        help="Token dtype used when the workload writes a temporary .bin file.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=2000,
        help="Maximum number of input lines to load into memory for profiling.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Repeat factor for encode-sample stage to amplify hot paths.",
    )
    parser.add_argument(
        "--output-prof",
        type=str,
        default="",
        help="Optional explicit output .prof path. Defaults to runs/profiles/tokenization/...",
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=["cumulative", "cumtime", "tottime", "time", "ncalls", "calls"],
        default="cumulative",
        help="Sort key for the on-screen stats preview.",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Number of rows to print in the on-screen preview.")
    return parser.parse_args()


def load_lines(input_text: Path, max_lines: int) -> list[str]:
    if max_lines <= 0:
        raise ValueError("--max-lines must be positive.")

    lines: list[str] = []
    with input_text.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            lines.append(line)
            if line_idx >= max_lines:
                break
    return lines


def count_tokens_in_memory(lines: list[str], tokenizer: tokbin.Tokenizer) -> int:
    total_tokens = 0
    for line in lines:
        total_tokens += len(tokenizer.encode(line))
    return total_tokens


def encode_sample(lines: list[str], tokenizer: tokbin.Tokenizer, repeat: int) -> dict[str, Any]:
    total_tokens = 0
    total_calls = 0
    for _ in range(repeat):
        for line in lines:
            total_tokens += len(tokenizer.encode(line))
            total_calls += 1
    return {
        "total_tokens": total_tokens,
        "encode_calls": total_calls,
        "repeat": repeat,
    }


def preencode_lines(lines: list[str], tokenizer: tokbin.Tokenizer) -> tuple[list[list[int]], int]:
    encoded_lines: list[list[int]] = []
    total_tokens = 0
    for line in lines:
        ids = tokenizer.encode(line)
        encoded_lines.append(ids)
        total_tokens += len(ids)
    return encoded_lines, total_tokens


def write_preencoded_tokens(
    encoded_lines: list[list[int]],
    output_bin: Path,
    dtype: np.dtype,
    total_tokens: int,
) -> dict[str, Any]:
    mmap = np.memmap(output_bin, mode="w+", dtype=dtype, shape=(total_tokens,))
    cursor = 0
    for ids in encoded_lines:
        n = len(ids)
        if n == 0:
            continue
        mmap[cursor : cursor + n] = np.asarray(ids, dtype=dtype)
        cursor += n
    mmap.flush()
    return {
        "tokens_written": cursor,
        "output_bin": str(output_bin),
    }


def full_pipeline(lines: list[str], tokenizer: tokbin.Tokenizer, output_bin: Path, dtype: np.dtype) -> dict[str, Any]:
    total_tokens = count_tokens_in_memory(lines, tokenizer)
    result = write_tokens_in_memory(lines, tokenizer, output_bin, dtype, total_tokens)
    result["counted_tokens"] = total_tokens
    return result


def write_tokens_in_memory(
    lines: list[str],
    tokenizer: tokbin.Tokenizer,
    output_bin: Path,
    dtype: np.dtype,
    total_tokens: int,
) -> dict[str, Any]:
    mmap = np.memmap(output_bin, mode="w+", dtype=dtype, shape=(total_tokens,))
    cursor = 0
    for line in lines:
        ids = tokenizer.encode(line)
        n = len(ids)
        if n == 0:
            continue
        mmap[cursor : cursor + n] = np.asarray(ids, dtype=dtype)
        cursor += n
    mmap.flush()
    return {
        "tokens_written": cursor,
        "output_bin": str(output_bin),
    }


def run_profile(
    stage: str,
    lines: list[str],
    tokenizer: tokbin.Tokenizer,
    dtype: np.dtype,
    repeat: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tokenization_profile_") as tmpdir:
        temp_bin = Path(tmpdir) / "profile_output.bin"

        if stage == "encode-sample":
            return encode_sample(lines, tokenizer, repeat)
        if stage == "count-pass":
            return {"counted_tokens": count_tokens_in_memory(lines, tokenizer)}
        if stage == "write-pass":
            encoded_lines, total_tokens = preencode_lines(lines, tokenizer)
            result = write_preencoded_tokens(encoded_lines, temp_bin, dtype, total_tokens)
            result["counted_tokens"] = total_tokens
            result["preencoded_lines"] = len(encoded_lines)
            return result
        if stage == "full-pipeline":
            return full_pipeline(lines, tokenizer, temp_bin, dtype)
        raise ValueError(f"Unsupported stage: {stage}")


def build_output_prof(args: argparse.Namespace) -> Path:
    if args.output_prof:
        return Path(args.output_prof)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_stem = Path(args.input_text).stem
    return DEFAULT_OUTPUT_DIR / f"{args.stage}_{input_stem}_{timestamp}.prof"


def render_preview(profile_path: Path, sort_key: str, top_k: int) -> str:
    buffer = io.StringIO()
    stats = pstats.Stats(str(profile_path), stream=buffer)
    stats.strip_dirs().sort_stats(sort_key).print_stats(top_k)
    return buffer.getvalue().rstrip()


def main() -> None:
    args = parse_args()

    input_text = Path(args.input_text)
    vocab_pkl = Path(args.vocab_pkl)
    merges_pkl = Path(args.merges_pkl)
    output_prof = build_output_prof(args)

    if not input_text.exists():
        raise FileNotFoundError(f"Input text not found: {input_text}")
    if not vocab_pkl.exists():
        raise FileNotFoundError(f"vocab pkl not found: {vocab_pkl}")
    if not merges_pkl.exists():
        raise FileNotFoundError(f"merges pkl not found: {merges_pkl}")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    print("Profiling config:")
    print(f"  stage: {args.stage}")
    print(f"  input_text: {input_text}")
    print(f"  vocab_pkl: {vocab_pkl}")
    print(f"  merges_pkl: {merges_pkl}")
    print(f"  max_lines: {args.max_lines}")
    print(f"  repeat: {args.repeat}")

    load_start = time.perf_counter()
    special_tokens = tokbin.parse_special_tokens(args.special_tokens)
    tokenizer = tokbin.load_tokenizer(vocab_pkl, merges_pkl, special_tokens)
    dtype = tokbin.choose_dtype(args.dtype, tokenizer)
    tokenizer_load_seconds = time.perf_counter() - load_start

    lines_start = time.perf_counter()
    lines = load_lines(input_text, args.max_lines)
    load_lines_seconds = time.perf_counter() - lines_start

    print(f"Loaded tokenizer in {tokenizer_load_seconds:.3f}s, dtype={dtype}")
    print(f"Loaded {len(lines):,} lines for profiling in {load_lines_seconds:.3f}s")

    output_prof.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()

    start = time.perf_counter()
    profiler.enable()
    result = run_profile(args.stage, lines, tokenizer, dtype, args.repeat)
    profiler.disable()
    elapsed = time.perf_counter() - start

    profiler.dump_stats(str(output_prof))

    print(f"Profile saved to: {output_prof}")
    print(f"Wall time: {elapsed:.3f}s")
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("")
    print(f"Top {args.top_k} functions sorted by {args.sort}:")
    print(render_preview(output_prof, args.sort, args.top_k))


if __name__ == "__main__":
    main()
