from __future__ import annotations

import argparse
import pickle
import threading
import time
from pathlib import Path
from typing import Any

from tests.adapters import run_train_bpe


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "tokenizer"
DEFAULT_INPUT_PATH = DEFAULT_DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
DEFAULT_VOCAB_PATH = DEFAULT_TOKENIZER_DIR / "tinystories_bpe_vocab.pkl"
DEFAULT_MERGES_PATH = DEFAULT_TOKENIZER_DIR / "tinystories_bpe_merges.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train byte-level BPE tokenizer with progress output.")
    parser.add_argument("--input-path", type=str, default=str(DEFAULT_INPUT_PATH), help="Training corpus path.")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Target tokenizer vocabulary size.")
    parser.add_argument(
        "--special-tokens",
        type=str,
        default="<|endoftext|>",
        help="Comma-separated special tokens.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print merge progress every N merges.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=15,
        help="Print heartbeat when there is no merge progress update.",
    )
    parser.add_argument("--tokenizer-dir", type=str, default=str(DEFAULT_TOKENIZER_DIR), help="Output directory.")
    parser.add_argument("--vocab-out", type=str, default=str(DEFAULT_VOCAB_PATH), help="Vocab output path.")
    parser.add_argument("--merges-out", type=str, default=str(DEFAULT_MERGES_PATH), help="Merges output path.")
    return parser.parse_args()


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - (minutes * 60)
    return f"{minutes}m{rem:04.1f}s"


def parse_special_tokens(raw_value: str) -> list[str]:
    tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    return tokens or ["<|endoftext|>"]


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    tokenizer_dir = Path(args.tokenizer_dir)
    vocab_path = Path(args.vocab_out)
    merges_path = Path(args.merges_out)
    special_tokens = parse_special_tokens(args.special_tokens)

    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive.")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")
    if args.heartbeat_seconds <= 0:
        raise ValueError("--heartbeat-seconds must be positive.")

    print("BPE training configuration:")
    print(f"  input_path: {input_path}")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  special_tokens: {special_tokens}")
    print(f"  progress_every: {args.progress_every}")
    print(f"  heartbeat_seconds: {args.heartbeat_seconds}")
    print(f"  vocab_out: {vocab_path}")
    print(f"  merges_out: {merges_path}")

    state: dict[str, Any] = {
        "stage": "init",
        "completed_merges": 0,
        "target_merges": None,
        "last_message_time": time.perf_counter(),
    }
    start_time = time.perf_counter()
    stop_heartbeat = threading.Event()

    def on_progress(event: dict[str, Any]) -> None:
        stage = event.get("stage", "unknown")
        event_name = event.get("event", "unknown")
        state["stage"] = stage
        state["last_message_time"] = time.perf_counter()

        if stage == "learn_merges" and event_name == "start":
            state["target_merges"] = int(event.get("target_merges", 0))
            print(f"[progress] merge stage started. target_merges={state['target_merges']}")
            return

        if stage == "learn_merges" and event_name == "progress":
            completed = int(event.get("completed_merges", 0))
            target = int(event.get("target_merges", 0))
            state["completed_merges"] = completed
            state["target_merges"] = target
            progress = float(event.get("progress", 0.0)) * 100.0
            rate = float(event.get("merge_rate_per_sec", 0.0))
            eta = format_seconds(event.get("eta_seconds"))
            print(
                "[progress] merges "
                f"{completed}/{target} ({progress:.1f}%) | "
                f"rate {rate:.2f}/s | eta {eta}"
            )
            return

        if stage == "learn_merges" and event_name == "end":
            print(
                "[progress] merge stage finished. "
                f"completed={event.get('completed_merges', 0)} "
                f"in {format_seconds(float(event.get('seconds', 0.0)))}"
            )
            return

        if event_name == "end":
            elapsed = format_seconds(float(event.get("seconds", 0.0)))
            if stage == "read_corpus":
                print(f"[progress] read_corpus done in {elapsed}, chars={event.get('num_chars', 0)}")
            elif stage == "pretokenize":
                print(
                    f"[progress] pretokenize done in {elapsed}, "
                    f"unique_words={event.get('num_unique_words', 0)}, "
                    f"word_instances={event.get('num_word_instances', 0)}"
                )
            elif stage == "init_vocab":
                print(f"[progress] init_vocab done, size={event.get('initial_vocab_size', 0)}")
            elif stage == "train":
                print(
                    f"[progress] train done in {elapsed}, "
                    f"merges={event.get('num_merges', 0)}, "
                    f"final_vocab={event.get('final_vocab_size', 0)}"
                )

    def heartbeat() -> None:
        while not stop_heartbeat.wait(args.heartbeat_seconds):
            elapsed_total = time.perf_counter() - start_time
            stage = state.get("stage", "unknown")
            target = state.get("target_merges")
            completed = int(state.get("completed_merges", 0))
            if stage == "learn_merges" and isinstance(target, int) and target > 0:
                pct = 100.0 * completed / target
                print(
                    "[heartbeat] "
                    f"elapsed={format_seconds(elapsed_total)} | "
                    f"stage={stage} | merges={completed}/{target} ({pct:.1f}%)"
                )
            else:
                print(f"[heartbeat] elapsed={format_seconds(elapsed_total)} | stage={stage}")

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        vocab, merges = run_train_bpe(
            input_path=str(input_path),
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            progress_callback=on_progress,
            progress_every=args.progress_every,
        )
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1.0)

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    merges_path.parent.mkdir(parents=True, exist_ok=True)

    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f)
    with merges_path.open("wb") as f:
        pickle.dump(merges, f)

    longest_token = max(vocab.values(), key=len)
    elapsed = time.perf_counter() - start_time
    print("Saved tokenizer artifacts:")
    print(f"  vocab_path: {vocab_path}")
    print(f"  merges_path: {merges_path}")
    print(f"  longest_token: {longest_token!r}")
    print(f"  longest_token_len: {len(longest_token)}")
    print(f"  total_elapsed: {format_seconds(elapsed)}")


if __name__ == "__main__":
    main()
