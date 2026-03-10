from __future__ import annotations

import argparse
import io
import pstats
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a .prof file produced by cProfile.")
    parser.add_argument("--profile", type=str, required=True, help="Path to the .prof file.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stats", "callers", "callees"],
        default="stats",
        help="Which view to print.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=["cumulative", "cumtime", "tottime", "time", "ncalls", "calls"],
        default="cumulative",
        help="Sort key for stats mode.",
    )
    parser.add_argument("--top-k", type=int, default=40, help="Maximum number of rows to print.")
    parser.add_argument(
        "--contains",
        type=str,
        default="",
        help="Optional regex-style filter passed to pstats print helpers.",
    )
    parser.add_argument(
        "--no-strip-dirs",
        action="store_true",
        help="Keep full paths in the printed profile output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_path = Path(args.profile)

    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    buffer = io.StringIO()
    stats = pstats.Stats(str(profile_path), stream=buffer)
    if not args.no_strip_dirs:
        stats.strip_dirs()
    stats.sort_stats(args.sort)

    if args.mode == "stats":
        if args.contains:
            stats.print_stats(args.contains, args.top_k)
        else:
            stats.print_stats(args.top_k)
    elif args.mode == "callers":
        if args.contains:
            stats.print_callers(args.contains)
        else:
            stats.print_callers(args.top_k)
    elif args.mode == "callees":
        if args.contains:
            stats.print_callees(args.contains)
        else:
            stats.print_callees(args.top_k)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print(buffer.getvalue().rstrip())


if __name__ == "__main__":
    main()
