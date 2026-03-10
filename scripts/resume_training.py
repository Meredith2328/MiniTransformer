from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


OMIT_CONFIG_KEYS = {
    "resolved_device",
    "resume",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume cs336_basics.train from a checkpoint using a saved run_config.json."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path to resume from, usually latest.pt or interrupted_step_*.pt.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to run_config.json. Defaults to <checkpoint_dir>/run_config.json.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config entries with key=value, e.g. --set total_iters=40000.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the reconstructed command without launching training.",
    )
    return parser.parse_args()


def infer_config_path(checkpoint_path: Path, config_arg: str) -> Path:
    if config_arg:
        return Path(config_arg)
    return checkpoint_path.parent / "run_config.json"


def parse_override(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid override '{raw}'. Expected key=value.")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"Invalid override '{raw}'. Key cannot be empty.")
    return key, value


def infer_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def coerce_override(raw_value: str, original_value: Any) -> Any:
    if isinstance(original_value, bool):
        lowered = raw_value.lower()
        if lowered not in {"true", "false"}:
            raise ValueError(f"Expected boolean override, got '{raw_value}'.")
        return lowered == "true"
    if isinstance(original_value, int) and not isinstance(original_value, bool):
        return int(raw_value)
    if isinstance(original_value, float):
        return float(raw_value)
    if original_value is None:
        return infer_scalar(raw_value)
    return raw_value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = dict(config)
    for raw in overrides:
        key, raw_value = parse_override(raw)
        original_value = merged.get(key)
        merged[key] = coerce_override(raw_value, original_value)
    return merged


def build_train_command(config: dict[str, Any], checkpoint_path: Path) -> list[str]:
    cmd = [sys.executable, "-m", "cs336_basics.train"]
    for key, value in config.items():
        if key in OMIT_CONFIG_KEYS:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if value is None:
            continue
        cmd.extend([flag, str(value)])
    cmd.extend(["--resume", str(checkpoint_path)])
    return cmd


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = infer_config_path(checkpoint_path, args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")

    effective_config = apply_overrides(config, args.overrides)
    command = build_train_command(effective_config, checkpoint_path)

    print("Resume configuration:")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  config:     {config_path}")
    if args.overrides:
        print(f"  overrides:  {args.overrides}")
    print("Command:")
    print("  " + " ".join(command))
    sys.stdout.flush()

    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
