"""Utilities for preparing and launching RENT baselines on MedQA."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_RENT_ROOT = Path(__file__).resolve().parent / "rent-rl-main"
_PREPROCESS_SCRIPT = _RENT_ROOT / "examples" / "data_preprocess" / "medqa.py"
_DEFAULT_EXPS = "[grpo,entropy,format,sampleval,medqa]"


def _run_command(command: Sequence[str], *, cwd: Path | None = None) -> None:
    process = subprocess.run(command, cwd=cwd, check=False)
    if process.returncode != 0:
        raise SystemExit(process.returncode)


def _cmd_prepare(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(_PREPROCESS_SCRIPT),
        "--local_dir",
        os.path.expanduser(args.local_dir),
        "--hf_dataset",
        args.hf_dataset,
        "--hf_split",
        args.hf_split,
        "--val_fraction",
        str(args.val_fraction),
        "--seed",
        str(args.seed),
    ]
    if args.hdfs_dir is not None:
        command.extend(["--hdfs_dir", args.hdfs_dir])
    _run_command(command, cwd=_RENT_ROOT)


def _cmd_train(args: argparse.Namespace) -> None:
    exps = args.exps or _DEFAULT_EXPS
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"exps={exps}",
        f"base_model={args.base_model}",
    ]
    if args.ngpus is not None:
        command.append(f"ngpus={args.ngpus}")
    if args.override:
        command.extend(args.override)
    _run_command(command, cwd=_RENT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MedQA RENT baseline helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Download and format MedQA for RENT")
    prep.add_argument("--local_dir", default="~/data/medqa", help="Local directory for parquet outputs")
    prep.add_argument("--hdfs_dir", default=None, help="Optional remote directory for copying parquet files")
    prep.add_argument("--hf_dataset", default="GBaker/MedQA-USMLE-4-options-hf", help="Hugging Face dataset identifier")
    prep.add_argument("--hf_split", default="train", help="Dataset split to load from Hugging Face")
    prep.add_argument("--val_fraction", type=float, default=0.1, help="Fraction reserved for validation")
    prep.add_argument("--seed", type=int, default=42, help="Random seed for the train/validation split")
    prep.set_defaults(func=_cmd_prepare)

    train = subparsers.add_parser("train", help="Launch RENT training using the MedQA config")
    train.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct", help="Base policy checkpoint")
    train.add_argument("--exps", default=_DEFAULT_EXPS, help="Comma-separated Hydra experiment list (default: %(default)s)")
    train.add_argument("--ngpus", type=int, default=None, help="Override the number of GPUs hydra should request")
    train.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional hydra overrides (repeatable, e.g. --override data.train_batch_size=64)",
    )
    train.set_defaults(func=_cmd_train)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()