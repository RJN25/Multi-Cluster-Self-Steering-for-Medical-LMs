"""Preprocess the MedQA dataset into the parquet format expected by RENT."""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

import datasets

RENT_ROOT = Path(__file__).resolve().parents[2]
if str(RENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RENT_ROOT))

PROJECT_ROOT = RENT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from verl.utils.hdfs_io import copy, makedirs
except ModuleNotFoundError:
    def makedirs(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def copy(src: str, dst: str) -> None:
        shutil.copytree(src, dst, dirs_exist_ok=True)

from medqa_steering.system_prompt import get_system_prompt

LETTER = ("A", "B", "C", "D")
SYSTEM_PROMPT = get_system_prompt().strip()
USER_TEMPLATE = (
    "You are a clinical reasoning assistant.\n"
    "Question:\n{stem}\n\n"
    "Options:\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n"
    "Answer with just the letter (A/B/C/D)."
)


def _format_user_prompt(stem: str, choices: Iterable[str]) -> str:
    a, b, c, d = list(choices)
    return USER_TEMPLATE.format(stem=stem, a=a, b=b, c=c, d=d)


def _build_record(example: dict, *, split: str, index: int) -> dict:
    stem = example["sent1"]
    choices: List[str] = [example[f"ending{i}"] for i in range(len(LETTER))]
    label = int(example["label"])

    return {
        "data_source": "medqa",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_user_prompt(stem, choices)},
        ],
        "ability": "medical-reasoning",
        "reward_model": {"style": "rule", "ground_truth": LETTER[label]},
        "extra_info": {
            "split": split,
            "index": index,
            "qid": example.get("id", index),
            "choices": choices,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MedQA preprocessing for RENT")
    parser.add_argument("--local_dir", default="~/data/medqa", help="Where to save the parquet files")
    parser.add_argument("--hdfs_dir", default=None, help="Optional remote directory for copying the parquet files")
    parser.add_argument("--hf_dataset", default="GBaker/MedQA-USMLE-4-options-hf", help="Hugging Face dataset identifier")
    parser.add_argument("--hf_split", default="train", help="Dataset split to load from Hugging Face")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of examples reserved for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/validation split")

    args = parser.parse_args()

    dataset = datasets.load_dataset(args.hf_dataset, split=args.hf_split)
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val_fraction must be between 0 and 1 (exclusive)")

    split_ds = dataset.train_test_split(test_size=args.val_fraction, seed=args.seed)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]

    remove_columns = dataset.column_names
    train_processed = train_ds.map(
        lambda ex, idx: _build_record(ex, split="train", index=idx),
        with_indices=True,
        remove_columns=remove_columns,
    )
    val_processed = val_ds.map(
        lambda ex, idx: _build_record(ex, split="validation", index=idx),
        with_indices=True,
        remove_columns=remove_columns,
    )

    local_dir = Path(os.path.expanduser(args.local_dir)).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)

    train_path = local_dir / "train.parquet"
    val_path = local_dir / "val.parquet"
    train_processed.to_parquet(train_path.as_posix())
    val_processed.to_parquet(val_path.as_posix())

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=str(local_dir), dst=args.hdfs_dir)

    print(f"Saved {len(train_processed)} training examples to {train_path}")
    print(f"Saved {len(val_processed)} validation examples to {val_path}")


if __name__ == "__main__":
    main()