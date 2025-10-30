"""Utility functions for scoring MedQA generations within RENT."""

from __future__ import annotations

import re
from typing import Optional

LETTER = ("A", "B", "C", "D")


def _normalise_letter(value: str) -> Optional[str]:
    if not value:
        return None
    value = value.strip().upper()
    if not value:
        return None
    candidate = value[0]
    if candidate in LETTER:
        return candidate
    return None


def extract_solution(solution_str: str) -> Optional[str]:
    """Extract the final option letter from a MedQA response."""

    if not solution_str:
        return None

    patterns = [
        r"<answer>\s*([A-D])\s*</answer>",
        r"<final_answer>\s*([A-D])\s*</final_answer>",
        r"answer\s*[:is]*\s*([A-D])\b",
        r"option\s*([A-D])\b",
        r"\(([A-D])\)",
        r"\b([A-D])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, solution_str, flags=re.IGNORECASE)
        if match:
            letter = _normalise_letter(match.group(1))
            if letter is not None:
                return letter

    return None


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Return ``1.0`` when the extracted option matches the ground truth letter."""

    predicted = extract_solution(solution_str)
    truth = _normalise_letter(ground_truth)
    if predicted is None or truth is None:
        return 0.0
    return float(predicted == truth)