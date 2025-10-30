# get_tag_contents.py
# Robust tag extraction from Qwen (or any LLM) generations.
# We look for <answer>...</answer>, <confidence>...</confidence>, <analysis>...</analysis>,
# and include fallbacks for minor formatting drift.

from __future__ import annotations
import re
from typing import Optional, Tuple

_ANSWER_TAG = re.compile(r"<\s*answer\s*>(?P<ans>.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)
_CONF_TAG   = re.compile(r"<\s*confidence\s*>(?P<conf>.*?)<\s*/\s*confidence\s*>", re.IGNORECASE | re.DOTALL)
_ANAL_TAG   = re.compile(r"<\s*analysis\s*>(?P<ana>.*?)<\s*/\s*analysis\s*>", re.IGNORECASE | re.DOTALL)

# Common letter tokens we’ll accept for MCQ
_VALID_LETTERS = {"A","B","C","D"}

def _clean(s: str) -> str:
    return s.strip().replace("\u200b", "").strip()

def _first_match(regex: re.Pattern, text: str) -> Optional[str]:
    m = regex.search(text)
    if not m:
        return None
    return _clean(m.group(1))

def parse_answer(raw: str) -> Optional[str]:
    """
    Returns 'A'|'B'|'C'|'D' if found inside <answer>...</answer>, else tries gentle fallbacks.
    """
    # Primary: tagged
    tagged = _first_match(_ANSWER_TAG, raw)
    if tagged:
        # Normalize: keep only first A/B/C/D if present
        tagged_up = tagged.upper()
        # direct single letter?
        if tagged_up in _VALID_LETTERS:
            return tagged_up
        # look for first letter occurrence
        m = re.search(r"\b([ABCD])\b", tagged_up)
        if m:
            return m.group(1)

    # Fallback 1: explicit “Answer: X”
    m = re.search(r"(?i)\banswer\s*:\s*([ABCD])\b", raw)
    if m:
        return m.group(1).upper()

    # Fallback 2: common patterns like “Final answer is C”
    m = re.search(r"(?i)\bfinal\s+answer(?:\s+is)?\s*([ABCD])\b", raw)
    if m:
        return m.group(1).upper()

    return None

def parse_confidence(raw: str) -> Optional[float]:
    """
    Extracts a float in [0,1] from <confidence>...</confidence>.
    Accepts '0.87', '87%', or '0,87' (European comma) and clamps to [0,1].
    """
    tagged = _first_match(_CONF_TAG, raw)
    if tagged is None:
        return None

    s = tagged.strip()
    # handle percent
    pct = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", s)
    if pct:
        v = float(pct.group(1)) / 100.0
        return float(min(max(v, 0.0), 1.0))

    # handle decimal with comma
    s = s.replace(",", ".")
    try:
        v = float(s)
        # If they gave something like 87, interpret as 0.87 only if >1 and <=100
        if v > 1.0 and v <= 100.0:
            v = v / 100.0
        return float(min(max(v, 0.0), 1.0))
    except Exception:
        return None

def parse_analysis(raw: str) -> Optional[str]:
    return _first_match(_ANAL_TAG, raw)

def extract_all(raw: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Convenience: returns (answer_letter, confidence_float, analysis_text)
    """
    return parse_answer(raw), parse_confidence(raw), parse_analysis(raw)
