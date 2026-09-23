"""
Download openlifescienceai/medmcqa, filter rows, and save to local disk.

Filter (applied to every split):
  - choice_type == "single"
  - non-empty exp with > 50 words
  - valid correct option (cop → A/B/C/D)
"""

import os
from typing import Any, Dict, Optional

from datasets import DatasetDict, load_dataset

MIN_EXP_WORDS = 50
LETTER_BY_INDEX = ("A", "B", "C", "D")
COP_TO_LETTER = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    "0": "A",
    "1": "B",
    "2": "C",
    "3": "D",
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
}

# Hugging Face download cache. Leave unset to use the default HF cache.
cache_dir = os.environ.get("HF_HOME") or None

# On-disk DatasetDict (load later with datasets.load_from_disk).
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openlifescienceai_medmcqa")


def word_count(text: str) -> int:
    return len((text or "").split())


def cop_to_letter(cop: Any) -> Optional[str]:
    """Map MedMCQA cop (int ClassLabel 0–3 or letter) to A/B/C/D."""
    if cop is None:
        return None
    if isinstance(cop, str):
        key: Any = cop.strip()
        if not key:
            return None
        # Some dumps use 1-indexed strings "1".."4".
        if key in {"1", "2", "3", "4"}:
            return LETTER_BY_INDEX[int(key) - 1]
        return COP_TO_LETTER.get(key) or COP_TO_LETTER.get(key.lower())
    try:
        idx = int(cop)
    except (TypeError, ValueError):
        return None
    # HF ClassLabel is 0–3; raw MedMCQA JSON is sometimes 1–4.
    if idx in COP_TO_LETTER:
        return COP_TO_LETTER[idx]
    if 1 <= idx <= 4:
        return LETTER_BY_INDEX[idx - 1]
    return None


def passes_filter(row: Dict[str, Any]) -> bool:
    if str(row.get("choice_type") or "").strip().lower() != "single":
        return False
    exp = row.get("exp")
    if exp is None:
        return False
    exp_s = str(exp).strip()
    if not exp_s or word_count(exp_s) <= MIN_EXP_WORDS:
        return False
    return cop_to_letter(row.get("cop")) is not None


marker = os.path.join(save_path, "dataset_dict.json")
if not os.path.isfile(marker):
    raw = load_dataset("openlifescienceai/medmcqa", cache_dir=cache_dir)
    filtered = DatasetDict(
        {
            split: ds.filter(passes_filter)
            for split, ds in raw.items()
        }
    )
    for split, ds in filtered.items():
        print(f"{split}: {len(raw[split])} → {len(ds)} rows after filter")
    filtered.save_to_disk(save_path)

print(f"Data saved to {save_path}")
