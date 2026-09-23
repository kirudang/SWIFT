"""
Shared GSM8K few-shot prompt construction for No-WM and SWIFT Generation.

Keep this prompt identical across Generation_no_wm.py and Generation.py.

Few-shot demos are the first 5 examples from openai/gsm8k main train
(indices 0–4), converted to the shared "Answer: X" response format.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "test"
GSM8K_N_SHOT = 5
# Arrow DatasetDict written under the HF cache root (train + test).
GSM8K_DISK_SUBDIR = "openai_gsm8k"

# First 5 rows of openai/gsm8k main train (question, answer with #### final).
_GSM8K_TRAIN_FEWSHOT_RAW: Tuple[Tuple[str, str], ...] = (
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5",
    ),
    (
        "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "Maila read 12 x 2 = <<12*2=24>>24 pages today.\nSo she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.\nThere are 120 - 36 = <<120-36=84>>84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.\n#### 42",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?",
        "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624",
    ),
)

_CALC_ANNOTATION_RE = re.compile(r"<<.*?>>")
_FINAL_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_ANSWER_COLON_RE = re.compile(
    r"(?i)\bAnswer\s*:\s*(.+?)(?:\n|$)",
)


def gsm8k_disk_path(cache_dir: str) -> str:
    """Absolute path for the on-disk GSM8K DatasetDict under cache_dir."""
    return os.path.join(os.path.abspath(cache_dir), GSM8K_DISK_SUBDIR)


def extract_gsm8k_final_answer(answer: str) -> str:
    """Extract the ground-truth final answer from a GSM8K #### line."""
    text = (answer or "").strip()
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def extract_output_answer(text: str) -> str:
    """
    Extract the model's final answer from a generation string.

    Prefers the last "Answer: X" match (few-shot format), then a #### line,
    otherwise returns an empty string.
    """
    s = (text or "").strip()
    if not s:
        return ""
    colon_matches = list(_ANSWER_COLON_RE.finditer(s))
    if colon_matches:
        return colon_matches[-1].group(1).strip()
    hash_match = _FINAL_ANSWER_RE.search(s)
    if hash_match:
        return hash_match.group(1).strip()
    return ""


def format_gsm8k_train_response(answer: str) -> str:
    """Convert a GSM8K train answer (#### final) into 'Answer: X' reasoning text."""
    text = (answer or "").strip()
    text = _CALC_ANNOTATION_RE.sub("", text)
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        final = match.group(1).strip()
        body = _FINAL_ANSWER_RE.sub("", text).strip()
        if body:
            return f"{body}\nAnswer: {final}"
        return f"Answer: {final}"
    return text


def _build_fewshot_block() -> str:
    parts: List[str] = []
    for i, (question, answer) in enumerate(_GSM8K_TRAIN_FEWSHOT_RAW[:GSM8K_N_SHOT], start=1):
        response = format_gsm8k_train_response(answer)
        parts.append(
            f"Example {i}:\n"
            f"Question: {question.strip()}\n"
            f"Response: {response}"
        )
    return "\n\n".join(parts)


GSM8K_FEWSHOT_EXAMPLES = _build_fewshot_block()

# 5-shot demos from GSM8K train; establishes concise reasoning + "Answer: X".
GSM8K_FEWSHOT_PROMPT_TEMPLATE = f"""Solve the math problem using a short reasoning process.
Keep the reasoning concise and give the final answer as "Answer: X".

{GSM8K_FEWSHOT_EXAMPLES}

Now solve:

Question: {{question}}
Response:"""


def build_gsm8k_fewshot_prompt(question: str) -> str:
    """Build the shared 5-shot generation prompt for one GSM8K question."""
    q = (question or "").strip()
    return GSM8K_FEWSHOT_PROMPT_TEMPLATE.format(question=q)


def ensure_gsm8k_on_disk(
    cache_dir: str,
    *,
    dataset_id: str = GSM8K_DATASET_ID,
    force_redownload: bool = False,
) -> str:
    """
    Ensure openai/gsm8k (main) is saved under cache_dir/openai_gsm8k.

    Returns the on-disk path. Reuses an existing save unless force_redownload.
    """
    from datasets import load_dataset, load_from_disk

    disk_path = gsm8k_disk_path(cache_dir)
    marker = os.path.join(disk_path, "dataset_dict.json")
    if (not force_redownload) and os.path.isfile(marker):
        # Validate it loads.
        load_from_disk(disk_path)
        return disk_path

    os.makedirs(cache_dir, exist_ok=True)
    ds = load_dataset(
        dataset_id,
        GSM8K_CONFIG,
        cache_dir=cache_dir,
    )
    ds.save_to_disk(disk_path)
    return disk_path


def load_gsm8k_examples(
    cache_dir: str,
    *,
    n_inputs: Optional[int] = None,
    split: str = GSM8K_SPLIT,
    dataset_id: str = GSM8K_DATASET_ID,
) -> List[Dict[str, Any]]:
    """
    Load GSM8K from disk under cache_dir (download + save_to_disk on first use).

    Each item: {"question": str, "answer": str, "final_answer": str, "prompt": str}
    - answer: full GSM8K ground-truth solution string
    - final_answer: extracted ground-truth value (from ####)
    Uses the same ordered split examples for No-WM and SWIFT runs.
    Prompts use a fixed 5-shot prefix from GSM8K train[0:5].
    """
    from datasets import load_from_disk

    disk_path = ensure_gsm8k_on_disk(cache_dir, dataset_id=dataset_id)
    dsd = load_from_disk(disk_path)
    if split not in dsd:
        raise ValueError(
            f"GSM8K split {split!r} not found on disk at {disk_path}. "
            f"Available: {list(dsd.keys())}"
        )
    ds = dsd[split]
    n = len(ds)
    if n_inputs is not None and int(n_inputs) > 0:
        n = min(n, int(n_inputs))

    examples: List[Dict[str, Any]] = []
    for i in range(n):
        question = str(ds[i]["question"]).strip()
        gold = str(ds[i]["answer"]).strip()
        examples.append(
            {
                "question": question,
                "answer": gold,
                "final_answer": extract_gsm8k_final_answer(gold),
                "prompt": build_gsm8k_fewshot_prompt(question),
            }
        )
    return examples


def render_chat_prompt(tokenizer: Any, user_text: str) -> str:
    """Apply the model chat template around the shared few-shot user text."""
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for tokenizers without a chat template.
        return user_text
