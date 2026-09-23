
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MEDMCQA_DATASET_ID = "openlifescienceai/medmcqa"
MEDMCQA_SPLIT = "validation"
MEDMCQA_N_SHOT = 5
MEDMCQA_MIN_EXP_WORDS = 50
# Arrow DatasetDict written under the HF cache root.
MEDMCQA_DISK_SUBDIR = "openlifescienceai_medmcqa"

_OPTION_KEYS: Tuple[str, ...] = ("opa", "opb", "opc", "opd")
_LETTER_BY_INDEX = ("A", "B", "C", "D")
_COP_TO_LETTER = {
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

_ANSWER_COLON_RE = re.compile(
    r"(?i)\bAnswer\s*:\s*(.+?)(?:\n|$)",
)
_FINAL_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_LEADING_CHOICE_RE = re.compile(r"^\s*([A-Da-d])(?:\s*[.\)\:\-]\s*|\s+|$)")


def medmcqa_disk_path(cache_dir: str) -> str:
    """Absolute path for the on-disk MedMCQA DatasetDict under cache_dir."""
    return os.path.join(os.path.abspath(cache_dir), MEDMCQA_DISK_SUBDIR)


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
            return _LETTER_BY_INDEX[int(key) - 1]
        return _COP_TO_LETTER.get(key) or _COP_TO_LETTER.get(key.lower())
    try:
        idx = int(cop)
    except (TypeError, ValueError):
        return None
    # HF ClassLabel is 0–3; raw MedMCQA JSON is sometimes 1–4.
    if idx in _COP_TO_LETTER:
        return _COP_TO_LETTER[idx]
    if 1 <= idx <= 4:
        return _LETTER_BY_INDEX[idx - 1]
    return None


def normalize_choice_answer(text: str) -> str:
    """Normalize a free-form answer string to a single letter when possible."""
    s = (text or "").strip()
    if not s:
        return ""
    m = _LEADING_CHOICE_RE.match(s)
    if m:
        return m.group(1).upper()
    return s


def extract_output_answer(text: str) -> str:
    """
    Extract the model's final answer from a generation string.

    Prefers the last "Answer: X" match (few-shot format), then a #### line,
    otherwise returns an empty string. Choice answers are normalized to A–D.
    """
    s = (text or "").strip()
    if not s:
        return ""
    colon_matches = list(_ANSWER_COLON_RE.finditer(s))
    if colon_matches:
        return normalize_choice_answer(colon_matches[-1].group(1))
    hash_match = _FINAL_ANSWER_RE.search(s)
    if hash_match:
        return normalize_choice_answer(hash_match.group(1))
    return ""


def format_medmcqa_question(row: Dict[str, Any]) -> str:
    """Format stem + A/B/C/D options as the prompt question block."""
    stem = str(row.get("question") or "").strip()
    lines = [stem, ""]
    for letter, key in zip(_LETTER_BY_INDEX, _OPTION_KEYS):
        opt = str(row.get(key) or "").strip()
        lines.append(f"{letter}. {opt}")
    return "\n".join(lines).strip()


def format_medmcqa_gold_response(letter: str, exp: str) -> str:
    """Gold response: answer letter first, then expert explanation."""
    body = (exp or "").strip()
    letter = (letter or "").strip().upper()
    if body:
        return f"Answer: {letter}\n{body}"
    return f"Answer: {letter}"


def passes_medmcqa_filter(row: Dict[str, Any], *, require_label: bool = True) -> bool:
    """
    Keep only single-choice rows with a non-empty explanation longer than
    MEDMCQA_MIN_EXP_WORDS words. When require_label=True, also require a valid cop.
    """
    if str(row.get("choice_type") or "").strip().lower() != "single":
        return False
    exp = row.get("exp")
    if exp is None:
        return False
    exp_s = str(exp).strip()
    if not exp_s:
        return False
    if word_count(exp_s) <= MEDMCQA_MIN_EXP_WORDS:
        return False
    if require_label and cop_to_letter(row.get("cop")) is None:
        return False
    return True


def _row_as_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return row
    try:
        return dict(row)
    except Exception:
        return {k: row[k] for k in row.keys()}  # type: ignore[attr-defined]


def _iter_filtered_rows(
    ds: Any,
    *,
    require_label: bool = True,
) -> Iterable[Dict[str, Any]]:
    for i in range(len(ds)):
        row = _row_as_dict(ds[i])
        if passes_medmcqa_filter(row, require_label=require_label):
            yield row


def _offline_hub_enabled() -> bool:
    """True when HF Hub / datasets are forced offline."""
    keys = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    return any(
        str(os.environ.get(k, "")).strip().lower() in {"1", "true", "yes", "y"}
        for k in keys
    )


def _with_hub_online_for_dataset(fn):
    """
    Temporarily clear offline Hub flags so a one-time dataset download can succeed.

    Generation_no_wm / Generation_offline set HF_HUB_OFFLINE=1 for model loads;
    MedMCQA still needs a first-time Hub fetch into cache_dir.
    """
    keys = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        # datasets / hub may already have imported module-level offline flags.
        try:
            from datasets import config as ds_config

            if hasattr(ds_config, "HF_HUB_OFFLINE"):
                ds_config.HF_HUB_OFFLINE = False  # type: ignore[attr-defined]
            if hasattr(ds_config, "HF_DATASETS_OFFLINE"):
                ds_config.HF_DATASETS_OFFLINE = False  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            import huggingface_hub.constants as hub_constants

            if hasattr(hub_constants, "HF_HUB_OFFLINE"):
                hub_constants.HF_HUB_OFFLINE = False
        except Exception:
            pass
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def ensure_medmcqa_on_disk(
    cache_dir: str,
    *,
    dataset_id: str = MEDMCQA_DATASET_ID,
    force_redownload: bool = False,
) -> str:
    """
    Ensure openlifescienceai/medmcqa is saved under cache_dir/openlifescienceai_medmcqa.

    Returns the on-disk path. Reuses an existing save unless force_redownload.
    If Generation scripts set HF_HUB_OFFLINE, briefly re-enables Hub access only
    for this one-time dataset download + save_to_disk.
    """
    from datasets import load_dataset, load_from_disk

    disk_path = medmcqa_disk_path(cache_dir)
    marker = os.path.join(disk_path, "dataset_dict.json")
    if (not force_redownload) and os.path.isfile(marker):
        load_from_disk(disk_path)
        return disk_path

    os.makedirs(cache_dir, exist_ok=True)

    def _download_and_save():
        print(
            f"[medmcqa] Downloading {dataset_id} into cache_dir={cache_dir} "
            f"and saving to {disk_path} ..."
        )
        ds = load_dataset(dataset_id, cache_dir=cache_dir)
        ds.save_to_disk(disk_path)
        return disk_path

    # Prefer already-materialized HF datasets cache without Hub metadata calls.
    try:
        ds = load_dataset(
            dataset_id,
            cache_dir=cache_dir,
            download_mode="reuse_dataset_if_exists",
        )
        ds.save_to_disk(disk_path)
        return disk_path
    except Exception as local_err:
        if _offline_hub_enabled():
            print(
                "[medmcqa] Local Arrow save missing and Hub is offline; "
                "temporarily enabling Hub for one-time MedMCQA download."
            )
            try:
                return _with_hub_online_for_dataset(_download_and_save)
            except Exception as online_err:
                raise ConnectionError(
                    f"Could not load {dataset_id} from local cache or Hub.\n"
                    f"Local attempt: {local_err}\n"
                    f"Online attempt: {online_err}\n"
                    "Fix: on a networked login node, run once:\n"
                    f'  python -c "from datasets import load_dataset; '
                    f"ds=load_dataset('{dataset_id}', cache_dir='{cache_dir}'); "
                    f"ds.save_to_disk('{disk_path}')\""
                ) from online_err
        raise


def _resolve_eval_splits(split: str) -> List[str]:
    """
    Map CLI split name to HF split keys.

    - train / validation / test: that split only
    - all: train + validation + test (filtered)
    - validation is also accepted as 'val' / 'dev'
    """
    s = (split or "").strip().lower()
    if s in {"all", "any", "*"}:
        return ["train", "validation", "test"]
    if s in {"val", "dev", "validation"}:
        return ["validation"]
    if s in {"train", "test"}:
        return [s]
    raise ValueError(
        f"Unsupported MedMCQA split {split!r}. "
        "Use train, validation (val/dev), test, or all."
    )


def _collect_fewshot_demos(
    dsd: Any,
    *,
    n_shot: int = MEDMCQA_N_SHOT,
) -> List[Dict[str, Any]]:
    """First n_shot filtered train rows for in-context demos."""
    if "train" not in dsd:
        raise ValueError("MedMCQA train split missing; cannot build few-shot demos.")
    demos: List[Dict[str, Any]] = []
    for row in _iter_filtered_rows(dsd["train"], require_label=True):
        letter = cop_to_letter(row.get("cop"))
        if letter is None:
            continue
        demos.append(
            {
                "id": str(row.get("id") or ""),
                "question": format_medmcqa_question(row),
                "letter": letter,
                "exp": str(row.get("exp") or "").strip(),
            }
        )
        if len(demos) >= n_shot:
            break
    if len(demos) < n_shot:
        raise ValueError(
            f"Need {n_shot} filtered train demos "
            f"(choice_type=single, exp>{MEDMCQA_MIN_EXP_WORDS} words); "
            f"found {len(demos)}."
        )
    return demos


def _build_fewshot_block(demos: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for i, demo in enumerate(demos, start=1):
        response = format_medmcqa_gold_response(demo["letter"], demo["exp"])
        parts.append(
            f"Example {i}:\n"
            f"Question: {demo['question']}\n"
            f"Response: {response}"
        )
    return "\n\n".join(parts)


def build_medmcqa_fewshot_prompt(question: str, fewshot_block: str) -> str:
    """Build the shared N-shot generation prompt for one MedMCQA question."""
    q = (question or "").strip()
    return (
        "Answer the medical multiple-choice question.\n"
        'First give the final choice as "Answer: X" where X is A, B, C, or D.\n'
        "Then explain briefly why that option is correct.\n"
        "Keep the explanation concise and medically accurate.\n"
        "\n"
        f"{fewshot_block}\n"
        "\n"
        "Now answer:\n"
        "\n"
        f"Question: {q}\n"
        "Response:"
    )


def _example_from_row(
    row: Dict[str, Any],
    *,
    fewshot_block: str,
    split_name: str,
) -> Optional[Dict[str, Any]]:
    letter = cop_to_letter(row.get("cop"))
    if letter is None:
        return None
    question = format_medmcqa_question(row)
    gold = format_medmcqa_gold_response(letter, str(row.get("exp") or ""))
    return {
        "question": question,
        "answer": gold,
        "final_answer": letter,
        "prompt": build_medmcqa_fewshot_prompt(question, fewshot_block),
        "id": str(row.get("id") or ""),
        "source_split": split_name,
    }


def load_medmcqa_examples(
    cache_dir: str,
    *,
    n_inputs: Optional[int] = None,
    split: str = MEDMCQA_SPLIT,
    dataset_id: str = MEDMCQA_DATASET_ID,
    n_shot: int = MEDMCQA_N_SHOT,
    pad_with_train: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load MedMCQA from disk under cache_dir (download + save_to_disk on first use).

    Filters every requested split to:
      choice_type == single, non-empty exp with > 50 words, valid cop.

    Each item: {"question": str, "answer": str, "final_answer": str, "prompt": str}
    - question: stem + options
    - answer: gold "Answer: X\\n{exp}" string
    - final_answer: letter A/B/C/D
    - prompt: shared few-shot user text (train demos; Answer-first format)

    Few-shot demo ids are always excluded from the eval pool.
    If n_inputs is set and the requested split(s) yield fewer rows, remaining
    slots are filled from filtered train (excluding few-shot demos), unless
    pad_with_train=False or train was already fully included (split=all/train).
    """
    from datasets import load_from_disk

    disk_path = ensure_medmcqa_on_disk(cache_dir, dataset_id=dataset_id)
    dsd = load_from_disk(disk_path)
    demos = _collect_fewshot_demos(dsd, n_shot=n_shot)
    fewshot_block = _build_fewshot_block(demos)
    demo_ids = {d["id"] for d in demos if d["id"]}

    eval_splits = _resolve_eval_splits(split)
    for name in eval_splits:
        if name not in dsd:
            raise ValueError(
                f"MedMCQA split {name!r} not found on disk at {disk_path}. "
                f"Available: {list(dsd.keys())}"
            )

    examples: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    limit = int(n_inputs) if n_inputs is not None and int(n_inputs) > 0 else None

    def _consume(split_name: str) -> None:
        for row in _iter_filtered_rows(dsd[split_name], require_label=True):
            row_id = str(row.get("id") or "")
            if row_id and (row_id in demo_ids or row_id in seen_ids):
                continue
            ex = _example_from_row(row, fewshot_block=fewshot_block, split_name=split_name)
            if ex is None:
                continue
            if row_id:
                seen_ids.add(row_id)
            examples.append(ex)
            if limit is not None and len(examples) >= limit:
                return

    for split_name in eval_splits:
        _consume(split_name)
        if limit is not None and len(examples) >= limit:
            return examples

    # Top up from train when the primary split(s) are short of n_inputs.
    need_pad = (
        pad_with_train
        and limit is not None
        and len(examples) < limit
        and "train" not in eval_splits
        and "train" in dsd
    )
    if need_pad:
        before = len(examples)
        _consume("train")
        added = len(examples) - before
        print(
            f"[medmcqa] Primary split(s) {eval_splits} yielded {before} filtered rows; "
            f"padded +{added} from train (excluded {len(demo_ids)} few-shot demos) "
            f"→ {len(examples)}/{limit}."
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
        return user_text
