import json
import random
import string
import os
from typing import List, Dict, Iterator, Tuple, Optional

# ------------- Utilities -------------

def set_seed(seed: int = 42):
    random.seed(seed)

def generate_random_key(length=8) -> str:
    """Generate a random key string (not used in single-key deployment; kept for completeness)."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def _iter_texts_from_file(path: str, text_field: str) -> Iterator[str]:
    """
    Yield text strings from a JSON array file or JSONL file using given field.

    - If the file is a JSON array of objects, each object should contain `text_field`.
    - If the file is JSON Lines (one JSON object per line), each object should contain `text_field`.
    - If a line isn't valid JSON and `text_field` is "text", the raw line is yielded as text.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Try JSON first
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and text_field in item and item[text_field] is not None:
                    t = str(item[text_field]).strip()
                    if t:
                        yield t
        elif isinstance(data, dict):
            # try to find a likely list
            for v in data.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and text_field in item and item[text_field] is not None:
                            t = str(item[text_field]).strip()
                            if t:
                                yield t
                    break
            # also allow a single field at root
            if text_field in data and data[text_field]:
                t = str(data[text_field]).strip()
                if t:
                    yield t
        return
    except json.JSONDecodeError:
        pass  # Fall back to JSONL

    # JSONL fallback
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and text_field in obj and obj[text_field] is not None:
                    t = str(obj[text_field]).strip()
                    if t:
                        yield t
            except json.JSONDecodeError:
                if text_field == "text":
                    yield line

def _print_stats(rows: List[Dict[str, object]], title: str):
    n = len(rows)
    pos = sum(1 for r in rows if int(r["label"]) == 1)
    neg = n - pos
    ex = rows[:3]
    print(f"[{title}] total={n}  pos={pos}  neg={neg}")
    if ex:
        print("  sample:", ex[0])

def _dedup_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Deduplicate by (text, key, label)."""
    seen = set()
    out = []
    for r in rows:
        tup = (r["text"], r["key"], int(r["label"]))
        if tup not in seen:
            seen.add(tup)
            out.append(r)
    return out

# ------------- Core builder -------------

def create_jsonl_from_file_specs(
    file_specs: List[Dict[str, object]],
    output_file: str,
    shuffle: bool = True,
    seed: int = 42,
    dedup: bool = True,
    limit_per_spec: Optional[int] = None,
    balance_labels: bool = False,
) -> None:
    """
    Create a combined JSONL dataset from multiple files with per-file schema.

    Each spec must contain:
      - path: str        -> input file path (JSON array or JSONL)
      - text_field: str  -> field name to read text from (e.g., "Watermarked_output")
      - key: str         -> secret key to attach to rows from this file (e.g., "AABBCC")
      - label: int       -> 1 or 0 label for all rows from this file
      - num: Optional[int] -> if provided, cap rows taken from this specific file (overrides limit_per_spec for this file)

    Args:
      shuffle: shuffle final rows.
      seed: RNG seed for reproducibility.
      dedup: drop duplicate (text,key,label) triples.
      limit_per_spec: cap number of rows read from each spec (useful for balancing).
      balance_labels: if True, randomly downsample the majority class to match the minority.
    """
    set_seed(seed)
    combined: List[Dict[str, object]] = []

    for spec in file_specs:
        path = str(spec.get("path", ""))
        text_field = str(spec.get("text_field", "text"))
        key = str(spec.get("key", ""))
        label = int(spec.get("label", 0))
        per_spec_limit = spec.get("num")

        count = 0
        for text in _iter_texts_from_file(path, text_field):
            if not text:
                continue
            combined.append({"text": text, "key": key, "label": label})
            count += 1
            # Respect per-file limit if provided, otherwise fall back to global limit_per_spec
            effective_limit = per_spec_limit if per_spec_limit is not None else limit_per_spec
            if effective_limit is not None and count >= int(effective_limit):
                break

    if dedup:
        before = len(combined)
        combined = _dedup_rows(combined)
        after = len(combined)
        if after < before:
            print(f"[dedup] removed {before-after} duplicates")

    if balance_labels:
        pos = [r for r in combined if int(r["label"]) == 1]
        neg = [r for r in combined if int(r["label"]) == 0]
        m = min(len(pos), len(neg))
        random.shuffle(pos); random.shuffle(neg)
        combined = pos[:m] + neg[:m]
        print(f"[balance] pos={len(pos)} neg={len(neg)} -> using {m}+{m}={2*m}")

    if shuffle:
        random.shuffle(combined)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in combined:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    _print_stats(combined, f"written -> {output_file}")

# ------------- Your concrete usage (single deployment key: AABBCC) -------------

if __name__ == "__main__":
    # TRAIN: Positive = WM(AABBCC) with AABBCC, Negatives = WM(Mysecretekey) with AABBCC, Clean with AABBCC
    create_jsonl_from_file_specs([
        ## PRIMARY DATA - MUST HAVE
        # watermarked text with correct key
        {"path": "Train_Llama2_WHOLE_CONTEXT_BERTScore_threshold_0.9_KEY_Adaptive_key_v1_m10_c2_h6_alpha1.0_0_20000_20000.json",
         "text_field": "Watermarked_output", "key": "Adaptive_key_v1", "label": 1, "num": 20000},
        # clean negatives (no watermark)
        {"path": "Train_LLaMA_100k.json",
         "text_field": "output_only", "key": "Adaptive_key_v1", "label": 0, "num": 10000},
         ## ADD MORE DATA IF YOU WANT TO TEST CROSS-MODEL
         # Mistral
        {"path": "Train_Mistral_WHOLE_CONTEXT_BERTScore_threshold_0.92_KEY_Mistral_user_002_m8_c3_h4_alpha1.2_0_20000_20000.json",
        "text_field": "Watermarked_output", "key": "Adaptive_key_v1", "label": 0, "num": 10000},
        {"path": "Train_Mistral_WHOLE_CONTEXT_BERTScore_threshold_0.92_KEY_Mistral_user_002_m8_c3_h4_alpha1.2_0_20000_20000.json",
        "text_field": "Watermarked_output", "key": "Mistral_user_002", "label": 1, "num": 20000},
        # # DeepSeek
        # {"path": "Train_DeepSeek_WHOLE_CONTEXT_BERTScore_threshold_0.93_KEY_DeepSeek_LLM_m10_c2_h6_alpha1.1_0_20000_20000.json",
        # "text_field": "Watermarked_output", "key": "Adaptive_key_v1", "label": 0, "num": 10000},
        # {"path": "Train_DeepSeek_WHOLE_CONTEXT_BERTScore_threshold_0.93_KEY_DeepSeek_LLM_m10_c2_h6_alpha1.1_0_20000_20000.json",
        # "text_field": "Watermarked_output", "key": "DeepSeek_LLM", "label": 1, "num": 20000},
        # # Gemma
        # {"path": "Train_Gemma_WHOLE_CONTEXT_BERTScore_threshold_0.935_KEY_Gemma_key_m10_c2_h6_alpha1.0_10000.json",
        # "text_field": "Watermarked_output", "key": "Adaptive_key_v1", "label": 0, "num": 10000},
        # {"path": "Train_Gemma_WHOLE_CONTEXT_BERTScore_threshold_0.935_KEY_Gemma_key_m10_c2_h6_alpha1.0_10000.json",
        # "text_field": "Watermarked_output", "key": "Gemma_key", "label": 1, "num": 10000},
        # # Qwen
        # {"path": "Train_Qwen_WHOLE_CONTEXT_BERTScore_threshold_0.94_KEY_Qwen_model_001_m10_c2_h6_alpha1.0_10000.json",
        # "text_field": "Watermarked_output", "key": "Adaptive_key_v1", "label": 0, "num": 10000},
        # {"path": "Train_Qwen_WHOLE_CONTEXT_BERTScore_threshold_0.94_KEY_Qwen_model_001_m10_c2_h6_alpha1.0_10000.json",
        # "text_field": "Watermarked_output", "key": "Qwen_model_001", "label": 1, "num": 10000},
        # File name for train dataset
    ], output_file="Train_Main_llama2_sentence_added_Mistral_DeepSeek_Gemma_Qwen_universal_5keys.jsonl",
       shuffle=True, seed=123, dedup=True, limit_per_spec=None, balance_labels=False)

    # TEST: same pattern (use held-out files)
    create_jsonl_from_file_specs([
        ## PRIMARY DATA - MUST HAVE
        # watermarked text with correct key
        {"path": "Test_Llama2_WHOLE_CONTEXT_BERTScore_threshold_0.9_KEY_Adaptive_key_v1_m10_c2_h6_alpha1.0_0_1000_1000.json",
         "text_field": "Watermarked_output", "key": "Adaptive_key_v1", "label": 1, "num": 1000},
        # clean negatives (no watermark)
        {"path": "output_llama2_test_final.json",
         "text_field": "output_only", "key": "Adaptive_key_v1", "label": 0, "num": 1000},
         ## ADD MORE DATA IF YOU WANT TO TEST CROSS-MODEL
         # Mistral
        {"path": "Test_Mistral_WHOLE_CONTEXT_BERTScore_threshold_0.92_KEY_Mistral_user_002_m8_c3_h4_alpha1.2_0_1000_1000.json",
           "text_field": "Watermarked_output", "key": "Adaptive_key_v1", "label": 0, "num": 1000},
        {"path": "Test_Mistral_WHOLE_CONTEXT_BERTScore_threshold_0.92_KEY_Mistral_user_002_m8_c3_h4_alpha1.2_0_1000_1000.json",
         "text_field": "Watermarked_output", "key": "Mistral_user_002", "label": 1, "num": 1000},
        # # DeepSeek
        # {"path": "Test_DeepSeek_WHOLE_CONTEXT_BERTScore_threshold_0.93_KEY_DeepSeek_LLM_m10_c2_h6_alpha1.1_0_1000_1000.json",
        # "text_field": "Watermarked_output", "key": "Adaptive_key_v1", "label": 0, "num": 1000},
        # {"path": "Test_DeepSeek_WHOLE_CONTEXT_BERTScore_threshold_0.93_KEY_DeepSeek_LLM_m10_c2_h6_alpha1.1_0_1000_1000.json",
        # "text_field": "Watermarked_output", "key": "DeepSeek_LLM", "label": 1, "num": 1000},
        # # Gemma
        # {"path": "Test_Gemma_WHOLE_CONTEXT_BERTScore_threshold_0.935_KEY_Gemma_key_m10_c2_h6_alpha1.0_0_1000_1000.json",
        # "text_field": "Watermarked_output", "key": "Adaptive_key_v1", "label": 0, "num": 1000},
        # {"path": "Test_Gemma_WHOLE_CONTEXT_BERTScore_threshold_0.935_KEY_Gemma_key_m10_c2_h6_alpha1.0_0_1000_1000.json",
        # "text_field": "Watermarked_output", "key": "Gemma_key", "label": 1, "num": 1000},
        # # Qwen
        # {"path": "Test_Qwen_WHOLE_CONTEXT_BERTScore_threshold_0.94_KEY_Qwen_model_001_m10_c2_h6_alpha1.0_0_1000_1000.json",
        # "text_field": "Watermarked_output", "key": "Adaptive_key_v1", "label": 0, "num": 1000},
        # {"path": "Test_Qwen_WHOLE_CONTEXT_BERTScore_threshold_0.94_KEY_Qwen_model_001_m10_c2_h6_alpha1.0_0_1000_1000.json",
        # "text_field": "Watermarked_output", "key": "Qwen_model_001", "label": 1, "num": 1000},
        # File name for test dataset
    ], output_file="Test_Main_llama2_sentence_added_Mistral_DeepSeek_Gemma_Qwen_universal_5keys.jsonl",
       shuffle=True, seed=123, dedup=True, limit_per_spec=None, balance_labels=False)
