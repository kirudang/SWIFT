"""
AlpacaEval-style pairwise OpenAI judge for watermark outputs.

Compares two watermarked responses against the original unwatermarked response
and reports win rates (with position randomization to reduce order bias).
"""

import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# -----------------------------------------------------------------------------
# API OPENAI
# -----------------------------------------------------------------------------
OPENAI_API_KEY = "Your OpenAI API Key"
MODEL = "gpt-5.5"

# Method labels (for reporting)
A = "SWIFT"
B = "SIRA"

# Watermarked method A
A_PATH = "./Data/method_a.json"
COLUMN_A = "Watermarked_output"

# Watermarked method B
B_PATH = "./Data/method_b.json"
COLUMN_B = "Watermarked_output"

# Original / reference (prompt + unwatermarked base response)
ORIGINAL_PATH = A_PATH
ORIGINAL_PROMPT_COL = "input"
ORIGINAL_BASE_COL = "Original_output"

OUTPUT_PATH = "SIRA_vs_SWIFT_win_rate.json"

MAX_SAMPLES = 1000
WORKERS = 10
SEED = 42


# ============================================================
# JUDGE PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an impartial evaluator for text watermarking utility.

You will be given:
1. A user prompt.
2. The original unwatermarked response from the base model.
3. Two watermarked responses (Response A and Response B).

Your task is to decide which watermarked response better preserves the original
response while still answering the prompt.

Evaluation criteria:
- Semantic preservation: preserves the overall meaning of the base response.
- Factual consistency and precision: does not add or remove important facts.
  It preserves named entities, numbers, dates, and technical terms.
- Fluency: remains natural and grammatically correct.

Return JSON only with:
- winner: "A", "B", or "Tie"
- reason: a brief explanation
""".strip()


# ============================================================
# LOAD DATA
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_paired_data(
    a_path=A_PATH,
    b_path=B_PATH,
    col_a=COLUMN_A,
    col_b=COLUMN_B,
    original_path=ORIGINAL_PATH,
    original_prompt_col=ORIGINAL_PROMPT_COL,
    original_col=ORIGINAL_BASE_COL,
    max_samples=MAX_SAMPLES,
):
    a_rows = load_json(a_path)

    if Path(a_path).resolve() == Path(b_path).resolve():
        b_rows = a_rows
    else:
        b_rows = load_json(b_path)

    if Path(original_path).resolve() == Path(a_path).resolve():
        original_rows = a_rows
    elif Path(original_path).resolve() == Path(b_path).resolve():
        original_rows = b_rows
    else:
        original_rows = load_json(original_path)

    n = min(len(a_rows), len(b_rows), len(original_rows), max_samples)

    data = []
    for i in range(n):
        a_row, b_row, orig_row = a_rows[i], b_rows[i], original_rows[i]

        missing = {
            "original": [
                col
                for col in (original_prompt_col, original_col)
                if col not in orig_row
            ],
            A: [col for col in (col_a,) if col not in a_row],
            B: [col for col in (col_b,) if col not in b_row],
        }
        missing = {label: cols for label, cols in missing.items() if cols}
        if missing:
            raise ValueError(f"Missing columns at index {i}: {missing}")

        data.append(
            {
                "prompt": str(orig_row[original_prompt_col] or ""),
                "base": str(orig_row[original_col] or ""),
                "a": str(a_row[col_a] or ""),
                "b": str(b_row[col_b] or ""),
            }
        )
    return data


# ============================================================
# OPENAI JUDGE
# ============================================================

def build_pair_request(row, index, seed):
    """Randomize A/B presentation order to reduce position bias."""
    rng = random.Random(f"{seed}:{index}")
    if rng.random() < 0.5:
        a_method, b_method = "a", "b"
    else:
        a_method, b_method = "b", "a"

    response_a, response_b = row[a_method], row[b_method]
    map_label = {"A": a_method, "B": b_method, "Tie": "Tie"}
    return response_a, response_b, map_label, a_method, b_method


def judge_one(client, index, row, seed):
    response_a, response_b, map_label, a_method, b_method = build_pair_request(
        row, index, seed
    )

    prompt = f"""
User prompt:
{row["prompt"]}

Original unwatermarked response:
{row["base"]}

Response A:
{response_a}

Response B:
{response_b}

Return JSON only:
{{
    "winner": "A",
    "reason": "brief explanation"
}}
"""

    for attempt in range(5):
        try:
            response = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "pairwise_winner",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "winner": {
                                    "type": "string",
                                    "enum": ["A", "B", "Tie"],
                                },
                                "reason": {
                                    "type": "string",
                                },
                            },
                            "required": ["winner", "reason"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    }
                },
            )

            result = json.loads(response.output_text)
            judge_label = result["winner"]
            winner = map_label[judge_label]

            return {
                "index": index,
                "a_method": a_method,
                "b_method": b_method,
                "judge_label": judge_label,
                "winner": winner,
                "reason": result["reason"],
                "error": "",
            }

        except Exception as e:
            if attempt == 4:
                return {
                    "index": index,
                    "a_method": a_method,
                    "b_method": b_method,
                    "judge_label": "Tie",
                    "winner": "Tie",
                    "reason": "",
                    "error": str(e),
                }

            time.sleep(2 ** attempt)


# ============================================================
# EVALUATE
# ============================================================

def evaluate(data, seed=SEED):
    client = OpenAI(api_key=OPENAI_API_KEY)

    results = [None] * len(data)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(judge_one, client, i, row, seed): i
            for i, row in enumerate(data)
        }

        completed = 0
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()

            completed += 1
            if completed % 10 == 0 or completed == len(data):
                print(f"[{A} vs {B}] completed {completed}/{len(data)}")

    return results


def summarize(results):
    counts = defaultdict(int)
    for r in results:
        if r["error"]:
            continue
        counts[r["winner"]] += 1

    n = len(results)
    scored = sum(1 for r in results if not r["error"])
    a_wins = counts["a"]
    b_wins = counts["b"]
    ties = counts["Tie"]

    return {
        "model": MODEL,
        "comparison": f"{A}_vs_{B}",
        "A": A,
        "B": B,
        "n": n,
        "successfully_scored": scored,
        f"{A}_wins": a_wins,
        f"{B}_wins": b_wins,
        "ties": ties,
        f"{A}_win_rate": (a_wins + 0.5 * ties) / n if n else 0.0,
        f"{B}_win_rate": (b_wins + 0.5 * ties) / n if n else 0.0,
        "errors": sum(1 for r in results if r["error"]),
    }


# ============================================================
# SAVE
# ============================================================

def save_results(summary, results, output_path=OUTPUT_PATH):
    payload = {
        "summary": summary,
        "judgments": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\nFinished")
    print(f"Comparison: {summary['comparison']}")
    print(f"{A} wins: {summary[f'{A}_wins']}")
    print(f"{B} wins: {summary[f'{B}_wins']}")
    print(f"Ties:     {summary['ties']}")
    print(f"{A} win rate: {summary[f'{A}_win_rate']:.4f}")
    print(f"{B} win rate: {summary[f'{B}_win_rate']:.4f}")
    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if not OPENAI_API_KEY or OPENAI_API_KEY == "Your OpenAI API Key":
        raise ValueError("Add your OpenAI API key to OPENAI_API_KEY.")

    data = load_paired_data()

    print(f"Model: {MODEL}")
    print(f"{A}: {A_PATH} -> {COLUMN_A}")
    print(f"{B}: {B_PATH} -> {COLUMN_B}")
    print(f"Original: {ORIGINAL_PATH} -> prompt={ORIGINAL_PROMPT_COL}, base={ORIGINAL_BASE_COL}")
    print(f"Samples: {len(data)}")
    print()

    results = evaluate(data)
    summary = summarize(results)
    save_results(summary, results)
    