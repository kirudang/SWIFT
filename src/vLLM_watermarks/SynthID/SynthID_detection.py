#!/usr/bin/env python3
"""Detect SynthID watermark on batch_generate_synthid_gemma.py JSON output.

Detector parameters must match generation
(``batch_generate_synthid_gemma.py`` defaults):
  ngram=4, sampling_table_seed=0, sampling_table_size=65536,
  keys=DEFAULT_SYNTHID_KEYS (depth=30), threshold=0.52.

Scores ``output_only`` (continuation text), same as inline detection during
generation. Generator-only knobs (``seed``, ``context_history_size``) are not
used by the detector.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from transformers import AutoTokenizer
from vllm_watermark.core import DetectionAlgorithm
from vllm_watermark.watermark_detectors import WatermarkDetectors

# Same keys as batch_generate_synthid_gemma.py / SynthIDDetector defaults.
DEFAULT_SYNTHID_KEYS = [
    654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
    590, 639, 13, 715, 468, 990, 966, 226, 324, 585,
    118, 504, 421, 521, 129, 669, 732, 225, 90, 960,
]

INPUT_JSON = "Your Data Path"
column_name = "Watermarked_output" #"Original_output" # "watermarked_answer" #
OUTPUT_CSV = "SynthID_Gemma3_27B_Med_wm.csv"
BASE_MODEL = "google/gemma-3-27b-it"
CACHE_DIR = "Your Cache Directory"  # adjust if needed

# Match batch_generate_synthid_gemma.py detector kwargs.
NGRAM = 4
SAMPLING_TABLE_SEED = 0
SAMPLING_TABLE_SIZE = 65536
DETECTION_THRESHOLD = 0.52  # mean g-value threshold

os.environ["HF_HOME"] = CACHE_DIR

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
detector = WatermarkDetectors.create(
    algo=DetectionAlgorithm.SYNTHID,
    tokenizer=tokenizer,
    ngram=NGRAM,
    sampling_table_seed=SAMPLING_TABLE_SEED,
    sampling_table_size=SAMPLING_TABLE_SIZE,
    keys=list(DEFAULT_SYNTHID_KEYS),
    threshold=DETECTION_THRESHOLD,
)

with open(INPUT_JSON) as f:
    data = json.load(f)

detected = 0
for row in data:
    text = row.get(column_name, "")
    if text:
        result = detector.detect(text)
        # SynthID: score = mean g-value; pvalue is a placeholder for API parity
        row["score"] = float(result["score"])
        row["pvalue"] = float(result["pvalue"])
        row["is_watermarked"] = bool(result["is_watermarked"])
        detected += int(row["is_watermarked"])

print(f"Scored {len(data)} rows")
print(f"Detected watermarked: {detected}/{len(data)} ({100 * detected / len(data):.1f}%)")

fieldnames = list(dict.fromkeys(k for row in data for k in row.keys()))
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(data)
print(f"Wrote {OUTPUT_CSV}")
