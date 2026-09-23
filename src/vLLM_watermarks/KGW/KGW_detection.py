#!/usr/bin/env python3
"""Detect KGW watermark on batch_generate_kgw_gemma.py JSON output."""
import csv
import json
import os
import sys

sys.path.insert(0, "Your Path")

from transformers import AutoTokenizer
from vllm_watermark.core import DetectionAlgorithm
from vllm_watermark.watermark_detectors import WatermarkDetectors

INPUT_JSON = "Your Data Path"
column_name = "Original_output" #"Watermarked_output" #
OUTPUT_CSV = "Your Output Name.csv"
BASE_MODEL = "google/gemma-3-27b-it"
CACHE_DIR = "Your Cache Directory"  # adjust if needed

os.environ["HF_HOME"] = CACHE_DIR

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
detector = WatermarkDetectors.create(
    algo=DetectionAlgorithm.MARYLAND_Z,
    tokenizer=tokenizer,
    ngram=2,
    seed=42,
    gamma=0.5,
    delta=2.0,
    threshold=0.05,
)

with open(INPUT_JSON) as f:
    data = json.load(f)

detected = 0
for row in data:
    text = row.get(column_name, "")
    if text:
        result = detector.detect(text)
        row["pvalue"] = float(result["pvalue"])
        row["is_watermarked"] = bool(result["is_watermarked"])
        detected += int(row["is_watermarked"])

print(f"Scored {len(data)} rows")
print(f"Detected watermarked: {detected}/{len(data)} ({100*detected/len(data):.1f}%)")

fieldnames = list(dict.fromkeys(k for row in data for k in row.keys()))
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(data)
print(f"Wrote {OUTPUT_CSV}")
