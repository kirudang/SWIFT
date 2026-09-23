"""Download openai/gsm8k (main) and save the DatasetDict to local disk."""

import os

from datasets import load_dataset

# Hugging Face download cache. Leave unset to use the default HF cache.
cache_dir = os.environ.get("HF_HOME") or None

# On-disk DatasetDict (load later with datasets.load_from_disk).
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openai_gsm8k")

marker = os.path.join(save_path, "dataset_dict.json")
if not os.path.isfile(marker):
    dataset = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)
    dataset.save_to_disk(save_path)

print(f"Data saved to {save_path}")
