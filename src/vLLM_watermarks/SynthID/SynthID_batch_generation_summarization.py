#!/usr/bin/env python3
"""
Batch SynthID watermark generation for CNN/Daily Mail summarization.

Same SynthID / vLLM plumbing as ``batch_generate_synthid_gemma.py``, but loads
CNN/Daily Mail articles from ``cnn.json``, wraps each article in a few-shot
summarization instruction, and writes ``Watermarked_summary`` (no separate
original summary).

How SynthID plugs into vLLM (same mechanism as KGW)
---------------------------------------------------
* ``WatermarkedLLMs.create(..., SYNTHID)`` swaps vLLM's sampler for the shared
  ``WatermarkSampler``. Each decode step it hands the ``(batch, vocab)`` logits
  to ``SynthIDGenerator.sample_next()``.
* SynthID is already batch-aware: it computes per-row g-values, applies the
  non-distortionary tournament update, and samples all rows together. Passing a
  whole mini-batch to ``generate()`` is the efficient path.
* ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` is required so the in-process sampler
  swap is visible.

Run
---
    python scripts/generation/batch_generate_synthid_gemma_summarization.py
    python scripts/generation/batch_generate_synthid_gemma_summarization.py --start_index 0 --Ninputs 1000
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Must be set before vLLM builds its engine so the in-process sampler swap works.
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# vLLM v1 defaults to FlashInfer sampling; JIT needs nvcc + CUDA_HOME.
# Disable so startup works without a full toolkit (override with =1 if available).
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import fire
from transformers import AutoTokenizer

# Allow running as a plain script from anywhere in the repo.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vllm import LLM, SamplingParams  # noqa: E402

from vllm_watermark.core import (  # noqa: E402
    DetectionAlgorithm,
    WatermarkedLLMs,
    WatermarkingAlgorithm,
)
from vllm_watermark.watermark_detectors import WatermarkDetectors  # noqa: E402

DEFAULT_CNN_DATA = "Your Data Path"
# Extra tokens reserved for the few-shot instruction wrapper around the article.
_SUMMARIZATION_INSTRUCTION_OVERHEAD = 256

# Library defaults from SynthIDGenerator / SynthIDDetector (depth = len(keys)).
DEFAULT_SYNTHID_KEYS = [
    654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
    590, 639, 13, 715, 468, 990, 966, 226, 324, 585,
    118, 504, 421, 521, 129, 669, 732, 225, 90, 960,
]


def _patch_gemma3_config_for_vllm(config: Any) -> Any:
    """Alias Gemma3 sliding-window fields expected by vLLM 0.10.

    Newer Hugging Face configs expose ``sliding_window`` / ``layer_types``, while
    vLLM 0.10 still reads ``interleaved_sliding_window`` (and can crash with
    AttributeError on ``Gemma3TextConfig``).
    """
    text_config = getattr(config, "text_config", config)
    sliding_window = getattr(text_config, "sliding_window", 1024)
    object.__setattr__(text_config, "interleaved_sliding_window", sliding_window)
    if not hasattr(text_config, "sliding_window_pattern"):
        object.__setattr__(
            text_config,
            "sliding_window_pattern",
            getattr(text_config, "_sliding_window_pattern", 6),
        )
    return config


def _build_summarization_prompt(article: str) -> str:
    """Few-shot summarization instruction used for CNN/Daily Mail generation."""
    return (
        "\n"
        "    Input: The CNN/Daily Mail dataset is one of the most widely used datasets for text summarization. \n"
        "    It contains news articles and their corresponding highlights, which act as summaries.\n"
        "    State-of-the-art models often use this dataset to fine-tune their summarization capabilities.\n"
        "\n"
        "    Example Summary: The CNN/Daily Mail dataset is commonly used for training summarization models with news articles and highlights.\n"
        "\n"
        f"    Now summarize the following text with maximum 60 words: {article}\n"
        "    The summary is:"
    )


def _extract_summary(generated: str) -> str:
    """Keep only the summary body if the model echoes the prompt cue."""
    marker = "The summary is:"
    if marker in generated:
        return generated.split(marker)[-1].strip()
    return generated.strip()


def _load_cnn_samples(
    data_path: str, split: str, start_index: int, n_inputs: int
) -> List[Dict[str, Any]]:
    """Load CNN/Daily Mail samples filtered by ``type`` (test/train)."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"data_path must be a JSON list (got {type(raw).__name__}): {data_path}")
    split_key = str(split).strip().lower()
    matched: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).strip().lower() != split_key:
            continue
        article = item.get("article")
        if not isinstance(article, str) or not article.strip():
            continue
        matched.append({
            "id": item.get("id"),
            "article": article,
            "highlights": item.get("highlights", ""),
            "type": item.get("type", split_key),
        })
    samples = matched[start_index : start_index + n_inputs]
    if not samples:
        raise ValueError(
            f"No usable CNN/Daily Mail samples in {data_path} for type={split_key!r} "
            f"with start_index={start_index} n_inputs={n_inputs}."
        )
    return samples


def run(
    # --- data / model ---
    data_path: str = DEFAULT_CNN_DATA,
    cache_dir: str = "Your Cache Directory",
    base_model: str = "google/gemma-3-27b-it",
    output_name: str = "Your Output Name",
    split: str = "test",
    start_index: int = 0,
    Ninputs: int = 1000,
    batch_size: int = 32,
    max_inp: int = 1024,
    max_out: int = 100,
    min_out: int = 20,
    saving_freq: int = 20,
    # --- SynthID watermark parameters ---
    seed: int = 42,
    ngram: int = 4,
    sampling_table_seed: int = 0,
    sampling_table_size: int = 65536,
    context_history_size: int = 1024,
    keys: Optional[list[int]] = None,
    # --- sampling ---
    temperature: float = 1.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.001,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = True,
    # Gemma-3-27B is multimodal with a huge default context; cap for text generation.
    dtype: str = "bfloat16",
    max_model_len: Optional[int] = None,
    # --- optional detection (adds extra keys to each record when True) ---
    detect: bool = False,
    detection_threshold: float = 0.52,
):
    """Generate SynthID-watermarked CNN/Daily Mail summaries.

    Args:
        data_path: Path to CNN/Daily Mail JSON (``id``, ``article``, ``highlights``, ``type``).
        cache_dir: HF cache / vLLM download dir (sets ``HF_HOME``).
        base_model: HF model id to serve with vLLM.
        output_name: Prefix for the JSON checkpoint files.
        split: Dataset split mapped to sample ``type`` (``test`` / ``train``).
        start_index: Offset into the filtered split to start from.
        Ninputs: Number of articles to process starting at ``start_index``.
        batch_size: Articles handed to each ``generate()`` call.
        max_inp: Max article tokens before wrapping in the summarization instruction.
        max_out: Max new tokens to generate per summary.
        min_out: Minimum new tokens before EOS is allowed (vLLM ``min_tokens``).
            Prevents empty summaries when Gemma-IT stops immediately after the
            instruction. Clamped to ``max_out``; use 0 to disable.
        saving_freq: Save a checkpoint every this many processed articles.
        seed: Base RNG seed (multinomial draws).
        ngram: Context window size (must match the detector).
        sampling_table_seed: Seed for the SynthID sampling table (must match
            the detector).
        sampling_table_size: Size of the binary sampling table (default 65536).
        context_history_size: Max remembered contexts; repeats skip the
            watermark update (generator only).
        keys: Per-layer tournament keys; ``len(keys)`` is the depth / number of
            rounds. Defaults to the library's 30-key secret. Pass a Fire list,
            e.g. ``--keys=1,2,3``.
        temperature, top_p: Sampling parameters (any temperature; 0 = greedy).
        frequency_penalty: Tiny penalty so vLLM V1 exposes token ids to the
            watermark sampler (set 0.0 to disable if not needed).
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        enforce_eager: Disable CUDA graph capture (matches the SynthID example;
            set False for speed if your setup is stable with graphs).
        dtype: vLLM weight dtype (Gemma3 checkpoints are bfloat16).
        max_model_len: Cap KV-cache context length. Defaults to
            ``max_inp + max_out + instruction overhead``.
        detect: If True, score each row and add g-value / flag keys.
        detection_threshold: Mean g-value threshold for the watermark flag.
    """
    # HF_HOME is read lazily at from_pretrained / LLM() time, so setting it here
    # (before those calls) is sufficient.
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_TOKEN"] = "Your HF Token"

    if max_model_len is None:
        max_model_len = max_inp + max_out + _SUMMARIZATION_INSTRUCTION_OVERHEAD

    if keys is None:
        keys = list(DEFAULT_SYNTHID_KEYS)
    elif isinstance(keys, (tuple, list)):
        keys = [int(k) for k in keys]
    else:
        raise TypeError(f"keys must be a list of ints, got {type(keys)}")

    print("=" * 60)
    print("BATCH SYNTHID SUMMARIZATION WATERMARK (Gemma3)")
    print("=" * 60)
    print(f"data:         {data_path}")
    print(f"split:        {split}")
    print(f"model:        {base_model}")
    print(f"output_name:  {output_name}")
    print(f"range:        [{start_index}, {start_index + Ninputs}) batch_size={batch_size}")
    min_out = max(0, min(int(min_out), int(max_out)))
    print(f"tokens:       max_inp={max_inp} min_out={min_out} max_out={max_out} max_model_len={max_model_len}")
    print(
        f"watermark:    seed={seed} ngram={ngram} "
        f"sampling_table_seed={sampling_table_seed} "
        f"sampling_table_size={sampling_table_size} "
        f"context_history_size={context_history_size} "
        f"depth={len(keys)}"
    )
    print(f"keys:         {keys}")
    print(f"sampling:     temperature={temperature} top_p={top_p} dtype={dtype}")
    print("=" * 60)

    # Tokenizer (used for article truncation before instruction wrapping).
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Base vLLM model, then wrap with SynthID watermarking.
    # hf_overrides patches the Gemma3 config rename that breaks vLLM 0.10.
    llm = LLM(
        model=base_model,
        download_dir=cache_dir,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        hf_overrides=_patch_gemma3_config_for_vllm,
    )
    wm_llm = WatermarkedLLMs.create(
        llm,
        algo=WatermarkingAlgorithm.SYNTHID,
        seed=seed,
        ngram=ngram,
        sampling_table_seed=sampling_table_seed,
        sampling_table_size=sampling_table_size,
        context_history_size=context_history_size,
        keys=keys,
    )

    detector = None
    if detect:
        detector = WatermarkDetectors.create(
            algo=DetectionAlgorithm.SYNTHID,
            model=llm,
            ngram=ngram,
            sampling_table_seed=sampling_table_seed,
            sampling_table_size=sampling_table_size,
            keys=keys,
            threshold=detection_threshold,
        )

    sampling_kwargs = dict(temperature=temperature, top_p=top_p, max_tokens=max_out)
    if min_out > 0:
        sampling_kwargs["min_tokens"] = min_out
    if frequency_penalty:
        sampling_kwargs["frequency_penalty"] = frequency_penalty
    sampling_params = SamplingParams(**sampling_kwargs)

    samples = _load_cnn_samples(data_path, split, start_index, Ninputs)
    print(f"Loaded {len(samples)} CNN/Daily Mail articles (type={split})")

    generated_data = []
    input_counter = 0
    last_saved_path: Optional[str] = None
    start_time = time.time()

    for batch_start in range(0, len(samples), batch_size):
        start_time_batch = time.time()

        batch_samples = samples[batch_start : batch_start + batch_size]
        if not batch_samples:
            break

        batch_articles = [s["article"] for s in batch_samples]

        # Truncate each article to max_inp tokens (HF), then wrap instruction.
        encoded = tokenizer(
            batch_articles,
            add_special_tokens=True,
            truncation=True,
            max_length=max_inp,
        )
        truncated_articles = tokenizer.batch_decode(
            encoded["input_ids"], skip_special_tokens=True
        )
        generated_prompts = [
            _build_summarization_prompt(article) for article in truncated_articles
        ]

        # One batched, watermarked generate call: outputs stay aligned by index.
        outputs = wm_llm.generate(generated_prompts, sampling_params=sampling_params)

        time_elapsed_batch = time.time() - start_time_batch
        n_in_batch = len(batch_samples)
        time_per_item = time_elapsed_batch / n_in_batch

        for local_idx in range(n_in_batch):
            sample = batch_samples[local_idx]
            watermarked_summary = _extract_summary(outputs[local_idx].outputs[0].text)

            data_dict = {
                "id": sample["id"],
                "article": sample["article"],
                "highlights": sample["highlights"],
                "Watermarked_summary": watermarked_summary,
                "type": sample["type"],
                "time": time_per_item,
            }

            if detector is not None and watermarked_summary:
                result = detector.detect(watermarked_summary)
                data_dict["pvalue"] = float(result["pvalue"])
                data_dict["is_watermarked"] = bool(result["is_watermarked"])

            generated_data.append(data_dict)
            input_counter += 1

            if input_counter % saving_freq == 0:
                prev_path = output_name + str(input_counter - saving_freq) + ".json"
                if os.path.isfile(prev_path):
                    os.remove(prev_path)
                last_saved_path = output_name + str(input_counter) + ".json"
                with open(last_saved_path, "w") as json_file:
                    json.dump(generated_data, json_file, indent=4)

        print(
            f"Processed {batch_start + n_in_batch}/{len(samples)} "
            f"batch_size={n_in_batch} time={time_elapsed_batch:.2f}s"
        )

    # Final flush for the remainder that did not hit a saving_freq boundary.
    if input_counter % saving_freq != 0:
        if last_saved_path and os.path.isfile(last_saved_path):
            os.remove(last_saved_path)
        with open(output_name + str(input_counter) + ".json", "w") as json_file:
            json.dump(generated_data, json_file, indent=4)

    elapsed_time = time.time() - start_time
    print(f"Total time taken: {elapsed_time} seconds")
    if input_counter:
        print(f"Total time taken per data: {elapsed_time / input_counter} seconds")


if __name__ == "__main__":
    fire.Fire(run)
