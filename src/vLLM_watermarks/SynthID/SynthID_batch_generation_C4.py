#!/usr/bin/env python3
"""
Batch SynthID watermark generation over a dataset.

Mirrors the data loading / batching / saving of ``1_No_WM_vLLM_batched.py``
(a ``.pt`` file whose ``data[0]`` is a list of prompt strings, HF-tokenizer
truncation to ``max_inp`` tokens, frequency-based JSON checkpoints) but replaces
plain vLLM generation with Google SynthID tournament watermarking.

How SynthID plugs into vLLM (same mechanism as KGW)
---------------------------------------------------
* ``WatermarkedLLMs.create(..., SYNTHID)`` swaps vLLM's sampler for the shared
  ``WatermarkSampler``. Each decode step it hands the ``(batch, vocab)`` logits
  to ``SynthIDGenerator.sample_next()``.
* SynthID is already batch-aware: it computes per-row g-values, applies the
  non-distortionary tournament update, and samples all rows together. Passing a
  whole mini-batch to ``generate()`` is the efficient path.
* ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` is required so the in-process sampler
  swap is visible (this replaces the ``spawn`` multiprocessing of the no-WM
  script).

Run
---
    python scripts/generation/batch_generate_synthid_gemma.py
    python scripts/generation/batch_generate_synthid_gemma.py --start_index 0 --Ninputs 1000
"""

import json
import os
import sys
import time
from typing import Any, Optional

# Must be set before vLLM builds its engine so the in-process sampler swap works.
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import fire
import torch
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


def run(
    # --- data / model (same as 1_No_WM_vLLM_batched.py) ---
    data_path: str = "Your Data Path",
    cache_dir: str = "Your Cache Directory",
    base_model: str = "google/gemma-3-27b-it",
    output_name: Optional[str] = None,
    start_index: int = 0,
    Ninputs: int = 1000,
    batch_size: int = 32,
    max_inp: int = 50,
    max_out: int = 200,
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
    """Generate SynthID-watermarked continuations for a ``.pt`` prompt dataset.

    Args:
        data_path: Path to a torch-saved object whose ``data[0]`` is a list of
            prompt strings (same layout as the no-watermark script).
        cache_dir: HF cache / vLLM download dir (sets ``HF_HOME``).
        base_model: HF model id to serve with vLLM.
        output_name: Prefix for the JSON checkpoint files.
        start_index: Offset into ``data[0]`` to start from.
        Ninputs: Number of prompts to process starting at ``start_index``.
        batch_size: Prompts handed to each ``generate()`` call.
        max_inp: Max prompt tokens (HF truncation, then decoded back to text).
        max_out: Max new tokens to generate.
        saving_freq: Save a checkpoint every this many processed prompts.
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
            ``max_inp + max_out + 64`` so Gemma's huge native context does not
            OOM. Pass an explicit int to override.
        detect: If True, score each row and add g-value / flag keys.
        detection_threshold: Mean g-value threshold for the watermark flag.
    """
    # HF_HOME is read lazily at from_pretrained / LLM() time, so setting it here
    # (before those calls) is sufficient.
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_TOKEN"] = "Your HF Token"

    if output_name is None:
        output_name = (
        f"Gemma3_27B_synthid_wm_test_batch_{batch_size}"
        f"_max_inp_{max_inp}_max_out_{max_out}"
        )

    if max_model_len is None:
        max_model_len = max_inp + max_out + 64

    if keys is None:
        keys = list(DEFAULT_SYNTHID_KEYS)
    elif isinstance(keys, (tuple, list)):
        keys = [int(k) for k in keys]
    else:
        raise TypeError(f"keys must be a list of ints, got {type(keys)}")

    print("=" * 60)
    print("BATCH SYNTHID WATERMARK GENERATION (Gemma3)")
    print("=" * 60)
    print(f"data:         {data_path}")
    print(f"model:        {base_model}")
    print(f"output_name:  {output_name}")
    print(f"range:        [{start_index}, {start_index + Ninputs}) batch_size={batch_size}")
    print(f"tokens:       max_inp={max_inp} max_out={max_out} max_model_len={max_model_len}")
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

    # Tokenizer (used for prompt truncation, same as the no-watermark script).
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
    if frequency_penalty:
        sampling_kwargs["frequency_penalty"] = frequency_penalty
    sampling_params = SamplingParams(**sampling_kwargs)

    # Load data (data[0] is the list/tensor of prompt strings).
    data = torch.load(data_path)

    generated_data = []
    input_counter = 0
    last_saved_path: Optional[str] = None
    start_time = time.time()

    for batch_start in range(0, Ninputs, batch_size):
        start_time_batch = time.time()

        batch_texts = data[0][
            start_index + batch_start : start_index + batch_start + batch_size
        ]
        batch_texts = list(batch_texts)
        if not batch_texts:
            break

        # Truncate each prompt to max_inp tokens (HF), then decode back to text.
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_inp,
        )
        generated_prompts = tokenizer.batch_decode(
            encoded["input_ids"], skip_special_tokens=True
        )

        # One batched, watermarked generate call: outputs stay aligned by index.
        outputs = wm_llm.generate(generated_prompts, sampling_params=sampling_params)

        time_elapsed_batch = time.time() - start_time_batch
        n_in_batch = len(batch_texts)
        time_per_item = time_elapsed_batch / n_in_batch

        for local_idx in range(n_in_batch):
            generated_prompt = generated_prompts[local_idx]
            generated_output_text = outputs[local_idx].outputs[0].text
            generated_text = generated_prompt + generated_output_text

            data_dict = {
                "input": generated_prompt,
                "input_output": generated_text,
                "output_only": generated_output_text,
                "time": time_per_item,
            }

            if detector is not None and generated_output_text:
                result = detector.detect(generated_output_text)
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
            f"Processed {batch_start + n_in_batch}/{Ninputs} "
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
    print(output_name)    

if __name__ == "__main__":
    fire.Fire(run)
