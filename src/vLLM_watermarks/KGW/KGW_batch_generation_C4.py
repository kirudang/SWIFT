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


def _patch_gemma3_config_for_vllm(config: Any) -> Any:
    """Alias Gemma3 sliding-window fields expected by vLLM 0.10.

    Newer Hugging Face configs expose ``sliding_window`` / ``layer_types``, while
    vLLM 0.10 still reads ``interleaved_sliding_window``.
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
    start_index: int = 0,
    Ninputs: int = 1000,
    N_start_index: int = 0,
    batch_size: int = 32,
    max_inp: int = 50,
    max_out: int = 200,
    output_name: Optional[str] = None,
    saving_freq: int = 20,
    # --- KGW watermark parameters ---
    seed: int = 42,
    ngram: int = 2,
    gamma: float = 0.5,
    delta: float = 2.0,
    # --- sampling ---
    temperature: float = 0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.001,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = True,
    # Gemma-3-27B is multimodal with a huge default context; cap for text generation.
    dtype: str = "bfloat16",
    max_model_len: Optional[int] = None,
    # --- optional detection (adds extra keys to each record when True) ---
    detect: bool = False,
    detection_threshold: float = 0.05,
):
    """Generate KGW-watermarked continuations for a ``.pt`` prompt dataset.

    Args:
        data_path: Path to a torch-saved object whose ``data[0]`` is a list of
            prompt strings (same layout as the no-watermark script).
        cache_dir: HF cache / vLLM download dir (sets ``HF_HOME``).
        base_model: HF model id to serve with vLLM.
        output_name: Prefix for the JSON checkpoint files.
        start_index: Extra offset into ``data[0]`` (usually leave at 0).
        N_start_index: Dataset index to start from (e.g. 50000 to continue
            after a previous 0-50k run).
        Ninputs: Number of prompts to process starting at ``N_start_index``.
            Defaults generate ``[50000, 100000)``.
        batch_size: Prompts handed to each ``generate()`` call.
        max_inp: Max prompt tokens (HF truncation, then decoded back to text).
        max_out: Max new tokens to generate.
        saving_freq: Save a checkpoint every this many processed prompts.
        seed, ngram, gamma, delta: KGW parameters (shared with the detector).
        temperature, top_p: Sampling parameters (any temperature; 0 = greedy).
        frequency_penalty: Tiny penalty so vLLM V1 exposes token ids to the
            watermark sampler (set 0.0 to disable if not needed).
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        enforce_eager: Disable CUDA graph capture.
        dtype: vLLM weight dtype (Gemma3 checkpoints are bfloat16).
        max_model_len: Cap KV-cache context. Defaults to
            ``max_inp + max_out + 64`` so Gemma's huge native context does not OOM.
        detect: If True, score each row and add p-value / flag keys.
        detection_threshold: P-value threshold for the watermark flag.
    """
    # HF_HOME is read lazily at from_pretrained / LLM() time, so setting it here
    # (before those calls) is sufficient.
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_TOKEN"] = "Your HF Token"

    if output_name is None:
        output_name = (
            f"Gemma3_27B_kgw_wm_test_batched_1k_batch_{batch_size}"
            f"_max_inp_{max_inp}_max_out_{max_out}"
            f"_from_{N_start_index}"
        )

    if max_model_len is None:
        max_model_len = max_inp + max_out + 64

    print("=" * 60)
    print("BATCH KGW (Maryland) WATERMARK GENERATION (Gemma3)")
    print("=" * 60)
    print(f"data:         {data_path}")
    print(f"model:        {base_model}")
    print(f"output_name:  {output_name}")
    abs_start = start_index + N_start_index
    abs_end = abs_start + Ninputs
    print(f"range:        [{abs_start}, {abs_end}) batch_size={batch_size}")
    print(f"tokens:       max_inp={max_inp} max_out={max_out} max_model_len={max_model_len}")
    print(f"watermark:    seed={seed} ngram={ngram} gamma={gamma} delta={delta}")
    print(f"sampling:     temperature={temperature} top_p={top_p} dtype={dtype}")
    print("=" * 60)

    # Tokenizer (used for prompt truncation, same as the no-watermark script).
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Base vLLM model, then wrap with KGW watermarking.
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
        algo=WatermarkingAlgorithm.MARYLAND,
        seed=seed,
        ngram=ngram,
        gamma=gamma,
        delta=delta,
    )

    detector = None
    if detect:
        detector = WatermarkDetectors.create(
            algo=DetectionAlgorithm.MARYLAND_Z,
            model=llm,
            ngram=ngram,
            seed=seed,
            gamma=gamma,
            delta=delta,
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

    for batch_start in range(N_start_index, N_start_index + Ninputs, batch_size):
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
            f"Processed {batch_start + n_in_batch}/{N_start_index + Ninputs} "
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
