
import json
import os
import resource
import sys
import time
from typing import Any, Dict, List, Optional


def _raise_virtual_memory_limit() -> None:
    """Avoid libtorch_cuda.so mmap failures under tiny Slurm ulimit -v."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        target = hard if hard != resource.RLIM_INFINITY else resource.RLIM_INFINITY
        if soft != resource.RLIM_INFINITY and (
            hard == resource.RLIM_INFINITY or soft < hard
        ):
            resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    except (ValueError, OSError):
        pass


_raise_virtual_memory_limit()

# Must be set before vLLM builds its engine so the in-process sampler swap works.
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# vLLM v1 defaults to FlashInfer sampling; JIT needs nvcc + CUDA_HOME.
# Disable so startup works without a full toolkit (override with =1 if available).
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import fire
from transformers import AutoTokenizer

# Allow running as a plain script from anywhere in the repo.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SWIFT_MED = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "SWIFT_Med")
)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SWIFT_MED)

from vllm import LLM, SamplingParams  # noqa: E402

from medmcqa_prompts import (  # noqa: E402
    MEDMCQA_DATASET_ID,
    MEDMCQA_DISK_SUBDIR,
    extract_output_answer,
    load_medmcqa_examples,
    render_chat_prompt,
)
from vllm_watermark.core import (  # noqa: E402
    DetectionAlgorithm,
    WatermarkedLLMs,
    WatermarkingAlgorithm,
)
from vllm_watermark.watermark_detectors import WatermarkDetectors  # noqa: E402

DEFAULT_CACHE_DIR = "Your Cache Directory"


def _resolve_local_model_path(model_id: str, cache_dir: str) -> str:
    """
    Prefer an on-disk HF snapshot under cache_dir when available.

    Passing a Hub id (e.g. google/gemma-3-27b-it) still triggers online
    ``model_info`` calls in newer transformers (Mistral regex patch), which
    can 429 even when weights are fully cached. A local snapshot path avoids that.
    """
    if os.path.isdir(model_id):
        path = model_id
    else:
        folder = "models--" + model_id.replace("/", "--")
        snaps_root = os.path.join(cache_dir, folder, "snapshots")
        if not os.path.isdir(snaps_root):
            snaps_root = os.path.join(cache_dir, "hub", folder, "snapshots")
        if not os.path.isdir(snaps_root):
            return model_id
        candidates = [
            os.path.join(snaps_root, d)
            for d in os.listdir(snaps_root)
            if os.path.isdir(os.path.join(snaps_root, d))
        ]
        if not candidates:
            return model_id
        path = None
        for cand in sorted(candidates):
            if os.path.isfile(os.path.join(cand, "config.json")):
                path = cand
                break
        if path is None:
            path = sorted(candidates)[0]

    _ensure_gemma3_processor_files(path)
    return path


def _ensure_gemma3_processor_files(model_path: str) -> None:
    """Write missing Gemma3 multimodal processor configs for offline vLLM load."""
    if not os.path.isdir(model_path):
        return
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if cfg.get("model_type") != "gemma3" and "Gemma3" not in str(
        cfg.get("architectures", [])
    ):
        return

    vision = cfg.get("vision_config") or {}
    image_size = int(vision.get("image_size", 896) or 896)
    image_seq_length = int(cfg.get("mm_tokens_per_image", 256) or 256)

    pre_path = os.path.join(model_path, "preprocessor_config.json")
    if not os.path.isfile(pre_path):
        pre = {
            "do_convert_rgb": None,
            "do_normalize": True,
            "do_pan_and_scan": None,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_processor_type": "Gemma3ImageProcessor",
            "image_seq_length": image_seq_length,
            "image_std": [0.5, 0.5, 0.5],
            "pan_and_scan_max_num_crops": None,
            "pan_and_scan_min_crop_size": None,
            "pan_and_scan_min_ratio_to_activate": None,
            "processor_class": "Gemma3Processor",
            "resample": 2,
            "rescale_factor": 0.00392156862745098,
            "size": {"height": image_size, "width": image_size},
        }
        with open(pre_path, "w", encoding="utf-8") as f:
            json.dump(pre, f, indent=2)
            f.write("\n")
        print(f"[offline] wrote missing {pre_path}")

    proc_path = os.path.join(model_path, "processor_config.json")
    if not os.path.isfile(proc_path):
        proc = {
            "image_seq_length": image_seq_length,
            "processor_class": "Gemma3Processor",
        }
        with open(proc_path, "w", encoding="utf-8") as f:
            json.dump(proc, f, indent=2)
            f.write("\n")
        print(f"[offline] wrote missing {proc_path}")

    jinja_path = os.path.join(model_path, "chat_template.jinja")
    tok_cfg_path = os.path.join(model_path, "tokenizer_config.json")
    if (not os.path.isfile(jinja_path)) and os.path.isfile(tok_cfg_path):
        try:
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            ct = tok_cfg.get("chat_template")
            if isinstance(ct, str) and ct.strip():
                with open(jinja_path, "w", encoding="utf-8") as f:
                    f.write(ct)
                print(f"[offline] wrote missing {jinja_path}")
        except (OSError, json.JSONDecodeError):
            pass


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


def _safe_output_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def run(
    # --- data / model ---
    cache_dir: str = DEFAULT_CACHE_DIR,
    dataset: str = MEDMCQA_DATASET_ID,
    dataset_split: str = "validation",
    base_model: str = "google/gemma-3-27b-it",
    output_name: str = "",
    split_label: str = "Test",
    start_index: int = 0,
    Ninputs: int = 1000,
    batch_size: int = 32,
    max_out: int = 200,
    min_out: int = 0,
    saving_freq: int = 20,
    # --- KGW watermark parameters ---
    seed: int = 42,
    ngram: int = 2,
    gamma: float = 0.5,
    delta: float = 2.0,
    # --- sampling ---
    temperature: float = 0.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.001,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = True,
    dtype: str = "bfloat16",
    max_model_len: int = 8192,
    # --- optional detection (adds extra keys to each record when True) ---
    detect: bool = False,
    detection_threshold: float = 0.05,
):
    """Generate KGW-watermarked MedMCQA answer+explanation traces with Gemma3.

    Args:
        cache_dir: HF cache / vLLM download dir (sets ``HF_HOME``).
        dataset: HF dataset id (default ``openlifescienceai/medmcqa``).
        dataset_split: MedMCQA split (default ``validation``; pads from train).
        base_model: HF model id to serve with vLLM.
        output_name: Prefix for JSON checkpoint files. Empty = auto name.
        split_label: Label used in the auto output filename (``Test`` / ``Train``).
        start_index: Offset into the loaded MedMCQA examples.
        Ninputs: Number of examples to process starting at ``start_index``.
        batch_size: Prompts handed to each ``generate()`` call (default 32).
        max_out: Max new tokens per answer+explanation (default 200).
        min_out: Minimum new tokens before EOS (0 disables).
        saving_freq: Save a checkpoint every this many processed examples.
        seed, ngram, gamma, delta: KGW parameters (shared with the detector).
        temperature, top_p: Sampling parameters (0 = greedy).
        frequency_penalty: Tiny penalty so vLLM V1 exposes token ids to the
            watermark sampler (set 0.0 to disable if not needed).
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        enforce_eager: Disable CUDA graph capture.
        dtype: vLLM weight dtype (Gemma3 checkpoints are bfloat16).
        max_model_len: Cap KV-cache context (default 8192 for 5-shot MedMCQA).
        detect: If True, score each row and add p-value / flag keys.
        detection_threshold: P-value threshold for the watermark flag.
    """
    os.environ["HF_HOME"] = cache_dir
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model_path = _resolve_local_model_path(base_model, cache_dir)

    if not output_name:
        model_slug = _safe_output_slug(base_model)
        output_name = (
            f"{split_label}_MedMCQA_Gemma_KGW_{model_slug}"
            f"_n{Ninputs}_new{max_out}_batch{batch_size}_"
        )

    print("=" * 60)
    print("BATCH KGW (Maryland) MEDMCQA WATERMARK (Gemma3)")
    print("=" * 60)
    print(f"dataset:      {dataset} split={dataset_split}")
    print(f"model:        {base_model}")
    print(f"model_path:   {model_path}")
    print(f"output_name:  {output_name}")
    print(f"range:        [{start_index}, {start_index + Ninputs}) batch_size={batch_size}")
    min_out = max(0, min(int(min_out), int(max_out)))
    print(
        f"tokens:       min_out={min_out} max_out={max_out} max_model_len={max_model_len}"
    )
    print(f"watermark:    seed={seed} ngram={ngram} gamma={gamma} delta={delta}")
    print(f"sampling:     temperature={temperature} top_p={top_p} dtype={dtype}")
    print("=" * 60)

    need = int(start_index) + int(Ninputs)
    examples = load_medmcqa_examples(
        cache_dir,
        n_inputs=need,
        split=dataset_split,
        dataset_id=dataset,
    )
    examples = examples[start_index : start_index + Ninputs]
    if not examples:
        raise ValueError(
            f"No MedMCQA examples for split={dataset_split!r} "
            f"start_index={start_index} Ninputs={Ninputs}."
        )
    print(
        f"Loaded {len(examples)} MedMCQA examples "
        f"(on-disk under {cache_dir}/{MEDMCQA_DISK_SUBDIR})"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Base vLLM model, then wrap with KGW watermarking.
    llm = LLM(
        model=model_path,
        download_dir=cache_dir,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        max_num_seqs=max(1, int(batch_size)),
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
    if min_out > 0:
        sampling_kwargs["min_tokens"] = min_out
    if frequency_penalty:
        sampling_kwargs["frequency_penalty"] = frequency_penalty
    sampling_params = SamplingParams(**sampling_kwargs)

    generated_data: List[Dict[str, Any]] = []
    input_counter = 0
    last_saved_path: Optional[str] = None
    start_time = time.time()

    for batch_start in range(0, len(examples), batch_size):
        start_time_batch = time.time()

        batch_examples = examples[batch_start : batch_start + batch_size]
        if not batch_examples:
            break

        prompts = [
            render_chat_prompt(tokenizer, ex["prompt"]) for ex in batch_examples
        ]

        outputs = wm_llm.generate(prompts, sampling_params=sampling_params)

        time_elapsed_batch = time.time() - start_time_batch
        n_in_batch = len(batch_examples)
        time_per_item = time_elapsed_batch / n_in_batch

        for local_idx in range(n_in_batch):
            ex = batch_examples[local_idx]
            watermarked = (
                outputs[local_idx].outputs[0].text if outputs[local_idx].outputs else ""
            )

            data_dict: Dict[str, Any] = {
                "question": ex["question"],
                "answer": ex["answer"],
                "final_answer": ex["final_answer"],
                "Watermarked_output": watermarked,
                "Watermarked_output_answer": extract_output_answer(watermarked),
                "time": time_per_item,
            }

            if detector is not None and watermarked:
                result = detector.detect(watermarked)
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
            f"Processed {batch_start + n_in_batch}/{len(examples)} "
            f"batch_size={n_in_batch} time={time_elapsed_batch:.2f}s"
        )

    if input_counter % saving_freq != 0:
        if last_saved_path and os.path.isfile(last_saved_path):
            os.remove(last_saved_path)
        with open(output_name + str(input_counter) + ".json", "w") as json_file:
            json.dump(generated_data, json_file, indent=4)

    elapsed_time = time.time() - start_time
    print(f"Total time taken: {elapsed_time} seconds")
    if input_counter:
        print(f"Total time taken per data: {elapsed_time / input_counter} seconds")
        print(f"Results prefix: {output_name}{input_counter}.json")


if __name__ == "__main__":
    fire.Fire(run)
