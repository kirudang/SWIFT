import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
# Ensure the HF_HOME environment variable points to your desired cache location
cache_dir = '/network/rit/lab/Lai_ReSecureAI/kiel/wmm'
os.environ["HF_HOME"] = cache_dir

# Force vLLM workers to use spawn (not fork)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import asyncio
import uuid
import queue as queue_module
from transformers import AutoTokenizer
import torch
import argparse
import time
import re
import warnings
from utils import (
    whole_context_process_sentences_batch,
    is_valid_sentence,
    tokenize_with_spans,
)
from typing import Optional, List, Tuple, Dict, Any, Union, Iterable

# Watermark job id: (prompt_index, sentence_index_within_prompt)
WMQueueKey = Tuple[int, int]
# Reserved prompt_id for pipeline warmup (must not collide with 0..N-1)
WARMUP_PROMPT_ID = -1


def _safe_output_slug(model_id: str) -> str:
    """HF hub ids contain '/'; strip for local filenames."""
    return model_id.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def _prepend_conda_cuda_lib_path() -> None:
    """
    FlashInfer JIT links with -L$CONDA_PREFIX/lib64 while conda cuda-cudart installs
    under lib/ (and stubs under lib/stubs or targets/.../lib/stubs). Prepend those
    dirs to LIBRARY_PATH / LD_LIBRARY_PATH so vLLM EngineCore subprocesses can link
    -lcudart / -lcuda when nvcc is present.
    """
    pfx = os.environ.get("CONDA_PREFIX")
    if not pfx:
        return
    candidates: List[str] = []
    for rel in (
        ("lib",),
        ("lib64",),
        ("lib", "stubs"),
        ("lib64", "stubs"),
    ):
        d = os.path.join(pfx, *rel)
        if os.path.isdir(d):
            candidates.append(d)
    for arch in ("x86_64-linux", "sbsa-linux"):
        tlib = os.path.join(pfx, "targets", arch, "lib")
        if os.path.isdir(tlib):
            candidates.append(tlib)
            tst = os.path.join(tlib, "stubs")
            if os.path.isdir(tst):
                candidates.append(tst)
    if not candidates:
        return
    seen: set[str] = set()
    ordered: List[str] = []
    for d in candidates:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    blob = ":".join(ordered)
    for key in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        cur = os.environ.get(key, "")
        os.environ[key] = blob + (":" + cur if cur else "")


def _ensure_conda_lib64_cuda_symlinks() -> None:
    """
    FlashInfer's ninja link uses only -L$CONDA_PREFIX/lib64 and lib64/stubs; conda
    often installs libcudart under lib/. EngineCore may not inherit LIBRARY_PATH.
    Add relative symlinks under lib64 so -lcudart / -lcuda resolve.
    """
    pfx = os.environ.get("CONDA_PREFIX")
    if not pfx:
        return
    lib_dir = os.path.join(pfx, "lib")
    lib64_dir = os.path.join(pfx, "lib64")
    if not os.path.isdir(lib_dir):
        return
    try:
        os.makedirs(lib64_dir, exist_ok=True)
    except OSError:
        return

    def _link(src_abs: str, dst_abs: str) -> None:
        if not (os.path.isfile(src_abs) or os.path.islink(src_abs)):
            return
        rel = os.path.relpath(src_abs, start=os.path.dirname(dst_abs))
        if os.path.lexists(dst_abs):
            if os.path.islink(dst_abs):
                try:
                    cur = os.path.normpath(
                        os.path.join(os.path.dirname(dst_abs), os.readlink(dst_abs))
                    )
                    if cur == os.path.normpath(src_abs):
                        return
                    os.unlink(dst_abs)
                except OSError:
                    return
            else:
                return
        try:
            os.symlink(rel, dst_abs)
        except OSError:
            return

    for ent in sorted(os.listdir(lib_dir)):
        if not (ent == "libcudart.so" or ent.startswith("libcudart.so.")):
            continue
        _link(os.path.join(lib_dir, ent), os.path.join(lib64_dir, ent))

    stub_src: Optional[str] = None
    for arch in ("x86_64-linux", "sbsa-linux"):
        c = os.path.join(pfx, "targets", arch, "lib", "stubs", "libcuda.so")
        if os.path.isfile(c) or os.path.islink(c):
            stub_src = c
            break
    if stub_src is None:
        c = os.path.join(pfx, "lib", "stubs", "libcuda.so")
        if os.path.isfile(c) or os.path.islink(c):
            stub_src = c
    if stub_src:
        stubs64 = os.path.join(lib64_dir, "stubs")
        try:
            os.makedirs(stubs64, exist_ok=True)
        except OSError:
            return
        _link(stub_src, os.path.join(stubs64, "libcuda.so"))


def _setup_conda_cuda_for_flashinfer_link() -> None:
    _ensure_conda_lib64_cuda_symlinks()
    _prepend_conda_cuda_lib_path()


def _mp_queue_qsize(q: Any) -> Optional[int]:
    """Best-effort queue depth (Linux: usually reliable; macOS: may be absent)."""
    try:
        return int(q.qsize())  # type: ignore[attr-defined]
    except Exception:
        return None


def _log_wm_queue_depths(
    enabled: bool,
    label: str,
    wm_in_q: Optional["mp.Queue"],
    wm_out_q: Optional["mp.Queue"],
    extra: str = "",
) -> None:
    if not enabled:
        return
    gin = _mp_queue_qsize(wm_in_q) if wm_in_q is not None else None
    gout = _mp_queue_qsize(wm_out_q) if wm_out_q is not None else None
    parts = [label]
    if gin is not None:
        parts.append(f"gen→wm_job_batches≈{gin}")
    if gout is not None:
        parts.append(f"wm→main_job_batches≈{gout}")
    if extra:
        parts.append(extra)
    print("[wm-queue] " + " | ".join(parts))


class PromptWMState:
    """
    Per-prompt watermark assembly (ordered sentence stitching + sampling log).
    Used when multiple generation streams share one watermark worker queue.
    """

    __slots__ = (
        "wm_by_idx",
        "watermarked_parts",
        "next_emit",
        "sent_idx_final",
        "producer_done",
        "sampling_flat",
        "done_event",
    )

    def __init__(self) -> None:
        self.wm_by_idx: Dict[int, str] = {}
        self.watermarked_parts: List[str] = []
        self.next_emit: int = 0
        self.sent_idx_final: int = 0
        self.producer_done: bool = False
        self.sampling_flat: List[Dict[str, Any]] = []
        self.done_event = asyncio.Event()

    def apply_result(self, sent_idx: int, wm_sent: str, samp: List[Dict[str, Any]]) -> None:
        self.wm_by_idx[sent_idx] = wm_sent
        if samp:
            self.sampling_flat.extend(samp)
        while self.next_emit in self.wm_by_idx:
            self.watermarked_parts.append(self.wm_by_idx.pop(self.next_emit))
            self.next_emit += 1
        self._try_mark_done()

    def _try_mark_done(self) -> None:
        if self.producer_done and self.next_emit >= self.sent_idx_final:
            self.done_event.set()

    def mark_producer_finished(self, total_sentences: int) -> None:
        self.sent_idx_final = total_sentences
        self.producer_done = True
        self._try_mark_done()

def _import_vllm():
    """
    Import vLLM lazily so each process can set CUDA_VISIBLE_DEVICES first.
    """
    try:
        from vllm import LLM, SamplingParams  # type: ignore
        return LLM, SamplingParams
    except Exception as e:
        raise RuntimeError(f"Failed to import vLLM: {e}")


def _import_async_vllm():
    """
    Prefer vLLM v1 AsyncLLM for true streaming generation (GPU 0).
    Returns (AsyncLLM class, AsyncEngineArgs, SamplingParams, RequestOutputKind) or None if unavailable.
    """
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
        from vllm.sampling_params import RequestOutputKind  # type: ignore
        from vllm.v1.engine.async_llm import AsyncLLM  # type: ignore

        return AsyncLLM, AsyncEngineArgs, RequestOutputKind
    except Exception:
        return None


def _make_gen_sampling_params(args: argparse.Namespace, *, max_tokens: Optional[int] = None) -> Any:
    """
    vLLM SamplingParams defaults temperature=1.0 (sampling). This script expects greedy
    decode unless --gen-temperature is set, matching historical float16 batching runs.
    """
    _, SamplingParams = _import_vllm()
    mt = int(args.max_new_tokens) if max_tokens is None else int(max_tokens)
    return SamplingParams(
        max_tokens=mt,
        temperature=float(args.gen_temperature),
        top_p=float(args.gen_top_p),
    )


_SCHEDULING_BACKOFF_KEYS = (
    "enable_prefix_caching",
    "scheduling_policy",
    "enable_chunked_prefill",
)


def _normalize_kv_cache_dtype(kv_dtype: str) -> str:
    """
    vLLM 0.12+ CacheConfig.cache_dtype only allows a small Literal set; older
    values like int8_per_token_head were removed (use FP8 KV dtypes instead).
    """
    s = (kv_dtype or "auto").strip().lower()
    legacy = {
        "int8_per_token_head": "fp8_e4m3",
        "int8": "fp8_e4m3",
    }
    if s in legacy:
        repl = legacy[s]
        warnings.warn(
            f"kv_cache_dtype {kv_dtype!r} is not supported in this vLLM; "
            f"using {repl!r} instead. Pick an explicit dtype from --help if needed.",
            UserWarning,
            stacklevel=2,
        )
        s = repl
    try:
        from typing import get_args

        from vllm.config.cache import CacheDType

        valid = frozenset(get_args(CacheDType))
    except Exception:
        valid = frozenset(
            {
                "auto",
                "bfloat16",
                "fp8",
                "fp8_e4m3",
                "fp8_e5m2",
                "fp8_inc",
                "fp8_ds_mla",
            }
        )
    if s not in valid:
        raise ValueError(
            f"Invalid kv_cache_dtype {kv_dtype!r}; must be one of {sorted(valid)}."
        )
    return s


def _iter_scheduling_backoff(scheduling: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Yield subsets of scheduling-related kwargs (largest first) so callers can try
    LLM(..., **subset) and fall back on TypeError when an older vLLM lacks a flag.
    """
    for mask in sorted(range(8), key=lambda m: (-bin(m).count("1"), -m)):
        extra: Dict[str, Any] = {}
        for i, key in enumerate(_SCHEDULING_BACKOFF_KEYS):
            if mask & (1 << i):
                extra[key] = scheduling[key]
        yield extra


def _vllm_engine_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Picklable dict for gen + watermark worker processes.

    - ``quantization`` (bool): KV-cache options only (not vLLM weight INT4).
    - ``vllm_model_quantization`` (str | None): vLLM ``LLM(quantization=...)`` for weights.
    """
    q_on = bool(getattr(args, "quantization", False))
    if q_on:
        kv_dtype = _normalize_kv_cache_dtype(getattr(args, "kv_cache_dtype", "auto") or "auto")
        calc_kv = bool(getattr(args, "calculate_kv_scales", False))
    else:
        kv_dtype = "auto"
        calc_kv = False
    wq = getattr(args, "vllm_model_quantization", None)
    if isinstance(wq, str):
        wq = wq.strip() or None

    mbt = getattr(args, "max_num_batched_tokens", None)
    if mbt is not None:
        mbt = int(mbt)

    return {
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "dtype": str(args.dtype),
        "trust_remote_code": bool(args.trust_remote_code),
        "quantization": q_on,
        "kv_cache_dtype": kv_dtype,
        "calculate_kv_scales": calc_kv,
        "vllm_model_quantization": wq,
        "enable_prefix_caching": bool(args.enable_prefix_caching),
        "scheduling_policy": str(args.scheduling_policy),
        "enable_chunked_prefill": bool(args.enable_chunked_prefill),
        "max_num_batched_tokens": mbt,
    }


def _vllm_common_kwargs(model_name: str, download_dir: str, engine_cfg: Dict[str, Any]) -> Dict[str, Any]:
    common: Dict[str, Any] = dict(
        model=model_name,
        tokenizer=model_name,
        tensor_parallel_size=engine_cfg["tensor_parallel_size"],
        dtype=engine_cfg["dtype"],
        trust_remote_code=engine_cfg["trust_remote_code"],
        download_dir=download_dir,
    )
    wq = engine_cfg.get("vllm_model_quantization")
    if wq:
        common["quantization"] = wq
    if engine_cfg.get("quantization"):
        kv_dtype = _normalize_kv_cache_dtype(engine_cfg.get("kv_cache_dtype") or "auto")
        weight_dt = str(engine_cfg.get("dtype") or "").strip().lower()
        if kv_dtype == weight_dt or kv_dtype in ("", "same", "match"):
            kv_dtype = "auto"
        if kv_dtype != "auto":
            common["kv_cache_dtype"] = kv_dtype
        common["calculate_kv_scales"] = bool(engine_cfg.get("calculate_kv_scales", False))
    mbt = engine_cfg.get("max_num_batched_tokens")
    if mbt is not None:
        common["max_num_batched_tokens"] = int(mbt)
    return common


def _scheduling_kwargs(engine_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enable_prefix_caching": engine_cfg["enable_prefix_caching"],
        "scheduling_policy": engine_cfg["scheduling_policy"],
        "enable_chunked_prefill": engine_cfg["enable_chunked_prefill"],
    }


def _build_llm_and_tokenizer(
    model_name: str,
    download_dir: str,
    *,
    engine_cfg: Dict[str, Any],
    max_num_seqs: Optional[int] = None,
):
    """
    Build a vLLM engine + HF tokenizer in the current process.
    ``max_num_seqs`` is the vLLM scheduler cap on concurrent sequences (continuous batching).
    """
    LLM, SamplingParams = _import_vllm()
    common = _vllm_common_kwargs(model_name, download_dir, engine_cfg)
    if max_num_seqs is not None:
        common["max_num_seqs"] = int(max_num_seqs)
    scheduling = _scheduling_kwargs(engine_cfg)
    last_err: Optional[TypeError] = None
    llm = None

    def _try_all_scheduling(c: Dict[str, Any]) -> Any:
        nonlocal last_err
        for extra in _iter_scheduling_backoff(scheduling):
            try:
                return LLM(**c, **extra)
            except TypeError as e:
                last_err = e
        return None

    llm = _try_all_scheduling(common)
    if llm is None and "max_num_seqs" in common:
        common.pop("max_num_seqs", None)
        llm = _try_all_scheduling(common)
    if llm is None and "max_num_batched_tokens" in common:
        common.pop("max_num_batched_tokens", None)
        llm = _try_all_scheduling(common)
    if llm is None:
        raise RuntimeError("Failed to construct vLLM LLM with any supported optional kwargs") from last_err
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=download_dir)
    return llm, tokenizer, SamplingParams


def _build_async_gen_llm(
    model_name: str,
    download_dir: str,
    *,
    engine_cfg: Dict[str, Any],
    max_num_seqs: Optional[int] = None,
):
    """
    Async generation engine on the current process GPU.
    ``max_num_seqs`` caps concurrent ``generate()`` streams; with ``asyncio.Semaphore(gen_max_inflight)``
    this matches vLLM continuous batching (dynamic interleaving of prefills/decodes).
    """
    imported = _import_async_vllm()
    if imported is None:
        return None
    AsyncLLM, AsyncEngineArgs, RequestOutputKind = imported
    from vllm import SamplingParams  # type: ignore

    common = _vllm_common_kwargs(model_name, download_dir, engine_cfg)
    if max_num_seqs is not None:
        common["max_num_seqs"] = int(max_num_seqs)
    scheduling = _scheduling_kwargs(engine_cfg)
    last_err: Optional[TypeError] = None
    engine_args = None

    def _try_all_engine_args(c: Dict[str, Any]) -> Any:
        nonlocal last_err
        for extra in _iter_scheduling_backoff(scheduling):
            try:
                return AsyncEngineArgs(**c, **extra)
            except TypeError as e:
                last_err = e
        return None

    engine_args = _try_all_engine_args(common)
    if engine_args is None and "max_num_seqs" in common:
        common.pop("max_num_seqs", None)
        engine_args = _try_all_engine_args(common)
    if engine_args is None and "max_num_batched_tokens" in common:
        common.pop("max_num_batched_tokens", None)
        engine_args = _try_all_engine_args(common)
    if engine_args is None:
        raise RuntimeError(
            "Failed to construct vLLM AsyncEngineArgs with any supported optional kwargs"
        ) from last_err

    llm = AsyncLLM.from_engine_args(engine_args)
    return llm, SamplingParams, RequestOutputKind


# Default model (override with --model)
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_SENTENCE_CLOSERS = set("\"'”’)]}")
_SENTENCE_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "e.g.", "i.e.", "etc.", "vs.", "u.s.", "u.k."
}

def apply_replacements(sentence, replacements):
    """
    Apply replacements to the sentence while preserving original formatting, spacing, and punctuation.
    Handles both single-word and phrase replacements (candidates can be phrases).
    
    Args:
        sentence: Original sentence
        replacements: List of (start_pos, end_pos, target_text, replacement) tuples
                     replacement can be a word or phrase and the target can be a span (e.g., "focus on")
    """
    # Use Treebank token to get the start and end positions of the target word
    token_items = tokenize_with_spans(sentence)
    if not token_items:
        return sentence

    updated_sentence = sentence

    # Apply replacements in reverse token order to keep character offsets valid.
    for start_pos, end_pos, target, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        if start_pos < 0 or end_pos < 0:
            continue
        if start_pos >= len(token_items) or end_pos >= len(token_items):
            continue
        if end_pos < start_pos:
            continue

        start_char = token_items[start_pos][1]
        end_char = token_items[end_pos][2]
        span_text = updated_sentence[start_char:end_char].strip()
        if span_text != target and span_text.lower() != str(target).lower():
            continue

        updated_sentence = (
            updated_sentence[:start_char] + replacement + updated_sentence[end_char:]
        )

    return updated_sentence


def _looks_like_abbreviation(text: str, dot_idx: int) -> bool:
    """
    Fast guard to avoid splitting on common abbreviations and initials.
    """
    start = max(0, dot_idx - 8)
    tail = text[start:dot_idx + 1].lower().strip()
    match = re.search(r"([a-z]\.){2,}$|[a-z]{1,5}\.$", tail)
    if not match:
        return False
    token = match.group(0)
    return token in _SENTENCE_ABBREVIATIONS or bool(re.fullmatch(r"([a-z]\.){2,}", token))


def _split_streaming_sentences(buffer: str, new_text: str, is_final: bool) -> Tuple[str, List[str]]:
    """
    Accumulate streaming decoder output and emit completed sentences.
    Sentence boundaries are punctuation-based (., !, ?) rather than newline-based.
    """
    text = buffer + new_text
    if not text:
        return "", []

    emitted: List[str] = []
    start = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch in ".!?":
            if ch == "." and _looks_like_abbreviation(text, i):
                i += 1
                continue

            j = i + 1
            while j < n and text[j] in _SENTENCE_CLOSERS:
                j += 1

            # Sentence boundary when punctuation is followed by whitespace or end-of-buffer.
            if j == n or text[j].isspace():
                k = j
                while k < n and text[k].isspace():
                    k += 1
                # Preserve exact generated formatting by keeping boundary whitespace
                # (spaces/newlines) with the emitted chunk.
                emitted.append(text[start:k])
                start = k
                i = k
                continue
        i += 1

    remainder = text[start:]
    if is_final and remainder:
        emitted.append(remainder)
        remainder = ""

    return remainder, emitted


def _watermark_batch_sentences(
    sentences: List[str],
    llm: Any,
    tokenizer: AutoTokenizer,
    Top_K: int,
    secret_key: str,
    m: int,
    c: int,
    h: int,
    alpha: float,
    output_name: str,
    wm_sentence_batch_size: int,
    *,
    wm_temperature: float = 0.0,
    wm_top_p: float = 1.0,
    wm_detect_max_tokens: Optional[int] = None,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Watermark a batch of sentences in one call to reuse shared prefix/KV cache.
    Returns per-sentence [(watermarked_sentence, sampling_results_for_sentence), ...]
    in the same order as input.
    """
    if not sentences:
        return []

    valid_flags = [is_valid_sentence(s) for s in sentences]
    valid_sentences = [s for s, ok in zip(sentences, valid_flags) if ok]

    batch_results = []
    if valid_sentences:
        batch_results = whole_context_process_sentences_batch(
            valid_sentences,
            llm,
            tokenizer,
            Top_K,
            secret_key,
            m,
            c,
            h,
            alpha,
            output_name,
            batch_size=max(1, wm_sentence_batch_size),
            wm_temperature=wm_temperature,
            wm_top_p=wm_top_p,
            wm_detect_max_tokens=wm_detect_max_tokens,
        )

    out: List[Tuple[str, List[Dict[str, Any]]]] = []
    valid_idx = 0
    for original_sentence, is_valid in zip(sentences, valid_flags):
        if not is_valid:
            out.append((original_sentence, []))
            continue
        replacements, sampling_results_line = batch_results[valid_idx]
        valid_idx += 1
        if not replacements:
            out.append((original_sentence, sampling_results_line))
            continue
        out.append((apply_replacements(original_sentence, replacements), sampling_results_line))
    return out

def _watermark_worker_main(
    gpu_id: int,
    model_name: str,
    download_dir: str,
    engine_cfg: Dict[str, Any],
    Top_K: int,
    secret_key: str,
    m: int,
    c: int,
    h: int,
    alpha: float,
    output_name: str,
    wm_sentence_batch_size: int,
    wm_temperature: float,
    wm_top_p: float,
    wm_detect_max_tokens: Optional[int],
    max_wm_seqs: int,
    in_q: "mp.Queue",
    out_q: "mp.Queue",
):
    """
    Separate process that runs watermarking on a dedicated GPU.
    Receives a batch [((prompt_id, sent_idx), sentence), ...] and returns
    [((prompt_id, sent_idx), watermarked_sentence, sampling_list), ...].
    Batches may mix sentences from different prompts (vLLM continuous batching + shared prefix).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _setup_conda_cuda_for_flashinfer_link()
    llm, tokenizer, _SamplingParams = _build_llm_and_tokenizer(
        model_name, download_dir, engine_cfg=engine_cfg, max_num_seqs=max(1, max_wm_seqs)
    )
    n_dev = torch.cuda.device_count()
    print(
        f"[GPU layout] Watermark worker: physical GPU id={gpu_id}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, "
        f"torch.cuda.device_count()={n_dev}"
    )
    if n_dev > 0:
        print(f"[GPU layout] Watermark worker device name: {torch.cuda.get_device_name(0)}")

    while True:
        item = in_q.get()
        if item is None:
            break
        batch_items = item if isinstance(item, list) else [item]
        if not batch_items:
            continue
        idxs = [idx for idx, _ in batch_items]
        texts = [txt for _, txt in batch_items]
        wm_results = _watermark_batch_sentences(
            texts,
            llm,
            tokenizer,
            Top_K,
            secret_key,
            m,
            c,
            h,
            alpha,
            output_name,
            wm_sentence_batch_size,
            wm_temperature=wm_temperature,
            wm_top_p=wm_top_p,
            wm_detect_max_tokens=wm_detect_max_tokens,
        )
        out_q.put([(idx, wm_sent, samp) for idx, (wm_sent, samp) in zip(idxs, wm_results)])

def _normalize_wm_queue_key(idx: Union[int, WMQueueKey], prompt_id: int = 0) -> WMQueueKey:
    """Legacy int ids become (prompt_id, sent_idx) for single-prompt / sync paths."""
    if isinstance(idx, tuple) and len(idx) == 2:
        return int(idx[0]), int(idx[1])
    return prompt_id, int(idx)


def _apply_drain_batch(
    wm_out_q: "mp.Queue",
    wm_by_idx: Dict[WMQueueKey, str],
    watermarked_parts: List[str],
    sampling_flat: List[Dict[str, Any]],
    next_emit: int,
    prompt_id: int,
    block: bool,
    timeout: float,
) -> Tuple[int, bool]:
    """
    Drain available watermark results for one prompt_id. Returns (next_emit, got_any).
    """
    got_any = False
    while True:
        try:
            if block and timeout > 0:
                payload = wm_out_q.get(block=True, timeout=timeout)
            elif not block:
                payload = wm_out_q.get(block=False)
            else:
                payload = wm_out_q.get(block=True)
        except queue_module.Empty:
            break
        items = payload if isinstance(payload, list) else [payload]
        if not items:
            continue
        got_any = True
        for idx, wm_sent, samp in items:
            key = _normalize_wm_queue_key(idx, prompt_id=prompt_id)
            if key[0] != prompt_id:
                continue
            wm_by_idx[key] = wm_sent
            if samp:
                sampling_flat.extend(samp)
        while (prompt_id, next_emit) in wm_by_idx:
            watermarked_parts.append(wm_by_idx.pop((prompt_id, next_emit)))
            next_emit += 1
    return next_emit, got_any


def _dispatch_wm_payload_to_states(
    items: List[Any],
    states: Dict[int, PromptWMState],
) -> None:
    """Apply one watermark worker payload to the correct per-prompt state."""
    for idx, wm_sent, samp in items:
        if isinstance(idx, tuple) and len(idx) == 2:
            pid, sid = int(idx[0]), int(idx[1])
        else:
            pid, sid = 0, int(idx)
        st = states.get(pid)
        if st is None:
            continue
        st.apply_result(sid, wm_sent, samp if samp else [])


async def _global_watermark_drain_loop(
    wm_out_q: "mp.Queue",
    wm_in_q: Optional["mp.Queue"],
    states: Dict[int, PromptWMState],
    shutdown: asyncio.Event,
    wm_log_queue: bool,
) -> None:
    """
    Single consumer for wm_out_q while multiple AsyncLLM generation tasks are in flight.
    """
    while not shutdown.is_set():
        try:
            payload = await asyncio.to_thread(wm_out_q.get, True, 0.25)
        except queue_module.Empty:
            await asyncio.sleep(0)
            continue
        items = payload if isinstance(payload, list) else [payload]
        if items:
            _dispatch_wm_payload_to_states(items, states)
            _log_wm_queue_depths(
                wm_log_queue,
                "after_wm→main_dequeue",
                wm_in_q,
                wm_out_q,
                extra=f"last_batch_sents={len(items)}",
            )
    # Final non-blocking drain
    for _ in range(4096):
        try:
            payload = wm_out_q.get(block=False)
        except queue_module.Empty:
            break
        items = payload if isinstance(payload, list) else [payload]
        if items:
            _dispatch_wm_payload_to_states(items, states)


async def _async_stream_generate_then_watermark_sentences(
    gen_llm: Any,
    SamplingParams: Any,
    RequestOutputKind: Any,
    prompt: str,
    gen_params: Any,
    Top_K: int,
    secret_key: str,
    m: int,
    c: int,
    h: int,
    alpha: float,
    output_name: str,
    wm_sentence_batch_size: int,
    tokenizer: AutoTokenizer,
    sync_llm: Optional[Any],
    wm_in_q: Optional["mp.Queue"],
    wm_out_q: Optional["mp.Queue"],
    prompt_id: int,
    wm_state: Optional[PromptWMState],
    wm_log_queue: bool = False,
    wm_temperature: float = 0.0,
    wm_top_p: float = 1.0,
    wm_detect_max_tokens: Optional[int] = None,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Producer (async stream on GPU 0) + sentence splitting + optional watermark process (GPU 1).

    Generation runs as one AsyncLLM request with DELTA outputs. Multiple such requests may run
    concurrently (see --gen_max_inflight); vLLM then continuous/dynamically batches prefill/decode
    across sequences. Watermark jobs use ids (prompt_id, sent_idx) so a shared worker can batch
    many sentences and still return results to the correct stream.
    """
    line_buffer = ""
    sent_idx = 0
    pending_wm_items: List[Tuple[WMQueueKey, str]] = []
    generated = ""

    use_wm_worker = wm_in_q is not None and wm_out_q is not None
    if use_wm_worker and wm_state is None:
        raise ValueError("wm_state is required when using wm_in_q / wm_out_q")

    wm_by_idx: Dict[WMQueueKey, str] = {}
    next_emit = 0
    watermarked_parts: List[str] = []
    sampling_flat: List[Dict[str, Any]] = []

    def _apply_inline_results(idxs: List[WMQueueKey], wm_results: List[Tuple[str, List[Dict[str, Any]]]]) -> None:
        nonlocal next_emit
        for key, (wm_sent, samp) in zip(idxs, wm_results):
            wm_by_idx[key] = wm_sent
            if samp:
                sampling_flat.extend(samp)
        while (prompt_id, next_emit) in wm_by_idx:
            watermarked_parts.append(wm_by_idx.pop((prompt_id, next_emit)))
            next_emit += 1

    async def _flush_pending_wm_items() -> None:
        if not pending_wm_items:
            return
        if wm_in_q is None:
            assert sync_llm is not None
            idxs = [idx for idx, _ in pending_wm_items]
            texts = [txt for _, txt in pending_wm_items]
            wm_results = _watermark_batch_sentences(
                texts,
                sync_llm,
                tokenizer,
                Top_K,
                secret_key,
                m,
                c,
                h,
                alpha,
                output_name,
                wm_sentence_batch_size,
                wm_temperature=wm_temperature,
                wm_top_p=wm_top_p,
                wm_detect_max_tokens=wm_detect_max_tokens,
            )
            _apply_inline_results(idxs, wm_results)
        else:
            n_batch = len(pending_wm_items)
            await asyncio.to_thread(wm_in_q.put, list(pending_wm_items))
            _log_wm_queue_depths(
                wm_log_queue,
                f"after_gen→wm_enqueue prompt_id={prompt_id}",
                wm_in_q,
                wm_out_q,
                extra=f"batch_sents={n_batch}",
            )
        pending_wm_items.clear()

    async def generation_loop() -> None:
        nonlocal generated, sent_idx, line_buffer
        request_id = f"e2e-p{prompt_id}-{uuid.uuid4()}"
        sampling_params = SamplingParams(
            max_tokens=gen_params.max_tokens,
            temperature=gen_params.temperature,
            top_p=gen_params.top_p,
            output_kind=RequestOutputKind.DELTA,
        )

        async for output in gen_llm.generate(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
        ):
            delta_text = ""
            for completion in output.outputs:
                if completion.text:
                    delta_text += completion.text
            if delta_text:
                generated += delta_text
                line_buffer, ready_sentences = _split_streaming_sentences(line_buffer, delta_text, is_final=False)
                for sent in ready_sentences:
                    pending_wm_items.append(((prompt_id, sent_idx), sent))
                    sent_idx += 1
                    if len(pending_wm_items) >= wm_sentence_batch_size:
                        await _flush_pending_wm_items()

            if output.finished:
                line_buffer, flush_sentences = _split_streaming_sentences(line_buffer, "", is_final=True)
                for sent in flush_sentences:
                    pending_wm_items.append(((prompt_id, sent_idx), sent))
                    sent_idx += 1
                    if len(pending_wm_items) >= wm_sentence_batch_size:
                        await _flush_pending_wm_items()
                await _flush_pending_wm_items()
                break

    await generation_loop()

    if use_wm_worker:
        assert wm_state is not None
        wm_state.mark_producer_finished(sent_idx)
        await wm_state.done_event.wait()
        watermarked = "".join(wm_state.watermarked_parts)
        return generated, watermarked, wm_state.sampling_flat

    watermarked = "".join(watermarked_parts)
    return generated, watermarked, sampling_flat


def _sync_fallback_single_shot_generate(
    prompt: str,
    llm: Any,
    tokenizer: AutoTokenizer,
    gen_params: Any,
    Top_K: int,
    secret_key: str,
    m: int,
    c: int,
    h: int,
    alpha: float,
    output_name: str,
    wm_sentence_batch_size: int,
    wm_in_q: Optional["mp.Queue"],
    wm_out_q: Optional["mp.Queue"],
    prompt_id: int = 0,
    wm_log_queue: bool = False,
    wm_temperature: float = 0.0,
    wm_top_p: float = 1.0,
    wm_detect_max_tokens: Optional[int] = None,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    If AsyncLLM is unavailable: one sync generate() for the full budget (no chunk staircase).
    Sentence splitting and watermark queue behavior match the async path (queue ids are
    (prompt_id, sent_idx); sync main uses prompt_id=0 per item).
    """
    line_buffer = ""
    sent_idx = 0
    pending_wm_items: List[Tuple[WMQueueKey, str]] = []
    sampling_flat: List[Dict[str, Any]] = []
    wm_by_idx: Dict[WMQueueKey, str] = {}
    next_emit = 0
    watermarked_parts: List[str] = []

    def _drain_worker(block: bool = False):
        nonlocal next_emit
        if wm_out_q is None:
            return
        timeout = 0.15 if block else 0.0
        next_emit, _ = _apply_drain_batch(
            wm_out_q,
            wm_by_idx,
            watermarked_parts,
            sampling_flat,
            next_emit,
            prompt_id,
            block,
            timeout,
        )

    def _flush_inline_batch(idxs: List[WMQueueKey], texts: List[str]) -> None:
        nonlocal next_emit
        wm_results = _watermark_batch_sentences(
            texts,
            llm,
            tokenizer,
            Top_K,
            secret_key,
            m,
            c,
            h,
            alpha,
            output_name,
            wm_sentence_batch_size,
            wm_temperature=wm_temperature,
            wm_top_p=wm_top_p,
            wm_detect_max_tokens=wm_detect_max_tokens,
        )
        for key, (wm_sent, samp) in zip(idxs, wm_results):
            wm_by_idx[key] = wm_sent
            if samp:
                sampling_flat.extend(samp)
        while (prompt_id, next_emit) in wm_by_idx:
            watermarked_parts.append(wm_by_idx.pop((prompt_id, next_emit)))
            next_emit += 1

    out = llm.generate([prompt], sampling_params=gen_params)
    delta = out[0].outputs[0].text if out and out[0].outputs else ""
    generated = delta

    line_buffer, ready_sentences = _split_streaming_sentences(line_buffer, delta, is_final=False)
    for sent in ready_sentences:
        pending_wm_items.append(((prompt_id, sent_idx), sent))
        sent_idx += 1
        if len(pending_wm_items) >= wm_sentence_batch_size:
            if wm_in_q is None:
                idxs = [idx for idx, _ in pending_wm_items]
                texts = [txt for _, txt in pending_wm_items]
                _flush_inline_batch(idxs, texts)
            else:
                n_batch = len(pending_wm_items)
                wm_in_q.put(list(pending_wm_items))
                _log_wm_queue_depths(
                    wm_log_queue,
                    f"after_gen→wm_enqueue prompt_id={prompt_id}",
                    wm_in_q,
                    wm_out_q,
                    extra=f"batch_sents={n_batch}",
                )
                _drain_worker(block=False)
            pending_wm_items.clear()

    line_buffer, flush_sentences = _split_streaming_sentences(line_buffer, "", is_final=True)
    for sent in flush_sentences:
        pending_wm_items.append(((prompt_id, sent_idx), sent))
        sent_idx += 1
        if len(pending_wm_items) >= wm_sentence_batch_size:
            if wm_in_q is None:
                idxs = [idx for idx, _ in pending_wm_items]
                texts = [txt for _, txt in pending_wm_items]
                _flush_inline_batch(idxs, texts)
            else:
                n_batch = len(pending_wm_items)
                wm_in_q.put(list(pending_wm_items))
                _log_wm_queue_depths(
                    wm_log_queue,
                    f"after_gen→wm_enqueue prompt_id={prompt_id}",
                    wm_in_q,
                    wm_out_q,
                    extra=f"batch_sents={n_batch}",
                )
                _drain_worker(block=False)
            pending_wm_items.clear()

    if pending_wm_items:
        if wm_in_q is None:
            idxs = [idx for idx, _ in pending_wm_items]
            texts = [txt for _, txt in pending_wm_items]
            _flush_inline_batch(idxs, texts)
        else:
            n_batch = len(pending_wm_items)
            wm_in_q.put(list(pending_wm_items))
            _log_wm_queue_depths(
                wm_log_queue,
                f"after_gen→wm_enqueue prompt_id={prompt_id}",
                wm_in_q,
                wm_out_q,
                extra=f"batch_sents={n_batch}",
            )
            _drain_worker(block=False)
        pending_wm_items.clear()

    if wm_in_q is not None and wm_out_q is not None:
        for _ in range(300):
            prev = next_emit
            _drain_worker(block=True)
            if next_emit == sent_idx and prev == next_emit:
                _drain_worker(block=False)
                if next_emit == sent_idx:
                    break

    watermarked = "".join(watermarked_parts)
    return generated, watermarked, sampling_flat


async def async_stream_generate_then_watermark_sentences(
    prompt: str,
    tokenizer: AutoTokenizer,
    gen_params: Any,
    Top_K: int,
    secret_key: str,
    m: int,
    c: int,
    h: int,
    alpha: float,
    output_name: str,
    wm_sentence_batch_size: int,
    wm_in_q: Optional["mp.Queue"],
    wm_out_q: Optional["mp.Queue"],
    async_gen_bundle: Tuple[Any, Any, Any],
    sync_gen_llm: Optional[Any] = None,
    prompt_id: int = 0,
    wm_state: Optional[PromptWMState] = None,
    wm_log_queue: bool = False,
    wm_temperature: float = 0.0,
    wm_top_p: float = 1.0,
    wm_detect_max_tokens: Optional[int] = None,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Async entry point: one streaming generation request (DELTA) + watermark queue.
    Reuse the same async_gen_bundle across many prompts under one event loop.
    When wm_in_q/wm_out_q are set, pass a per-prompt PromptWMState (see _run_all_prompts_async).
    """
    gen_llm, SamplingParams, RequestOutputKind = async_gen_bundle
    return await _async_stream_generate_then_watermark_sentences(
        gen_llm,
        SamplingParams,
        RequestOutputKind,
        prompt,
        gen_params,
        Top_K,
        secret_key,
        m,
        c,
        h,
        alpha,
        output_name,
        wm_sentence_batch_size,
        tokenizer,
        sync_gen_llm,
        wm_in_q,
        wm_out_q,
        prompt_id,
        wm_state,
        wm_log_queue,
        wm_temperature,
        wm_top_p,
        wm_detect_max_tokens,
    )


def sync_stream_generate_then_watermark_sentences(
    prompt: str,
    tokenizer: AutoTokenizer,
    gen_params: Any,
    Top_K: int,
    secret_key: str,
    m: int,
    c: int,
    h: int,
    alpha: float,
    output_name: str,
    wm_sentence_batch_size: int,
    wm_in_q: Optional["mp.Queue"],
    wm_out_q: Optional["mp.Queue"],
    sync_gen_llm: Any,
    prompt_id: int = 0,
    wm_log_queue: bool = False,
    wm_temperature: float = 0.0,
    wm_top_p: float = 1.0,
    wm_detect_max_tokens: Optional[int] = None,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Sync fallback when v1 AsyncLLM is unavailable (single-shot generate, no chunk staircase)."""
    return _sync_fallback_single_shot_generate(
        prompt,
        sync_gen_llm,
        tokenizer,
        gen_params,
        Top_K,
        secret_key,
        m,
        c,
        h,
        alpha,
        output_name,
        wm_sentence_batch_size,
        wm_in_q,
        wm_out_q,
        prompt_id=prompt_id,
        wm_log_queue=wm_log_queue,
        wm_temperature=wm_temperature,
        wm_top_p=wm_top_p,
        wm_detect_max_tokens=wm_detect_max_tokens,
    )


async def _shutdown_async_gen_llm(gen_llm: Any) -> None:
    shutdown = getattr(gen_llm, "shutdown", None)
    if not callable(shutdown):
        return
    maybe = shutdown()
    if asyncio.iscoroutine(maybe):
        await maybe

def _resolve_gen_max_num_seqs(args: argparse.Namespace) -> int:
    """vLLM scheduler cap for generation: align with concurrent AsyncLLM streams unless overridden."""
    if getattr(args, "max_num_seqs", None) is not None:
        return max(1, int(args.max_num_seqs))
    return max(1, int(args.gen_max_inflight))


def _resolve_wm_max_num_seqs(args: argparse.Namespace) -> int:
    """vLLM scheduler cap for watermark worker: align with max sentences batched unless overridden."""
    if getattr(args, "max_num_seqs", None) is not None:
        return max(1, int(args.max_num_seqs))
    return max(1, int(args.wm_sentence_batch_size))


def main(args):
    start_time_all = time.time()
    torch.cuda.empty_cache()

    # Validate inputs BEFORE initializing vLLM to avoid long warmup then crash.
    if not args.data and not args.prompt_pt:
        raise ValueError(
            "Missing input source. Provide either:\n"
            "- --data <path/to/prompts.json> (expects a list of objects with 'input' or 'Input') OR\n"
            "- --prompt_pt <path/to/prompts.pt> (expects the same shape used by c4_prompt_test.pt)\n"
        )
    if args.wm_sentence_batch_size < 1:
        raise ValueError("--wm_sentence_batch_size must be >= 1")
    if args.gen_max_inflight < 1:
        raise ValueError("--gen_max_inflight must be >= 1")
    
    # Pin this (generation) process to one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gen_gpu)
    _setup_conda_cuda_for_flashinfer_link()

    model_name = args.model if hasattr(args, "model") and args.model else DEFAULT_MODEL
    print(f"Initializing generation with model: {model_name}")
    engine_cfg = _vllm_engine_config_dict(args)
    print(
        "vLLM engine: "
        f"dtype={engine_cfg['dtype']} tp={engine_cfg['tensor_parallel_size']} "
        f"weight_quant={engine_cfg.get('vllm_model_quantization')!r} "
        f"kv_quantization={engine_cfg['quantization']} "
        f"kv_cache_dtype={engine_cfg['kv_cache_dtype']!r} "
        f"calculate_kv_scales={engine_cfg['calculate_kv_scales']} "
        f"prefix_cache={engine_cfg['enable_prefix_caching']} "
        f"scheduling_policy={engine_cfg['scheduling_policy']!r} "
        f"chunked_prefill={engine_cfg['enable_chunked_prefill']} "
        f"max_num_batched_tokens={engine_cfg.get('max_num_batched_tokens')!r}"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    gen_cap = _resolve_gen_max_num_seqs(args)
    wm_seq_cap = _resolve_wm_max_num_seqs(args)
    async_bundle = _build_async_gen_llm(
        model_name, cache_dir, engine_cfg=engine_cfg, max_num_seqs=gen_cap
    )
    sync_gen_llm = None
    if async_bundle is None:
        print(
            "vLLM v1 AsyncLLM not found; using sync LLM with one generate() per prompt "
            "(no chunked staircase; not token-streamed)."
        )
        sync_gen_llm, tokenizer, SamplingParams = _build_llm_and_tokenizer(
            model_name, cache_dir, engine_cfg=engine_cfg, max_num_seqs=gen_cap
        )
    else:
        from vllm import SamplingParams  # type: ignore

        print(
            f"AsyncLLM (streaming DELTA) loaded; vLLM max_num_seqs={gen_cap} "
            f"(app concurrency cap: --gen_max_inflight={args.gen_max_inflight}); "
            f"watermark engine will use max_num_seqs={wm_seq_cap}."
        )
    print("Similarity scoring disabled: all LLM candidates share equal weight in sampling.")

    n_dev_main = torch.cuda.device_count()
    print(
        f"[GPU layout] Generation process: physical GPU id={args.gen_gpu}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, "
        f"torch.cuda.device_count()={n_dev_main}"
    )
    if n_dev_main > 0:
        print(f"[GPU layout] Generation device name: {torch.cuda.get_device_name(0)}")

    # Bounded queues between streaming generator (gen GPU) and watermark worker (wm GPU)
    qmax = args.wm_queue_maxsize
    wm_in_q: "mp.Queue" = mp.Queue(maxsize=qmax)
    wm_out_q: "mp.Queue" = mp.Queue(maxsize=qmax)

    wm_proc = mp.Process(
        target=_watermark_worker_main,
        args=(
            args.wm_gpu,
            model_name,
            cache_dir,
            engine_cfg,
            args.wm_top_k,
            args.secret_key,
            args.wm_m,
            args.wm_c,
            args.wm_h,
            args.wm_alpha,
            f"{args.split}_{args.data_model}_WM",
            args.wm_sentence_batch_size,
            float(args.wm_temperature),
            float(args.wm_top_p),
            int(args.wm_detect_max_tokens),
            wm_seq_cap,
            wm_in_q,
            wm_out_q,
        ),
        # vLLM spawns worker subprocesses; daemon processes cannot spawn children
        daemon=False,
    )
    wm_proc.start()
    if args.gen_gpu != args.wm_gpu:
        print(
            f"[GPU layout] Two-GPU pipeline: generation on physical GPU {args.gen_gpu}, "
            f"watermark on physical GPU {args.wm_gpu} (two processes; check worker log above)."
        )
    else:
        print(
            f"[GPU layout] Warning: --gen_gpu and --wm_gpu are both {args.gen_gpu}; "
            "both engines use the same GPU (no true two-GPU parallelism, often slower than one engine)."
        )

    generated_data: List[Any] = []
    Top_K = args.wm_top_k
    secret_key = args.secret_key
    m = args.wm_m
    c = args.wm_c
    h = args.wm_h
    alpha = args.wm_alpha
    wm_temperature = float(args.wm_temperature)
    wm_top_p = float(args.wm_top_p)
    wm_detect_max_tokens = int(args.wm_detect_max_tokens)
    data_model = args.data_model

    # Load prompts: --data JSON list, or --prompt_pt torch dataset (c4_prompt_test.pt shape)
    prompts: List[str] = []
    if args.data:
        N_start = 0
        N_end = args.n_inputs if args.n_inputs and args.n_inputs > 0 else None
        with open(args.data, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("--data must be a JSON list of prompt items.")
        items = raw[N_start:N_end] if N_end is not None else raw[N_start:]
        for item in items:
            if isinstance(item, dict):
                p = item.get("input", None) or item.get("Input", None)
                if isinstance(p, str) and p.strip():
                    prompts.append(p)
            elif isinstance(item, str) and item.strip():
                prompts.append(item)
        if not prompts:
            raise ValueError("No usable prompts found in --data. Expected 'input'/'Input' fields or string items.")
    else:
        pt = torch.load(args.prompt_pt)
        if not pt or not isinstance(pt, (list, tuple)) or not pt[0]:
            raise ValueError("--prompt_pt does not look like the expected prompt dataset.")
        prompts = [str(x) for x in pt[0][: args.n_inputs]]

    N_start = 0
    N_end = len(prompts)
    wq = engine_cfg.get("vllm_model_quantization") or "none"
    model_slug = _safe_output_slug(model_name)
    output_name = (
        f"{model_slug}_Batching_wq{wq}_kv{engine_cfg['quantization']}_{engine_cfg['kv_cache_dtype']}_{engine_cfg['calculate_kv_scales']}_{args.split}_{data_model}_E2E_KEY_{secret_key}_m{m}_c{c}_h{h}_alpha{alpha}"
        f"_n{N_end}_in{args.max_inp_tokens}_new{args.max_new_tokens}"
        f"_gen_inflight_{args.gen_max_inflight}_sentence_batch_{args.wm_sentence_batch_size}"
    )

    print(f"Starting end-to-end generation + watermarking with {len(prompts)} prompts")
    print(
        "Pipeline: AsyncLLM runs up to --gen_max_inflight streaming requests at once so vLLM can "
        "continuously/dynamically batch decodes; completed sentences are queued to the watermark "
        "worker (shared instruct prefix, batched detect/generate)."
    )
    print(f"Generation max in-flight requests (asyncio): {args.gen_max_inflight}")
    print(f"vLLM max_num_seqs (gen / wm): {gen_cap} / {wm_seq_cap}")
    print(f"Watermark sentence batch size: {args.wm_sentence_batch_size}")
    print(f"Watermark LLM decode: temperature={wm_temperature} top_p={wm_top_p}")

    if getattr(args, "wm_log_queue", False):
        print(
            "[wm-queue] Logging enabled: gen→wm counts batched jobs (each job has up to "
            f"{args.wm_sentence_batch_size} sentences); approximate depths via Queue.qsize()."
        )

    def _process_one_item(query: str, i: int) -> None:
        print(f"Processing item {i+1} / {N_end - N_start}")

        if args.max_inp_tokens and args.max_inp_tokens > 0:
            enc = tokenizer(
                query,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_inp_tokens,
            )
            query = tokenizer.batch_decode(enc["input_ids"], skip_special_tokens=True)[0]

        gen_params = _make_gen_sampling_params(args)
        generated_output_text, watermarked_output_text, sampling_flat = sync_stream_generate_then_watermark_sentences(
            prompt=query,
            tokenizer=tokenizer,
            gen_params=gen_params,
            Top_K=Top_K,
            secret_key=secret_key,
            m=m,
            c=c,
            h=h,
            alpha=alpha,
            output_name=output_name,
            wm_sentence_batch_size=args.wm_sentence_batch_size,
            wm_in_q=wm_in_q,
            wm_out_q=wm_out_q,
            sync_gen_llm=sync_gen_llm,
            prompt_id=i,
            wm_log_queue=getattr(args, "wm_log_queue", False),
            wm_temperature=wm_temperature,
            wm_top_p=wm_top_p,
            wm_detect_max_tokens=wm_detect_max_tokens,
        )
        final_text = watermarked_output_text
        text = generated_output_text

        data_dict = {
            "input": query,
            "Original_output": text,
            "Watermarked_output": final_text,
            "time": 0.0,
        }
        generated_data.append(data_dict)

    # Wall clock: t0 = start of timed batch (after optional warmup), t1 = all prompts done
    pipeline_timing: Dict[str, Optional[float]] = {"t0": None, "t1": None}

    async def _run_all_prompts_async() -> None:
        assert async_bundle is not None
        n_prompts = len(prompts)
        wm_states: Dict[int, PromptWMState] = {i: PromptWMState() for i in range(n_prompts)}
        shutdown_drain = asyncio.Event()
        drain_task = asyncio.create_task(
            _global_watermark_drain_loop(
                wm_out_q,
                wm_in_q,
                wm_states,
                shutdown_drain,
                getattr(args, "wm_log_queue", False),
            )
        )
        sem = asyncio.Semaphore(max(1, args.gen_max_inflight))

        async def _pipeline_warmup_async() -> None:
            if not getattr(args, "warmup", True):
                return
            wm_states[WARMUP_PROMPT_ID] = PromptWMState()
            try:
                gp = _make_gen_sampling_params(args, max_tokens=max(1, int(args.warmup_max_tokens)))
                await async_stream_generate_then_watermark_sentences(
                    prompt=str(args.warmup_prompt),
                    tokenizer=tokenizer,
                    gen_params=gp,
                    Top_K=Top_K,
                    secret_key=secret_key,
                    m=m,
                    c=c,
                    h=h,
                    alpha=alpha,
                    output_name=output_name,
                    wm_sentence_batch_size=args.wm_sentence_batch_size,
                    wm_in_q=wm_in_q,
                    wm_out_q=wm_out_q,
                    async_gen_bundle=async_bundle,
                    sync_gen_llm=None,
                    prompt_id=WARMUP_PROMPT_ID,
                    wm_state=wm_states[WARMUP_PROMPT_ID],
                    wm_log_queue=getattr(args, "wm_log_queue", False),
                    wm_temperature=wm_temperature,
                    wm_top_p=wm_top_p,
                    wm_detect_max_tokens=wm_detect_max_tokens,
                )
                print("[warmup] Completed one short gen+watermark run (excluded from pipeline wall time).")
            finally:
                wm_states.pop(WARMUP_PROMPT_ID, None)

        async def _one_async(i: int, query: str) -> None:
            print(f"Processing item {i+1} / {n_prompts} (slot acquired)")

            q = query
            if args.max_inp_tokens and args.max_inp_tokens > 0:
                enc = tokenizer(
                    q,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=args.max_inp_tokens,
                )
                q = tokenizer.batch_decode(enc["input_ids"], skip_special_tokens=True)[0]

            gen_params = _make_gen_sampling_params(args)
            generated_output_text, watermarked_output_text, sampling_flat = (
                await async_stream_generate_then_watermark_sentences(
                    prompt=q,
                    tokenizer=tokenizer,
                    gen_params=gen_params,
                    Top_K=Top_K,
                    secret_key=secret_key,
                    m=m,
                    c=c,
                    h=h,
                    alpha=alpha,
                    output_name=output_name,
                    wm_sentence_batch_size=args.wm_sentence_batch_size,
                    wm_in_q=wm_in_q,
                    wm_out_q=wm_out_q,
                    async_gen_bundle=async_bundle,
                    sync_gen_llm=None,
                    prompt_id=i,
                    wm_state=wm_states[i],
                    wm_log_queue=getattr(args, "wm_log_queue", False),
                    wm_temperature=wm_temperature,
                    wm_top_p=wm_top_p,
                    wm_detect_max_tokens=wm_detect_max_tokens,
                )
            )
            generated_data[i] = {
                "input": q,
                "Original_output": generated_output_text,
                "Watermarked_output": watermarked_output_text,
                "time": 0.0,
            }

        try:
            async def _bounded(i: int, query: str) -> None:
                async with sem:
                    await _one_async(i, query)

            await _pipeline_warmup_async()
            generated_data.clear()
            generated_data.extend([None] * n_prompts)
            pipeline_timing["t0"] = time.perf_counter()
            await asyncio.gather(*(_bounded(i, q) for i, q in enumerate(prompts)))
            compact = [
                generated_data[i] for i in range(n_prompts) if generated_data[i] is not None
            ]
            generated_data.clear()
            generated_data.extend(compact)
            pipeline_timing["t1"] = time.perf_counter()
        finally:
            shutdown_drain.set()
            await drain_task
            await _shutdown_async_gen_llm(async_bundle[0])

    if async_bundle is not None:
        asyncio.run(_run_all_prompts_async())
    else:
        if getattr(args, "warmup", True) and sync_gen_llm is not None:
            gp = _make_gen_sampling_params(args, max_tokens=max(1, int(args.warmup_max_tokens)))
            sync_stream_generate_then_watermark_sentences(
                prompt=str(args.warmup_prompt),
                tokenizer=tokenizer,
                gen_params=gp,
                Top_K=Top_K,
                secret_key=secret_key,
                m=m,
                c=c,
                h=h,
                alpha=alpha,
                output_name=output_name,
                wm_sentence_batch_size=args.wm_sentence_batch_size,
                wm_in_q=wm_in_q,
                wm_out_q=wm_out_q,
                sync_gen_llm=sync_gen_llm,
                prompt_id=WARMUP_PROMPT_ID,
                wm_log_queue=getattr(args, "wm_log_queue", False),
                wm_temperature=wm_temperature,
                wm_top_p=wm_top_p,
                wm_detect_max_tokens=wm_detect_max_tokens,
            )
            print("[warmup] Completed one short gen+watermark run (excluded from pipeline wall time).")
        pipeline_timing["t0"] = time.perf_counter()
        for i, query in enumerate(prompts):
            _process_one_item(query, i)
        pipeline_timing["t1"] = time.perf_counter()

    pipeline_wall_s = 0.0
    if pipeline_timing["t0"] is not None and pipeline_timing["t1"] is not None:
        pipeline_wall_s = pipeline_timing["t1"] - pipeline_timing["t0"]

    end_time_all = time.time()
    time_elapsed_all = end_time_all - start_time_all

    if generated_data:
        n_out = len(generated_data)
        avg_pipeline = (pipeline_wall_s / n_out) if n_out else 0.0
        for row in generated_data:
            row["time"] = avg_pipeline
        out_path = output_name + ".json"
        with open(out_path, "w") as json_file:
            json.dump(generated_data, json_file, indent=4)
        print(
            f"Pipeline wall time (after warmup → last prompt done): {pipeline_wall_s:.4f} s "
            f"({n_out} prompts)"
        )
        print(f"Avg seconds per prompt (pipeline_wall / N): {avg_pipeline:.4f} s")
        print(f"Results written to {out_path}")
    print(f"Total wall time (incl. model load & init): {time_elapsed_all:.4f} s")
    if generated_data and pipeline_wall_s > 0.0:
        print(
            f"Overhead outside measured pipeline: {max(0.0, time_elapsed_all - pipeline_wall_s):.4f} s"
        )

    # Shutdown watermark worker cleanly
    try:
        wm_in_q.put(None)
    except Exception:
        pass
    try:
        wm_proc.join(timeout=30)
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Whole Context Watermark Generation')
    # --- Data input ---
    parser.add_argument('--data', default=None, type=str, help="Path to JSON list of prompts (items may be strings or objects with 'input'/'Input'). If omitted, uses --prompt_pt.")
    parser.add_argument('--data_model', default='Llama3', type=str, choices=['Llama3', 'Misrtal', 'DeepSeek', 'Qwen', 'Gemma'], help='Dataset model')
    parser.add_argument('--split', default='Test', type=str, choices=['Test', 'Train'], help='Dataset split (Test or Train)')
    parser.add_argument('--prompt_pt', default='/network/rit/lab/Lai_ReSecureAI/kiel/wmm/c4_prompt_test.pt', type=str, help='Path to .pt prompt dataset (used when --data is not provided)')
    parser.add_argument('--n_inputs', default=20, type=int, help='Number of prompts to take from --prompt_pt (used when --data is not provided)')
    parser.add_argument('--max_inp_tokens', default=50, type=int, help='Max input tokens kept from each prompt before generation')
    parser.add_argument('--max_new_tokens', default=200, type=int, help='Max new tokens to generate per prompt')
    
    # --- Text generation parameters ---
    gen_sample = parser.add_argument_group('Text generation parameters')
    gen_sample.add_argument('--gen-temperature', default=0.0, type=float, help='Temperature for generation (0 = greedy). Passing only max_tokens used vLLM default 1.0 (sampling), which tends to longer / more variable outputs.')
    gen_sample.add_argument('--gen-top-p', dest='gen_top_p', default=1.0, type=float, help='top_p for generation.')
    
    # --- Watermarking parameters ---
    watermark = parser.add_argument_group('Watermarking parameters')
    watermark.add_argument('--wm-temperature', dest='wm_temperature', default=0.0, type=float, help='Temperature for watermark LLM detect/generate (0 = greedy).')
    watermark.add_argument('--wm-top-p', dest='wm_top_p', default=1.0, type=float, help='top_p for watermark LLM detect/generate.')
    watermark.add_argument('--wm-top-k', dest='wm_top_k', default=15, type=int, help='Top K synonym alternatives per target word.')
    watermark.add_argument('--wm-m', dest='wm_m', default=6, type=int, help='Number of tournament rounds.')
    watermark.add_argument('--wm-c', dest='wm_c', default=2, type=int, help='Number of competitors per tournament match.')
    watermark.add_argument('--wm-h', dest='wm_h', default=4, type=int, help='Left context tokens for tournament hashing.')
    watermark.add_argument('--wm-alpha', dest='wm_alpha', default=1.0, type=float, help='Softmax temperature for tournament draws (similarity is uniform).')
    watermark.add_argument('--secret_key', default='Adaptive_key_v1', type=str, help='Secret key for tournament randomization.')
    watermark.add_argument('--wm-detect-max-tokens', dest='wm_detect_max_tokens', default=512, type=int, help='Max tokens for watermark detect/generate JSON per sentence (raise if you see truncated JSON).')

   
    # --- Warmup parameters (No need to change) ---
    timing = parser.add_argument_group('Pipeline timing: wall clock starts immediately before the timed prompt batch (after optional warmup).')
    timing.add_argument('--warmup', action=argparse.BooleanOptionalAction, default=True, help='Run one short gen+watermark request before the timed region (JIT/CUDA graphs). Use --no-warmup to skip.')
    timing.add_argument('--warmup-prompt', default='Hello.', type=str, help='Input text for the warmup request.')
    timing.add_argument('--warmup-max-tokens', default=8, type=int, help='max_tokens for the warmup generation (keep small).')

    # --- vLLM engine optimization parameters ---
    vllm_engine = parser.add_argument_group('vLLM engine parameters')
    vllm_engine.add_argument('--model', default=DEFAULT_MODEL, type=str, help='HF hub id or local path for model.')
    vllm_engine.add_argument('--tensor_parallel_size', default=1, type=int, help='Tensor parallel size for vLLM')
    vllm_engine.add_argument('--dtype', default='bfloat16', type=str, help='vLLM ModelConfig dtype (auto, bfloat16, float16, ...). auto works for most checkpoints.')
    vllm_engine.add_argument('--vllm-model-quantization', dest='vllm_model_quantization', default=None, type=str, help='vLLM weight quantization (LLM quantization=...), e.g. compressed-tensors, gptq. Omit for auto-detect.')
    vllm_engine.add_argument('--trust-remote-code', action=argparse.BooleanOptionalAction, default=True, help='HF trust_remote_code when loading the model')
    vllm_engine.add_argument('--enable-prefix-caching', action=argparse.BooleanOptionalAction, default=True, help='vLLM prefix caching (KV block reuse for shared prefixes)')
    vllm_engine.add_argument('--scheduling-policy', dest='scheduling_policy', default='priority', type=str, choices=['fcfs', 'priority'], help='vLLM scheduler policy')
    vllm_engine.add_argument('--enable-chunked-prefill', action=argparse.BooleanOptionalAction, default=True, help='Chunked prefill in the scheduler')
    vllm_engine.add_argument('--quantization', action=argparse.BooleanOptionalAction, default=False, help='If true, pass KV cache dtype / calculate_kv_scales (not weight INT4; see --vllm-model-quantization).')
    vllm_engine.add_argument('--kv-cache-dtype', dest='kv_cache_dtype', default='auto', type=str, help='KV cache storage when --quantization (auto, bfloat16, fp8, ...). Prefer auto on Ampere.')
    vllm_engine.add_argument('--calculate-kv-scales', action=argparse.BooleanOptionalAction, default=False, help='Passed to vLLM when --quantization is true.')

    # --- Resource allocation parameters ---
    parser.add_argument('--gen_gpu', default=0, type=int, help='GPU id for generation engine (one GPU per engine)')
    parser.add_argument('--wm_gpu', default=1, type=int, help='GPU id for watermark engine (one GPU per engine)')
    

    # --- Continuous batching / queue tuning ---
    batching = parser.add_argument_group('Continuous batching / queue tuning')
    batching.add_argument('--wm_queue_maxsize', default=256, type=int, help='Max queue size for watermark engine to wait for watermarked sentences.')
    batching.add_argument('--wm_sentence_batch_size', default=32, type=int, help='Watermark sentence batch size.')
    batching.add_argument('--gen_max_inflight', default=32, type=int, help='Text generation batch size.')
    batching.add_argument('--wm-log-queue', dest='wm_log_queue', action='store_true', help='Log watermark queue depth.')

    main(parser.parse_args())
