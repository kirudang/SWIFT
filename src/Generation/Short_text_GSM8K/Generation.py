import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
import resource

# Slurm/login shells sometimes set a tiny virtual-memory soft limit (~500MB).
# libtorch_cuda.so is ~1GB and fails with: failed to map segment from shared object.
def _raise_virtual_memory_limit() -> None:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        target = hard if hard != resource.RLIM_INFINITY else resource.RLIM_INFINITY
        if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
            resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    except (ValueError, OSError):
        pass


_raise_virtual_memory_limit()

# Ensure the HF_HOME environment variable points to your desired cache location
cache_dir = 'Your Cache Directory'
os.environ["HF_HOME"] = cache_dir

# Force vLLM workers to use spawn (not fork)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# vLLM v1 enables FlashInfer sampling by default; JIT needs nvcc + curand.h.
# Greedy decode (temperature=0) does not need it. Override with =1 if toolkit is complete.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import json
import asyncio
import collections
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
    apply_replacements,
    build_watermark_chat_messages,
    finalize_watermarked_sentence,
    build_sampling_params_for_synonyms,
)
from mixed_serving import (
    AdaptiveRatioController,
    GlobalWatermarkQueue,
    WatermarkQueueItem,
    ServingRequestMeta,
    compute_available_gen_slots,
    compute_watermark_admission,
)
from gsm8k_prompts import (
    GSM8K_DATASET_ID,
    extract_output_answer,
    load_gsm8k_examples,
    render_chat_prompt,
)
from typing import Optional, List, Tuple, Dict, Any, Iterable

# Legacy watermark job id: (prompt_index, sentence_index_within_prompt)
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


def _configure_cuda_home_for_nvcc() -> None:
    """
    FlashInfer JIT (vLLM v1 sampler) needs nvcc + CUDA_HOME at runtime.
    DGX nodes often have no /usr/local/cuda; conda envs may ship nvcc under $CONDA_PREFIX/bin.
    Set FLASHINFER_CUDA_HOME to another env (e.g. wmm_env) if the active env lacks nvcc.
    """
    def _nvcc_in(root: str) -> bool:
        return os.path.isfile(os.path.join(root, "bin", "nvcc"))

    roots: List[str] = []
    for key in ("FLASHINFER_CUDA_HOME", "VLLM_CUDA_HOME", "CUDA_HOME"):
        v = os.environ.get(key)
        if v and v not in roots:
            roots.append(v)
    pfx = os.environ.get("CONDA_PREFIX")
    if pfx and pfx not in roots:
        roots.append(pfx)
    for rel in ("/usr/local/cuda", "/usr/local/cuda-12.6", "/usr/local/cuda-12.4"):
        if os.path.isdir(rel) and rel not in roots:
            roots.append(rel)

    for root in roots:
        if not _nvcc_in(root):
            continue
        os.environ["CUDA_HOME"] = root
        bin_dir = os.path.join(root, "bin")
        path = os.environ.get("PATH", "")
        if bin_dir not in path.split(":"):
            os.environ["PATH"] = bin_dir + (":" + path if path else "")
        print(f"[cuda] CUDA_HOME={root} (nvcc available for FlashInfer JIT)")
        return

    print(
        "[cuda] Warning: nvcc not found. FlashInfer JIT will likely fail during engine init.\n"
        "         Fix A: conda install -c nvidia cuda-nvcc   (in the active env)\n"
        "         Fix B: export FLASHINFER_CUDA_HOME=/path/to/env/with/nvcc\n"
        "                e.g. export FLASHINFER_CUDA_HOME=$CONDA_PREFIX/../wmm_env"
    )


def _setup_conda_cuda_for_flashinfer_link() -> None:
    _ensure_conda_lib64_cuda_symlinks()
    _prepend_conda_cuda_lib_path()
    _configure_cuda_home_for_nvcc()


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
DEFAULT_MODEL = "google/gemma-3-27b-it" # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
_SENTENCE_CLOSERS = set("\"'”’)]}")
_SENTENCE_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "e.g.", "i.e.", "etc.", "vs.", "u.s.", "u.k."
}

class MixedServingOrchestrator:
    """
    Single-GPU adaptive mixed continuous batching for generation + watermarking.

    Both request types share one AsyncLLM engine.  A global FIFO watermark queue
    and ratio controller decide how many watermark sequences are admitted per step.
    """

    def __init__(
        self,
        *,
        gen_llm: Any,
        SamplingParams: Any,
        RequestOutputKind: Any,
        tokenizer: AutoTokenizer,
        batch_size: int,
        ratio_ctrl: AdaptiveRatioController,
        top_k: int,
        secret_key: str,
        m: int,
        c: int,
        h: int,
        alpha: float,
        wm_temperature: float,
        wm_top_p: float,
        wm_detect_max_tokens: Optional[int],
    ) -> None:
        self.gen_llm = gen_llm
        self.SamplingParams = SamplingParams
        self.RequestOutputKind = RequestOutputKind
        self.tokenizer = tokenizer
        self.batch_size = max(1, int(batch_size))
        self.ratio_ctrl = ratio_ctrl
        self.top_k = top_k
        self.secret_key = secret_key
        self.m = m
        self.c = c
        self.h = h
        self.alpha = alpha
        self.wm_temperature = wm_temperature
        self.wm_top_p = wm_top_p
        self.wm_detect_max_tokens = wm_detect_max_tokens

        self.wm_queue = GlobalWatermarkQueue()
        self.wm_states: Dict[int, PromptWMState] = {}
        self._pending_gen: collections.deque[Tuple[int, str, Any]] = collections.deque()
        self._active_gen: Set[int] = set()
        self._active_wm: Set[str] = set()
        self._request_meta: Dict[str, ServingRequestMeta] = {}
        self._schedule_lock = asyncio.Lock()
        self._prompt_generated: Dict[int, str] = {}
        self._prompt_sent_count: Dict[int, int] = {}
        self._wm_sampling_params: Optional[Any] = None
        self._wm_structured_used: bool = False

    @staticmethod
    def _gen_request_id(prompt_id: int) -> str:
        return "gen-warmup" if prompt_id == WARMUP_PROMPT_ID else f"gen-p{prompt_id}"

    @staticmethod
    def _wm_request_id(item: WatermarkQueueItem) -> str:
        return f"wm-p{item.prompt_id}-s{item.sent_idx}"

    def _wm_sampling(self) -> Tuple[Any, bool]:
        if self._wm_sampling_params is None:
            self._wm_sampling_params, self._wm_structured_used = build_sampling_params_for_synonyms(
                max_tokens=self.wm_detect_max_tokens,
                temperature=self.wm_temperature,
                top_p=self.wm_top_p,
            )
        return self._wm_sampling_params, self._wm_structured_used

    async def _try_schedule(self) -> None:
        async with self._schedule_lock:
            queue_len = await self.wm_queue.__len__()
            oldest_wait = await self.wm_queue.oldest_wait_s()
            self.ratio_ctrl.maybe_update(queue_len, oldest_wait)
            r_admit = self.ratio_ctrl.admission_ratio(queue_len, oldest_wait)

            new_wm = compute_watermark_admission(
                r_t=r_admit,
                batch_size=self.batch_size,
                queue_len=queue_len,
                active_watermark=len(self._active_wm),
            )
            capacity = compute_available_gen_slots(
                batch_size=self.batch_size,
                active_generation=len(self._active_gen),
                active_watermark=len(self._active_wm),
            )

            for _ in range(new_wm):
                if capacity <= 0:
                    break
                item = await self.wm_queue.dequeue()
                if item is None:
                    break
                req_id = self._wm_request_id(item)
                self._active_wm.add(req_id)
                self._request_meta[req_id] = ServingRequestMeta(
                    request_id=req_id,
                    request_type="watermark",
                    prompt_id=item.prompt_id,
                    sent_idx=item.sent_idx,
                )
                capacity -= 1
                asyncio.create_task(self._run_watermark(item, req_id))

            while capacity > 0 and self._pending_gen:
                prompt_id, prompt, gen_params = self._pending_gen.popleft()
                self._active_gen.add(prompt_id)
                gen_req_id = self._gen_request_id(prompt_id)
                self._request_meta[gen_req_id] = ServingRequestMeta(
                    request_id=gen_req_id,
                    request_type="generation",
                    prompt_id=prompt_id,
                )
                capacity -= 1
                asyncio.create_task(self._run_generation(prompt_id, prompt, gen_params))

    async def _run_watermark(self, item: WatermarkQueueItem, req_id: str) -> None:
        structured_used = False
        decoded = ""
        try:
            messages = build_watermark_chat_messages(item.sentence_text)
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            sampling_params, structured_used = self._wm_sampling()
            async for output in self.gen_llm.generate(
                request_id=req_id,
                prompt=prompt,
                sampling_params=sampling_params,
            ):
                if output.outputs:
                    decoded = output.outputs[0].text or decoded
                if output.finished:
                    break
        except Exception as exc:
            print(
                f"[orchestrator] watermark failed prompt_id={item.prompt_id} "
                f"sent_idx={item.sent_idx}: {exc}",
                flush=True,
            )
            wm_sent, samp = item.sentence_text, []
        else:
            wm_sent, samp = finalize_watermarked_sentence(
                item.sentence_text,
                decoded,
                self.top_k,
                self.secret_key,
                self.m,
                self.c,
                self.h,
                self.alpha,
                structured_used=structured_used,
            )
        try:
            st = self.wm_states.get(item.prompt_id)
            if st is not None:
                st.apply_result(item.sent_idx, wm_sent, samp)
        finally:
            self._active_wm.discard(req_id)
            self._request_meta.pop(req_id, None)
            await self._try_schedule()

    async def _run_generation(self, prompt_id: int, prompt: str, gen_params: Any) -> None:
        line_buffer = ""
        sent_idx = 0
        generated = ""
        req_id = self._gen_request_id(prompt_id)
        try:
            sampling_params = self.SamplingParams(
                max_tokens=gen_params.max_tokens,
                temperature=gen_params.temperature,
                top_p=gen_params.top_p,
                output_kind=self.RequestOutputKind.DELTA,
            )
            async for output in self.gen_llm.generate(
                request_id=req_id,
                prompt=prompt,
                sampling_params=sampling_params,
            ):
                delta_text = ""
                for completion in output.outputs:
                    if completion.text:
                        delta_text += completion.text
                if delta_text:
                    generated += delta_text
                    line_buffer, ready_sentences = _split_streaming_sentences(
                        line_buffer, delta_text, is_final=False
                    )
                    for sent in ready_sentences:
                        await self.wm_queue.enqueue(prompt_id, sent_idx, sent)
                        sent_idx += 1
                        await self._try_schedule()

                if output.finished:
                    line_buffer, flush_sentences = _split_streaming_sentences(
                        line_buffer, "", is_final=True
                    )
                    for sent in flush_sentences:
                        await self.wm_queue.enqueue(prompt_id, sent_idx, sent)
                        sent_idx += 1
                        await self._try_schedule()
                    break
        except Exception as exc:
            print(f"[orchestrator] generation failed prompt_id={prompt_id}: {exc}", flush=True)
        finally:
            self._active_gen.discard(prompt_id)
            self._request_meta.pop(req_id, None)
            self._prompt_generated[prompt_id] = generated
            self._prompt_sent_count[prompt_id] = sent_idx
            st = self.wm_states.get(prompt_id)
            if st is not None:
                st.mark_producer_finished(sent_idx)
            await self._try_schedule()

    async def run_prompt(
        self,
        prompt_id: int,
        prompt: str,
        gen_params: Any,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        self.wm_states[prompt_id] = PromptWMState()
        self._pending_gen.append((prompt_id, prompt, gen_params))
        await self._try_schedule()
        st = self.wm_states[prompt_id]
        await st.done_event.wait()
        generated = self._prompt_generated.get(prompt_id, "")
        watermarked = "".join(st.watermarked_parts)
        return generated, watermarked, st.sampling_flat

    async def run_all_prompts(
        self,
        items: List[Tuple[int, str, Any]],
    ) -> List[Tuple[int, str, str, List[Dict[str, Any]]]]:
        for prompt_id, prompt, gen_params in items:
            self.wm_states[prompt_id] = PromptWMState()
            self._pending_gen.append((prompt_id, prompt, gen_params))
        await self._try_schedule()
        await asyncio.gather(*(self.wm_states[pid].done_event.wait() for pid, _, _ in items))
        out: List[Tuple[int, str, str, List[Dict[str, Any]]]] = []
        for prompt_id, _, _ in items:
            st = self.wm_states[prompt_id]
            out.append((
                prompt_id,
                self._prompt_generated.get(prompt_id, ""),
                "".join(st.watermarked_parts),
                st.sampling_flat,
            ))
        return out

    async def shutdown(self) -> None:
        await _shutdown_async_gen_llm(self.gen_llm)


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


def _sync_single_gpu_prompt(
    prompt: str,
    llm: Any,
    tokenizer: AutoTokenizer,
    gen_params: Any,
    output_name: str,
    wm_sentence_batch_size: int,
    Top_K: int,
    secret_key: str,
    m: int,
    c: int,
    h: int,
    alpha: float,
    *,
    wm_temperature: float = 0.0,
    wm_top_p: float = 1.0,
    wm_detect_max_tokens: Optional[int] = None,
    prompt_id: int = 0,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Sync fallback when AsyncLLM is unavailable: one vLLM engine on one GPU.
    Generates the full response, then watermarks sentence segments on the same engine.
    """
    del prompt_id  # reserved for API compatibility
    out = llm.generate([prompt], sampling_params=gen_params)
    generated = out[0].outputs[0].text if out and out[0].outputs else ""
    line_buffer, sentences = _split_streaming_sentences("", generated, is_final=True)
    if line_buffer:
        sentences.append(line_buffer)

    if not sentences:
        return generated, "", []

    wm_results = _watermark_batch_sentences(
        sentences,
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
    watermarked = "".join(wm for wm, _ in wm_results)
    sampling_flat: List[Dict[str, Any]] = []
    for _, samp in wm_results:
        if samp:
            sampling_flat.extend(samp)
    return generated, watermarked, sampling_flat


async def _shutdown_async_gen_llm(gen_llm: Any) -> None:
    shutdown = getattr(gen_llm, "shutdown", None)
    if not callable(shutdown):
        return
    maybe = shutdown()
    if asyncio.iscoroutine(maybe):
        await maybe

def _resolve_batch_size(args: argparse.Namespace) -> int:
    """Mixed-batch capacity: max concurrent sequences in the shared vLLM engine."""
    return max(1, int(args.gen_max_inflight))


def _build_ratio_controller(args: argparse.Namespace) -> AdaptiveRatioController:
    return AdaptiveRatioController(
        r_min=float(args.wm_ratio_r_min),
        r_max=float(args.wm_ratio_r_max),
        b_max=float(args.wm_ratio_b_max),
        w_max=float(args.wm_ratio_w_max),
        k=int(args.wm_ratio_k),
    )


def _truncate_prompt(tokenizer: AutoTokenizer, query: str, max_inp_tokens: int) -> str:
    if not max_inp_tokens or max_inp_tokens <= 0:
        return query
    enc = tokenizer(
        query,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_inp_tokens,
    )
    return tokenizer.batch_decode(enc["input_ids"], skip_special_tokens=True)[0]


def _preflight_vllm_import() -> None:
    """Fail fast with a clear message if vLLM native extensions are broken."""
    try:
        _import_vllm()
    except RuntimeError as e:
        raise RuntimeError(
            f"{e}\n\n"
            "This is usually a torch/vLLM ABI mismatch in the active conda env.\n"
            "Fix wmm_env:\n"
            "  pip uninstall -y vllm torch torchvision torchaudio\n"
            "  pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install vllm==0.12.0\n"
            "Or use vllm_env and: pip install nltk"
        ) from e


def main(args):
    start_time_all = time.time()
    torch.cuda.empty_cache()

    # Validate inputs BEFORE initializing vLLM to avoid long warmup then crash.
    # Default source is GSM8K test (same examples as Generation_no_wm.py).
    if args.gen_max_inflight < 1:
        raise ValueError("--gen_max_inflight must be >= 1")

    gpu_id = int(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _setup_conda_cuda_for_flashinfer_link()

    model_name = args.model if hasattr(args, "model") and args.model else DEFAULT_MODEL
    print(f"Initializing single-GPU mixed serving with model: {model_name}")
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
    batch_size = _resolve_batch_size(args)
    ratio_ctrl = _build_ratio_controller(args)
    async_bundle = _build_async_gen_llm(
        model_name, cache_dir, engine_cfg=engine_cfg, max_num_seqs=batch_size
    )
    sync_gen_llm = None
    if async_bundle is None:
        print(
            "vLLM v1 AsyncLLM not found; using sync LLM with one generate() per prompt "
            "(no adaptive mixed streaming; same single engine for gen + watermark)."
        )
        sync_gen_llm, tokenizer, SamplingParams = _build_llm_and_tokenizer(
            model_name, cache_dir, engine_cfg=engine_cfg, max_num_seqs=batch_size
        )
    else:
        print(
            f"AsyncLLM (streaming DELTA) loaded; shared engine max_num_seqs={batch_size} "
            f"(mixed batch capacity; adaptive watermark ratio K={ratio_ctrl.k})."
        )
    print("Similarity scoring disabled: all LLM candidates share equal weight in sampling.")

    n_dev_main = torch.cuda.device_count()
    print(
        f"[GPU layout] Single-GPU mixed serving: physical GPU id={gpu_id}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, "
        f"torch.cuda.device_count()={n_dev_main}"
    )
    if n_dev_main > 0:
        print(f"[GPU layout] Device name: {torch.cuda.get_device_name(0)}")

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

    # Load prompts: GSM8K test (default), --data JSON list, or --prompt_pt (legacy C4).
    prompts: List[str] = []
    gsm8k_meta: List[Dict[str, Any]] = []
    use_chat_template = False
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
    elif args.prompt_pt:
        pt = torch.load(args.prompt_pt)
        if not pt or not isinstance(pt, (list, tuple)) or not pt[0]:
            raise ValueError("--prompt_pt does not look like the expected prompt dataset.")
        prompts = [str(x) for x in pt[0][: args.n_inputs]]
    else:
        gsm8k_meta = load_gsm8k_examples(
            cache_dir,
            n_inputs=args.n_inputs,
            split=str(args.dataset_split),
            dataset_id=str(args.dataset),
        )
        if not gsm8k_meta:
            raise ValueError("No GSM8K examples loaded.")
        prompts = [ex["prompt"] for ex in gsm8k_meta]
        use_chat_template = True
        print(
            f"Loaded GSM8K {args.dataset_split}: {len(prompts)} examples "
            f"(shared 5-shot train prompt; same order as Generation_no_wm.py)"
        )

    N_start = 0
    N_end = len(prompts)
    wq = engine_cfg.get("vllm_model_quantization") or "none"
    model_slug = _safe_output_slug(model_name)
    output_name = (
        f"{args.split}_{data_model}_Batching_wq{wq}_kv{engine_cfg['quantization']}_{engine_cfg['kv_cache_dtype']}_{engine_cfg['calculate_kv_scales']}_{args.split}_{data_model}_E2E_KEY_{secret_key}_m{m}_c{c}_h{h}_alpha{alpha}"
        f"_n{N_end}_in{args.max_inp_tokens}_new{args.max_new_tokens}"
        f"_gen_inflight_{args.gen_max_inflight}"
    )

    print(f"Starting end-to-end generation + watermarking with {len(prompts)} prompts")
    print(
        "Pipeline: single shared AsyncLLM engine on one GPU; request_type=generation and "
        "request_type=watermark are mixed via adaptive ratio control over a global FIFO queue."
    )
    print(f"Mixed batch capacity (max_num_seqs): {batch_size}")
    print(
        f"Adaptive watermark ratio: r_min={ratio_ctrl.r_min} r_max={ratio_ctrl.r_max} "
        f"B_max={ratio_ctrl.b_max} W_max={ratio_ctrl.w_max} K={ratio_ctrl.k}"
    )
    print(f"Watermark LLM decode: temperature={wm_temperature} top_p={wm_top_p}")

    pipeline_timing: Dict[str, Optional[float]] = {"t0": None, "t1": None}

    async def _run_all_prompts_async() -> None:
        assert async_bundle is not None
        gen_llm, SamplingParams, RequestOutputKind = async_bundle
        orchestrator = MixedServingOrchestrator(
            gen_llm=gen_llm,
            SamplingParams=SamplingParams,
            RequestOutputKind=RequestOutputKind,
            tokenizer=tokenizer,
            batch_size=batch_size,
            ratio_ctrl=ratio_ctrl,
            top_k=Top_K,
            secret_key=secret_key,
            m=m,
            c=c,
            h=h,
            alpha=alpha,
            wm_temperature=wm_temperature,
            wm_top_p=wm_top_p,
            wm_detect_max_tokens=wm_detect_max_tokens,
        )

        async def _pipeline_warmup_async() -> None:
            if not getattr(args, "warmup", True):
                return
            gp = _make_gen_sampling_params(args, max_tokens=max(1, int(args.warmup_max_tokens)))
            wp = _truncate_prompt(tokenizer, str(args.warmup_prompt), args.max_inp_tokens)
            await orchestrator.run_prompt(WARMUP_PROMPT_ID, wp, gp)
            orchestrator.wm_states.pop(WARMUP_PROMPT_ID, None)
            orchestrator._prompt_generated.pop(WARMUP_PROMPT_ID, None)
            print("[warmup] Completed one short gen+watermark run (excluded from pipeline wall time).")

        items: List[Tuple[int, str, Any]] = []
        for i, query in enumerate(prompts):
            text = render_chat_prompt(tokenizer, query) if use_chat_template else query
            q = _truncate_prompt(tokenizer, text, args.max_inp_tokens)
            items.append((i, q, _make_gen_sampling_params(args)))

        try:
            await _pipeline_warmup_async()
            pipeline_timing["t0"] = time.perf_counter()
            results = await orchestrator.run_all_prompts(items)
            pipeline_timing["t1"] = time.perf_counter()
            for prompt_id, original, watermarked, _samp in results:
                if gsm8k_meta:
                    row = {
                        "question": gsm8k_meta[prompt_id]["question"],
                        "answer": gsm8k_meta[prompt_id]["answer"],
                        "final_answer": gsm8k_meta[prompt_id]["final_answer"],
                        "Original_output": original,
                        "Original_output_answer": extract_output_answer(original),
                        "Watermarked_output": watermarked,
                        "Watermarked_output_answer": extract_output_answer(watermarked),
                    }
                else:
                    row = {
                        "question": items[prompt_id][1],
                        "answer": "",
                        "final_answer": "",
                        "Original_output": original,
                        "Original_output_answer": extract_output_answer(original),
                        "Watermarked_output": watermarked,
                        "Watermarked_output_answer": extract_output_answer(watermarked),
                    }
                generated_data.append(row)
        finally:
            await orchestrator.shutdown()

    if async_bundle is not None:
        asyncio.run(_run_all_prompts_async())
    else:
        assert sync_gen_llm is not None
        if getattr(args, "warmup", True):
            gp = _make_gen_sampling_params(args, max_tokens=max(1, int(args.warmup_max_tokens)))
            _sync_single_gpu_prompt(
                _truncate_prompt(tokenizer, str(args.warmup_prompt), args.max_inp_tokens),
                sync_gen_llm,
                tokenizer,
                gp,
                output_name,
                batch_size,
                Top_K,
                secret_key,
                m,
                c,
                h,
                alpha,
                wm_temperature=wm_temperature,
                wm_top_p=wm_top_p,
                wm_detect_max_tokens=wm_detect_max_tokens,
            )
            print("[warmup] Completed one short gen+watermark run (excluded from pipeline wall time).")
        pipeline_timing["t0"] = time.perf_counter()
        for i, query in enumerate(prompts):
            print(f"Processing item {i+1} / {N_end - N_start}")
            text = render_chat_prompt(tokenizer, query) if use_chat_template else query
            q = _truncate_prompt(tokenizer, text, args.max_inp_tokens)
            gen_params = _make_gen_sampling_params(args)
            generated_output_text, watermarked_output_text, _sampling_flat = _sync_single_gpu_prompt(
                q,
                sync_gen_llm,
                tokenizer,
                gen_params,
                output_name,
                batch_size,
                Top_K,
                secret_key,
                m,
                c,
                h,
                alpha,
                wm_temperature=wm_temperature,
                wm_top_p=wm_top_p,
                wm_detect_max_tokens=wm_detect_max_tokens,
                prompt_id=i,
            )
            if gsm8k_meta:
                row = {
                    "question": gsm8k_meta[i]["question"],
                    "answer": gsm8k_meta[i]["answer"],
                    "final_answer": gsm8k_meta[i]["final_answer"],
                    "Original_output": generated_output_text,
                    "Original_output_answer": extract_output_answer(generated_output_text),
                    "Watermarked_output": watermarked_output_text,
                    "Watermarked_output_answer": extract_output_answer(watermarked_output_text),
                }
            else:
                row = {
                    "question": q,
                    "answer": "",
                    "final_answer": "",
                    "Original_output": generated_output_text,
                    "Original_output_answer": extract_output_answer(generated_output_text),
                    "Watermarked_output": watermarked_output_text,
                    "Watermarked_output_answer": extract_output_answer(watermarked_output_text),
                }
            generated_data.append(row)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Whole Context Watermark Generation')
    # --- Data input ---
    parser.add_argument('--data', default=None, type=str, help="Optional JSON list of prompts (items may be strings or objects with 'input'/'Input'). If omitted with no --prompt_pt, loads GSM8K.")
    parser.add_argument('--dataset', default=GSM8K_DATASET_ID, type=str, help='HF dataset id used when neither --data nor --prompt_pt is set (default: openai/gsm8k).')
    parser.add_argument('--dataset_split', default='test', type=str, help='Dataset split for --dataset (default: test).')
    parser.add_argument('--data_model', default='Gemma', type=str, choices=['Llama3', 'Misrtal', 'DeepSeek', 'Qwen', 'Gemma'], help='Dataset model label for output naming')
    parser.add_argument('--split', default='Test', type=str, choices=['Test', 'Train'], help='Dataset split label for output naming (Test or Train)')
    parser.add_argument('--prompt_pt', default=None, type=str, help='Optional path to legacy .pt prompt dataset (e.g. c4_prompt_test.pt). If omitted with no --data, loads GSM8K.')
    parser.add_argument('--n_inputs', default=1000, type=int, help='Number of examples to take from GSM8K / --prompt_pt / --data')
    parser.add_argument('--max_inp_tokens', default=0, type=int, help='Max input tokens kept from each prompt before generation (0 = no truncation; needed for few-shot GSM8K prompts)')
    parser.add_argument('--max_new_tokens', default=200, type=int, help='Max new tokens to generate per prompt (short reasoning traces)')
    
    # --- Text generation parameters ---
    gen_sample = parser.add_argument_group('Text generation parameters')
    gen_sample.add_argument('--gen-temperature', default=0.0, type=float, help='Temperature for generation (0 = greedy). Passing only max_tokens used vLLM default 1.0 (sampling), which tends to longer / more variable outputs.')
    gen_sample.add_argument('--gen-top-p', dest='gen_top_p', default=1.0, type=float, help='top_p for generation.')
    
    # --- Watermarking parameters ---
    watermark = parser.add_argument_group('Watermarking parameters')
    watermark.add_argument('--wm-temperature', dest='wm_temperature', default=0.0, type=float, help='Temperature for watermark LLM detect/generate (0 = greedy).')
    watermark.add_argument('--wm-top-p', dest='wm_top_p', default=1.0, type=float, help='top_p for watermark LLM detect/generate.')
    watermark.add_argument('--wm-top-k', dest='wm_top_k', default=10, type=int, help='Top K synonym alternatives per target word.')
    watermark.add_argument('--wm-m', dest='wm_m', default=6, type=int, help='Number of tournament rounds.')
    watermark.add_argument('--wm-c', dest='wm_c', default=2, type=int, help='Number of competitors per tournament match.')
    watermark.add_argument('--wm-h', dest='wm_h', default=4, type=int, help='Left context tokens for tournament hashing.')
    watermark.add_argument('--wm-alpha', dest='wm_alpha', default=1.0, type=float, help='Softmax temperature for tournament draws (similarity is uniform).')
    watermark.add_argument('--secret_key', default='Adaptive_key_v1', type=str, help='Secret key for tournament randomization.')
    watermark.add_argument('--wm-detect-max-tokens', dest='wm_detect_max_tokens', default=200, type=int, help='Max tokens for watermark detect/generate JSON per sentence (raise if you see truncated JSON).')

   
    # --- Warmup parameters (No need to change) ---
    warmup_grp = parser.add_argument_group('Warmup (excluded from measured pipeline wall time)')
    warmup_grp.add_argument('--warmup', action=argparse.BooleanOptionalAction, default=True, help='Run one short gen+watermark request before the timed batch (JIT/CUDA graphs). Use --no-warmup to skip.')
    warmup_grp.add_argument('--warmup-prompt', default='Hello.', type=str, help='Input text for the warmup request.')
    warmup_grp.add_argument('--warmup-max-tokens', default=8, type=int, help='max_tokens for the warmup generation (keep small).')

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
    parser.add_argument('--gpu', default=0, type=int, help='Physical GPU id for the shared vLLM engine.')

    # --- Continuous batching / adaptive mixed serving ---
    batching = parser.add_argument_group('Continuous batching / adaptive mixed serving')
    batching.add_argument(
        '--gen_max_inflight',
        default=32,
        type=int,
        help='Mixed batch capacity: max concurrent sequences (generation + watermark) in the shared vLLM engine / adaptive ratio controller.',
    )
    batching.add_argument('--wm-ratio-k', dest='wm_ratio_k', default=5, type=int, help='Update watermark ratio r_t every K scheduling batches.')
    batching.add_argument('--wm-ratio-r-min', dest='wm_ratio_r_min', default=0.0, type=float, help='Minimum watermark slot ratio r_min.')
    batching.add_argument('--wm-ratio-r-max', dest='wm_ratio_r_max', default=1.0, type=float, help='Maximum watermark slot ratio r_max.')
    batching.add_argument('--wm-ratio-b-max', dest='wm_ratio_b_max', default=32, type=int, help='Queue depth normalization B_max for pressure.')
    batching.add_argument('--wm-ratio-w-max', dest='wm_ratio_w_max', default=1.5, type=float, help='Oldest-item wait normalization W_max (seconds) for pressure.')

    main(parser.parse_args())
