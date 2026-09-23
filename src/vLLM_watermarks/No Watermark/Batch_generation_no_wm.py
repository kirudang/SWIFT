import logging
import multiprocessing as mp
import sys
from pathlib import Path

import json
import os
import time

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Reuse the offline-LLM LoggingStatLogger helper from the KGW/SynthID throughput scripts.
_SYS_ROOT = Path(__file__).resolve().parents[1] / "vLLM_WM" / "scripts" / "gen_throughput"
if str(_SYS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SYS_ROOT))
from vllm_throughput_logging import PeriodicVllmStatsLogger  # noqa: E402

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Ensure the HF_HOME environment variable points to your desired cache location
    os.environ["HF_TOKEN"] = "Your HF Token"
    cache_dir = "Your Cache Directory"
    os.environ["HF_HOME"] = cache_dir

    # Force vLLM workers to use spawn (not fork)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    continuous_batch_size = 32
    base_model = "google/gemma-3-27b-it"  # Qwen/Qwen2.5-32B-Instruct" #"google/gemma-3-27b-it"
    saving_freq = 20
    max_inp = 50
    max_out = 1000
    Ninputs = 200
    output_name = f"Gemma-3-27b_no_wm_test_batched_{continuous_batch_size}_inp_{max_inp}_out_{max_out}_"
    stats_log_path = output_name + "_throughput.txt"
    log_stats_interval = 10.0  # seconds between Avg prompt/generation lines

    # Continuous-batch concurrency: keep up to this many sequences in flight.
    # As one finishes, vLLM immediately schedules the next waiting prompt.
    

    # Tee stdout/stderr for print()/tqdm. Also attach a logging FileHandler so
    # LoggingStatLogger INFO lines (KV cache %, prefix hit rate, ...) are saved:
    # those go through Python logging, which keeps the pre-Tee stderr stream.
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    stats_log_file = open(stats_log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, stats_log_file)
    sys.stderr = _Tee(sys.__stderr__, stats_log_file)

    _file_handler = logging.FileHandler(stats_log_path, mode="a", encoding="utf-8")
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(
        logging.Formatter("%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
                          datefmt="%m-%d %H:%M:%S")
    )
    # Root + common vLLM logger names used by LoggingStatLogger.
    for _name in ("", "vllm", "vllm.v1.metrics.loggers"):
        _lg = logging.getLogger(_name)
        _lg.setLevel(logging.INFO)
        _lg.addHandler(_file_handler)
        _lg.propagate = True

    print(f"[vllm stats] writing throughput lines to {stats_log_path}")
    print(
        "[vllm stats] Look for lines like: "
        "'Avg prompt throughput: ... GPU KV cache usage: ... Prefix cache hit rate: ...'"
        " — not the tqdm 'est. speed' counters."
    )

    # Initialize the tokenizer (still use HF tokenizer for truncation)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Initialize vLLM model with continuous batching capped at 32 seqs.
    # disable_log_stats=False keeps iteration stats recording on.
    llm = LLM(
        model=base_model,
        download_dir=cache_dir,
        max_num_seqs=continuous_batch_size,
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(
        max_tokens=max_out,  # number of new tokens to generate
    )

    # Load your data file
    data_path = "Your Data Path"
    data = torch.load(data_path)

    texts = list(data[0][:Ninputs])

    # Truncate all prompts upfront so the engine can refill freely
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=max_inp,
    )
    generated_prompts = tokenizer.batch_decode(
        encoded["input_ids"], skip_special_tokens=True
    )

    print(
        f"Submitting {len(generated_prompts)} prompts with "
        f"continuous batching max_num_seqs={continuous_batch_size}"
    )

    # Offline sync LLM does not print Avg * throughput by default; this installs
    # LoggingStatLogger and logs every log_stats_interval seconds from step().
    stats_logger = PeriodicVllmStatsLogger(
        llm,
        enabled=True,
        interval_s=log_stats_interval,
    )
    stats_logger.start()

    start_time = time.time()
    try:
        # One generate call: vLLM continuous-batches up to max_num_seqs at a time
        outputs = llm.generate(generated_prompts, sampling_params=sampling_params)
    finally:
        stats_logger.stop()
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_per_item = elapsed_time / len(generated_prompts)

    generated_data = []
    for i, (generated_prompt, req_out) in enumerate(
        zip(generated_prompts, outputs), start=1
    ):
        generated_output_text = req_out.outputs[0].text  # continuation only
        generated_text = generated_prompt + generated_output_text

        generated_data.append(
            {
                "input": generated_prompt,
                "input_output": generated_text,
                "output_only": generated_output_text,
                "time": time_per_item,
            }
        )

        if i % saving_freq == 0:
            prev_path = output_name + str(i - saving_freq) + ".json"
            if os.path.isfile(prev_path):
                os.remove(prev_path)

            with open(output_name + str(i) + ".json", "w") as json_file:
                json.dump(generated_data, json_file, indent=4)

            print(f"Saved checkpoint at {i}/{Ninputs}")

    # Final save
    with open(output_name + str(len(generated_data)) + ".json", "w") as json_file:
        json.dump(generated_data, json_file, indent=4)

    print(f"Total time taken: {elapsed_time} seconds")
    print(f"Total time taken per data: {elapsed_time / Ninputs} seconds")
    print(f"Throughput log saved to: {stats_log_path}")
    stats_log_file.close()
