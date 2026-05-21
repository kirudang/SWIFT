# SWIFT
# vLLM Batching — Generation + Watermarking

End-to-end pipeline that **generates text with vLLM**, splits it into sentences, **watermarks each sentence** (synonym substitution), and writes JSON results. Designed for **two GPUs**: one engine for generation, one for watermarking, connected by bounded multiprocessing queues.

## What it does

For each input prompt:

1. **Generate** continuation text (streaming AsyncLLM when available).
2. **Split** the stream into sentences (`.`, `!`, `?` boundaries).
3. **Watermark** each sentence: detect replaceable words → LLM synonyms → tournament pick → apply replacements.
4. Save **`Original_output`** (generated) and **`Watermarked_output`** (after substitutions).

## Repository layout

| File | Purpose |
|------|---------|
| `Generation.py` | Main entry point: CLI, gen engine, queues, worker process, output JSON |
| `utils.py` | Watermark LLM (detect + synonyms), JSON parsing, validation, tournament driver |
| `Tournament_randomization.py` | Deterministic HMAC tournament to choose one synonym per target word |

## Requirements

- Python 3.10+
- CUDA GPUs (recommended: **2 GPUs** — `--gen_gpu` and `--wm_gpu`)
- [vLLM](https://docs.vllm.ai/) with AsyncLLM support (v1) for streaming generation

Example dependencies (adjust versions to your cluster):

```
torch>=2.5
transformers>=4.46
vllm>=0.6
nltk>=3.9
pydantic>=2.9
```

Set cache paths before running (edit in `Generation.py` / `utils.py` or via env):

- `HF_HOME` — Hugging Face model cache
- `HF_TOKEN` — if needed for gated models (do not commit tokens)

Download NLTK data once if needed:

```bash
python -c "import nltk; nltk.download('punkt')"
```

## Quick start

```bash
cd vLLM-batching

python Generation.py \
  --prompt_pt /path/to/c4_prompt_test.pt \
  --n_inputs 20 \
  --gen_gpu 0 \
  --wm_gpu 1 \
  --gen-temperature 0 \
  --wm-temperature 0 \
  --gen_max_inflight 32 \
  --wm_sentence_batch_size 32
```

Or from a JSON prompt list:

```bash
python Generation.py \
  --data /path/to/prompts.json \
  --n_inputs 100 \
  --gen_gpu 0 --wm_gpu 1
```

Results are written to `<output_name>.json` in the working directory (name is built from model, split, watermark key, batch sizes, etc.).

## Pipeline (high level)

```
Prompt
  → [Gen GPU] stream tokens (--gen-temperature, --max_new_tokens)
  → split into sentences
  → queue batches → [WM GPU] worker process
        → LLM: JSON {word: [synonyms]}  (utils.py)
        → tournament pick per word       (Tournament_randomization.py)
        → replace tokens in sentence     (apply_replacements)
  → queue back → stitch sentences in order
  → JSON {input, Original_output, Watermarked_output}
```

**Two separate vLLM engines** load the same `--model` on different GPUs. Generation and watermarking do not share one runtime instance.

## Main CLI groups

### Data

| Flag | Description |
|------|-------------|
| `--data` | JSON list of prompts (`input` / `Input` fields or strings) |
| `--prompt_pt` | Torch prompt file (default if `--data` omitted) |
| `--n_inputs` | Number of prompts to run |
| `--max_inp_tokens` | Truncate prompt length |
| `--max_new_tokens` | Max tokens to generate per prompt |

### Text generation (`--gen_gpu`)

| Flag | Default | Description |
|------|---------|-------------|
| `--gen-temperature` | `0.0` | `0` = greedy / deterministic decode |
| `--gen-top-p` | `1.0` | Nucleus sampling for generation |
| `--gen_max_inflight` | `32` | Max concurrent streaming prompts (also gen `max_num_seqs`) |

### Watermarking (`--wm_gpu`)

| Flag | Default | Description |
|------|---------|-------------|
| `--wm-temperature` | `0.0` | Greedy decode for detect/synonym JSON |
| `--wm-top-p` | `1.0` | top_p for watermark LLM |
| `--wm-detect-max-tokens` | `512` | Max tokens per sentence JSON (increase if JSON truncates) |
| `--wm-top-k` | `15` | Max synonyms kept per target word |
| `--wm-m`, `--wm-c` | `6`, `2` | Tournament rounds and bracket size |
| `--wm-h` | `4` | Left context tokens for tournament hashing |
| `--wm-alpha` | `1.0` | Softmax temperature in tournament (similarities are uniform) |
| `--secret_key` | `Adaptive_key_v1` | Secret key for deterministic tournament |
| `--wm_sentence_batch_size` | `32` | Sentences per watermark vLLM batch |

### Queues / batching

| Flag | Default | Description |
|------|---------|-------------|
| `--wm_queue_maxsize` | `256` | Backpressure between gen and watermark processes |
| `--wm-log-queue` | off | Log approximate queue depths |

### Model / vLLM

| Flag | Description |
|------|-------------|
| `--model` | Hugging Face model id (default: `meta-llama/Llama-3.1-8B-Instruct`) |
| `--dtype` | e.g. `bfloat16`, `auto` |
| `--vllm-model-quantization` | Weight quant (e.g. W4A16 checkpoint) |
| `--enable-prefix-caching` | Reuse KV for shared watermark prompt prefix |

## Output format

Each item in the output JSON array:

```json
{
  "input": "<truncated prompt>",
  "Original_output": "<full generated text, sentences concatenated in order>",
  "Watermarked_output": "<same structure with synonym replacements>",
  "time": 0.0
}
```

## Determinism notes

- **Generation:** `--gen-temperature 0` → greedy decode (subject to vLLM/CUDA nondeterminism).
- **Watermark LLM:** `--wm-temperature 0` → greedy JSON generation.
- **Word choice:** Tournament is **deterministic** given `--secret_key`, sentence context, and candidates (HMAC-based), not random LLM sampling.

Embedding similarity is **disabled**; all candidates use uniform weight in the tournament.

## Environment variables

| Variable | Effect |
|----------|--------|
| `DETECT_MAX_TOKENS` | Default max tokens for watermark JSON (overrides code default if set) |
| `DISABLE_STRUCTURED_OUTPUTS=1` | Disable vLLM structured JSON schema |
| `WM_PARSE_DEBUG=1` | Print JSON parse failures (otherwise skipped silently) |
| `WM_TOURNAMENT_USE_THREADS=1` | Optional threaded tournament (usually off) |

## Troubleshooting

| Issue | Try |
|-------|-----|
| Watermarked text equals original | JSON parse failed or &lt;2 synonyms per word; raise `--wm-detect-max-tokens` |
| OOM on gen GPU | Lower `--gen_max_inflight` |
| OOM on wm GPU | Lower `--wm_sentence_batch_size` |
| Gen waits / stalls | Watermark slower than gen; normal with full queue — lower `--gen_max_inflight` or speed up wm GPU |
| AsyncLLM not found | Falls back to sync one-shot generation per prompt (no token streaming) |

## License


