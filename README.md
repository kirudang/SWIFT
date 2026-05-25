
# SWIFT Watermark
## Requirements
To facilitate the setup, we recommend creating a separate environment and installing the necessary packages from `requirements.txt`. The experiments were conducted on Python 3.10+, using **two NVIDIA GPUs** (one for text generation, one for watermarking) with PyTorch (`torch==2.5.1`), vLLM (`vllm==0.6.4.post1`), and CUDA 12.

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

## I. SWIFT Watermarking
All code related to our paper is located in the `src/` folder. Sample dataset notes and C4 download links are in `data/data.txt`. Instructions are as below:

### 1. End-to-End Generation and Watermarking
The repository root contains `src/Generation.py` (main entry point), `src/utils.py` (detect/generate synonyms, validation, tournament driver), and `src/Tournament_randomization.py` (deterministic HMAC tournament).

This pipeline **generates** continuations from prompts with vLLM, splits them into sentences, then **watermarks** each sentence via LLM-based synonym detection and tournament sampling. Generation and watermarking run on **separate GPUs** with bounded multiprocessing queues.

**First, set up the environment variables in `src/Generation.py` and `src/utils.py`:**
- `os.environ["HF_TOKEN"]` = `'Your_HuggingFace_Token'` # Hugging Face token for gated models
- `cache_dir` = `'Your/Cache/Directory'` # Directory for model caching (`HF_HOME`)

**Then run from the `src/` directory:**

```bash
cd src
python Generation.py [arguments]
```

**Data input parameters:**
- `--data`: Path to input JSON file (list of prompts). Each item may be a string or a dict with `input` or `Input`.
- `--prompt_pt`: Path to a Torch prompt file (used when `--data` is omitted; same layout as `c4_prompt_test.pt`)
- `--data_model`: Dataset model tag for output naming (choices: `'Llama3'`, `'Misrtal'`, `'DeepSeek'`, `'Qwen'`, `'Gemma'`, default: `'Llama3'`)
- `--split`: Dataset split (choices: `'Test'`, `'Train'`, default: `'Test'`)
- `--n_inputs`: Number of prompts to use from `--prompt_pt` (default: `20`)
- `--max_inp_tokens`: Max prompt tokens kept before generation (default: `50`)
- `--max_new_tokens`: Max new tokens to generate per prompt (default: `200`)

**Note**: For JSON input (`--data`), the dataset should be in one of the following formats:
```json
[
    {"input": "Your prompt here."},
    "Or a plain string prompt."
]
```

For C4 experiments, download the clean Train/Test `.pt` prompt files (see `data/data.txt`)

**Example (C4 prompts via `--prompt_pt`):**
```bash
cd src
python Generation.py \
  --prompt_pt /path/to/c4_prompt_test.pt \
  --data_model Llama3 \
  --split Test \
  --n_inputs 20 \
  --gen_gpu 0 \
  --wm_gpu 1 \
  --gen-temperature 0 \
  --wm-temperature 0 \
  --gen_max_inflight 32 \
  --wm_sentence_batch_size 32
```

**Example (JSON prompts via `--data`):**
```bash
cd src
python Generation.py \
  --data ../data/prompts.json \
  --n_inputs 100 \
  --gen_gpu 0 \
  --wm_gpu 1
```

**Text generation parameters (`--gen_gpu`):**
- `--model`: Hugging Face model id (default: `'meta-llama/Llama-3.1-8B-Instruct'`)
- `--gen-temperature`: Generation temperature (`0` = greedy, default: `0.0`)
- `--gen-top-p`: Nucleus sampling for generation (default: `1.0`)
- `--gen_max_inflight`: Max concurrent streaming generation requests (default: `32`)
- `--gen_gpu`: Physical GPU id for the generation engine (default: `0`)

**Watermarking / candidate parameters (`--wm_gpu`):**
- `--wm-temperature`: Temperature for watermark detect/generate JSON (`0` = greedy, default: `0.0`)
- `--wm-top-p`: top_p for watermark LLM (default: `1.0`)
- `--wm-detect-max-tokens`: Max tokens per sentence JSON (default: `512`; increase if JSON truncates)
- `--wm-top-k`: Max synonym candidates kept per target word (default: `15`)
- `--wm_gpu`: Physical GPU id for the watermark engine (default: `1`)
- `--wm_sentence_batch_size`: Sentences per watermark vLLM batch (default: `32`)
- `--wm_queue_maxsize`: Max queue size between gen and watermark processes (default: `256`)
- `--wm-log-queue`: Log approximate watermark queue depths (flag)

**Sampling parameters (tournament):**
- `--secret_key`: Secret key for randomization (default: `'Adaptive_key_v1'`)
- `--wm-m`: Number of tournament rounds (default: `6`)
- `--wm-c`: Number of competitors per tournament match (default: `2`)
- `--wm-h`: Left context tokens for tournament hashing (default: `4`)
- `--wm-alpha`: Softmax temperature for tournament draws (default: `1.0`)

**vLLM engine parameters:**
- `--dtype`: Model dtype, e.g. `bfloat16`, `auto` (default: `bfloat16`)
- `--vllm-model-quantization`: Weight quantization (optional)
- `--enable-prefix-caching`: Reuse KV for shared watermark instruct prefix (default: on)
- `--tensor_parallel_size`: Tensor parallel size (default: `1`)

**Pipeline timing:**
- `--warmup` / `--no-warmup`: Run one short gen+watermark request before timed batch (default: warmup on)
- `--warmup-prompt`: Warmup input text (default: `'Hello.'`)
- `--warmup-max-tokens`: Warmup generation length (default: `8`)

**Output format:**
The script generates a JSON file with the following structure for each item:
- `input`: Truncated input prompt
- `Original_output`: Generated text (sentences concatenated in order)
- `Watermarked_output`: Text after synonym replacements
- `time`: Processing time for this item

The output file is automatically saved in the working directory (`src/` when you `cd src`) with a name following the pattern:
`{model_slug}_Batching_wq{quant}_kv{kv_quant}_{kv_dtype}_{kv_scales}_{split}_{data_model}_E2E_KEY_{secret_key}_m{m}_c{c}_h{h}_alpha{alpha}_n{n}_in{max_inp}_new{max_new}_gen_inflight_{gen_inflight}_sentence_batch_{wm_batch}.json`

**Note**: Embedding similarity is **disabled** in this release; all synonym candidates receive uniform weight in the tournament. Word choice is **deterministic** given `--secret_key`, sentence context, and candidates (HMAC-based tournament), not random LLM sampling. Use `--gen-temperature 0` and `--wm-temperature 0` for greedy decoding (minor vLLM/CUDA nondeterminism may still occur).

**Optional environment variables:**
- `DETECT_MAX_TOKENS`: Override default max tokens for watermark JSON
- `DISABLE_STRUCTURED_OUTPUTS=1`: Disable vLLM structured JSON schema
- `WM_PARSE_DEBUG=1`: Print JSON parse failures
- `WM_TOURNAMENT_USE_THREADS=1`: Optional threaded tournament

**Troubleshooting:**
- Watermarked text equals original: JSON parse failed or fewer than 2 synonyms per word; try raising `--wm-detect-max-tokens`
- OOM on generation GPU: lower `--gen_max_inflight`
- OOM on watermark GPU: lower `--wm_sentence_batch_size`
- Generation stalls: watermark queue full; lower `--gen_max_inflight` or use a second GPU (`--wm_gpu` different from `--gen_gpu`)

## II. Other Watermark Implementation

We adhere to the original settings specified in their uploaded codes, allowing for straightforward replication. Please refer to the detailed guidance provided for each type of watermark by accessing the following resources:
- KGW: [KGW](https://github.com/jwkirchenbauer/lm-watermarking)
- SynthID: [SynthID](https://github.com/google-deepmind/synthid-text)
- SafeSeal: [SafeSeal](https://anonymous.4open.science/r/SafeSeal-8E76))


Enjoy the code!
