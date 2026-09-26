Code for **SWIFT**: Adaptive Co-Serving LLM Watermarking on Modern Inference Engines
## Requirements
To facilitate the setup, we recommend creating a separate environment and installing the necessary packages from `requirements.txt`. The experiments were conducted on Python 3.10+, PyTorch (`torch==2.5.1`), vLLM (`vllm==0.6.4.post1`), and CUDA 12.


```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

Pinned packages in `requirements.txt` include `torch==2.5.1`, `vllm==0.6.4.post1`, and `transformers==4.46.3`. Utility scripts that call OpenAI also need the `openai` and `openpyxl` packages.

Before running generation or baseline scripts, set placeholders in each entry script:

- `os.environ["HF_TOKEN"]` → your Hugging Face token (for gated models)
- `cache_dir` / `HF_HOME` → local model/dataset cache directory

---

## Repository layout

```
SWIFT/
├── data/                          # Dataset download helpers for GSM8K, MedMCQA, CNN/DailyMail + C4 links
├── requirements.txt
└── src/
    ├── Generation/                # SWIFT end-to-end generate + watermark
    │   ├── Text_generation/      # C4 / free-form continuation
    │   ├── Text_summarization/   # CNN/DailyMail summarization
    │   ├── Short_text_GSM8K/      # GSM8K math reasoning
    │   └── Mecial_text_MedMCQA/   # MedMCQA medical QA
    ├── Detection/                 # Train / evaluate a watermark detector
    ├── Utility/                   # OpenAI semantic score & pairwise judge
    └── vLLM_watermarks/           # Baseline generation (No-WM, KGW, SynthID)
```

Each SWIFT task folder contains:

| File | Role |
|------|------|
| `Generation.py` | Main watermark generation file|
| `utils.py` | Synonym detect/generate, validation, tournament driver |
| `Tournament_randomization.py` | Deterministic HMAC tournament |
| `mixed_serving.py` | Shared vLLM scheduling for gen + watermark |
| `wm_sentence_batcher.py` | Sentence micro-batching helpers |
| `*_prompts.py` | Task-specific few-shot prompts (GSM8K / MedMCQA) |

---

## I. Data preparation

### C4 (text generation)

Clean Train/Test `.pt` prompt files used in experiments (Google Drive): see [`data/data_c4.txt`](data/data_c4.txt).

### CNN / Daily Mail (summarization)

```bash
python data/CNN_data.py
```

Edit `cache_dir` / `HF_TOKEN` and `save_path` at the top of the script. Writes a local JSON (`cnn.json` by default).

### GSM8K

```bash
python data/GSM8K.py
```

Downloads `openai/gsm8k` (`main`) and saves a DatasetDict under `data/openai_gsm8k`.

### MedMCQA

```bash
python data/MedMCQA.py
```

Downloads `openlifescienceai/medmcqa`, filters to single-choice rows with a valid `cop` and explanation longer than 50 words, and saves under `data/openlifescienceai_medmcqa`.

---

## II. SWIFT generation & watermarking

Each task uses a **single shared vLLM engine**. Generation and watermarking share that engine via mixed serving.

**First**, set `HF_TOKEN` and `cache_dir` in that task’s `Generation.py` and `utils.py`.

### 1. Text generation (C4 / JSON / `.pt` prompts)

```bash
cd src/Generation/Text_generation
python Generation.py \
  --prompt_pt /path/to/c4_prompt_test.pt \
  --data_model Gemma \
  --split Test \
  --n_inputs 1000 \
  --max_inp_tokens 50 \
  --max_new_tokens 200 \
  --gpu 0 \
  --gen-temperature 0 \
  --wm-temperature 0 \
  --gen_max_inflight 32 \
  --wm_sentence_batch_size 64
```

Or pass a JSON list of prompts:

```bash
python Generation.py --data /path/to/prompts.json --n_inputs 100 --gpu 0
```

JSON items may be plain strings or objects with `input` / `Input`.

### 2. Text summarization (CNN/DailyMail)

```bash
cd src/Generation/Text_summarization
python Generation.py \
  --data /path/to/cnn.json \
  --split Test \
  --n_inputs 1000 \
  --max_inp_tokens 1024 \
  --max_new_tokens 100 \
  --gpu 0
```

### 3. Short text / GSM8K

Loads GSM8K from the HF cache / on-disk save by default (or use `--data` / `--prompt_pt`).

```bash
cd src/Generation/Short_text_GSM8K
python Generation.py \
  --dataset_split test \
  --n_inputs 1000 \
  --max_new_tokens 200 \
  --gpu 0
```

### 4. Medical text / MedMCQA

```bash
cd src/Generation/Mecial_text_MedMCQA
python Generation.py \
  --dataset_split validation \
  --n_inputs 1000 \
  --max_new_tokens 200 \
  --gpu 0
```

### Shared SWIFT arguments (all tasks)
Please keep them defaults as we already set.

**Generation**

| Flag | Meaning |
|------|---------|
| `--model` | HF model id (default: `google/gemma-3-27b-it`) |
| `--gen-temperature` / `--gen-top-p` | Decoding (use `0` for greedy) |
| `--max_inp_tokens` / `--max_new_tokens` | Prompt truncate / generation length |
| `--n_inputs` | Number of examples |

**Watermark / tournament**

| Flag | Meaning |
|------|---------|
| `--secret_key` | HMAC key (default: `Adaptive_key_v1`) |
| `--wm-m` / `--wm-c` / `--wm-h` / `--wm-alpha` | Tournament rounds, competitors, context, softmax temp |
| `--wm-top-k` | Max synonym candidates per word |
| `--wm-detect-max-tokens` | Max tokens for synonym JSON (raise if truncated) |
| `--wm-temperature` / `--wm-top-p` | Watermark LLM decoding |

**Serving / batching**

| Flag | Meaning |
|------|---------|
| `--gpu` | Physical GPU for the shared engine |
| `--gen_max_inflight` / `--max_num_seqs` | Concurrent sequences |
| `--wm_sentence_batch_size` | Synonym detect/generate micro-batch size |
| `--wm-ratio-*` | Adaptive gen vs watermark slot ratio |

**Output fields** (JSON list written in the working directory):

- `input` — prompt (truncated as configured)
- `Original_output` — generated text before watermarking
- `Watermarked_output` — text after synonym replacements
- `time` — per-item processing time

**Notes**

- Embedding similarity is disabled; tournament candidates use uniform weights. Word choice is deterministic given `--secret_key`, context, and candidates.
- Prefer `--gen-temperature 0` and `--wm-temperature 0` for greedy decoding (minor vLLM/CUDA nondeterminism may still occur).
- If watermarked text equals original: raise `--wm-detect-max-tokens`, or check that enough synonyms (≥2) were returned.
- OOM: lower `--gen_max_inflight` or `--wm_sentence_batch_size`.

Optional env vars: `DETECT_MAX_TOKENS`, `DISABLE_STRUCTURED_OUTPUTS=1`, `WM_PARSE_DEBUG=1`, `WM_TOURNAMENT_USE_THREADS=1`.

---

## III. Detection

Build detector datasets, train, then run inference:

```bash
cd src/Detection
python 1.dataset.py      # build train/test JSONL from original vs watermarked texts
python 2.training.py     # train detector (set HF token / cache_dir / data paths)
python 3.inference.py    # evaluate on held-out JSONL
```

Edit paths and `HF_TOKEN` / `cache_dir` at the top of `2.training.py` and `3.inference.py` before running.

---

## IV. Utility evaluation (OpenAI)

Semantic preservation score (0–5) and pairwise win-rate judges:

```bash
cd src/Utility
# Edit OPENAI_API_KEY, MODEL, DATA_PATH_*, FIELD_* in each script
python Utility.py      # pairwise semantic similarity → Excel
python Pairwise.py     # A vs B watermark utility judge → JSON
```

---

## V. Baseline watermarks (vLLM)

KGW and SynthID follow the vLLM watermarking approach from [vLLM_WM](https://vermaapurv.com/2025-10-04-vllm-watermark/), with continuous batching across requests added for a fair comparison against SWIFT.

Scripts under `src/vLLM_watermarks/` generate **no-watermark**, **KGW**, and **SynthID** outputs for the same task families (C4, summarization, GSM8K, MedMCQA), plus simple detection helpers for KGW and SynthID.

```
src/vLLM_watermarks/
├── No Watermark/Batch_generation_no_wm.py
├── KGW/
│   ├── KGW_batch_generation_C4.py
│   ├── KGW_batch_generation_Summarization.py
│   ├── KGW_batch_generation_GSM8K.py
│   ├── KGW_batch_generation_Med.py
│   └── KGW_detection.py
└── SynthID/
    ├── SynthID_batch_generation_C4.py
    ├── SynthID_batch_generation_summarization.py
    ├── SynthID_batch_generation_GSM8K.py
    ├── SynthID_batch_generation_MedMCQA.py
    └── SynthID_detection.py
```

Set `HF_TOKEN`, `cache_dir`, and data paths at the top of each script (or via CLI defaults), then run from that folder.

For **SafeSeal**, generate unwatermarked text with `No Watermark/Batch_generation_no_wm.py`, then apply watermarking with [SafeSeal](https://anonymous.4open.science/r/SafeSeal-8E76).

---

Enjoy the code, and feel free to open an issue!

