"""
Watermark utilities: LLM detect/generate synonym candidates, tournament sampling, validation.

Per-candidate embedding similarity is disabled; tournament draws use uniform weights (1.0).
LLM prompt ordering ("best first") is not passed into the tournament softmax.
"""

import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import json
import os
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple


from nltk.tokenize import TreebankWordTokenizer

# HF / vLLM worker env (watermark subprocess imports this module directly)
os.environ["HF_TOKEN"] = "Your HF Token"
cache_dir = 'Your Cache Directory'
os.environ["HF_HOME"] = cache_dir
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Structured decoding can trigger nanobind refcount crashes on some vLLM builds
_DISABLE_STRUCTURED_OUTPUTS = os.environ.get("DISABLE_STRUCTURED_OUTPUTS", "0").strip().lower() in {
    "1", "true", "yes", "y",
}
_DETECT_MAX_TOKENS = int(os.environ.get("DETECT_MAX_TOKENS", "512"))

try:
    from vllm import SamplingParams

    try:
        from vllm.sampling_params import StructuredOutputsParams

        STRUCTURED_OUTPUTS_AVAILABLE = True
    except ImportError:
        StructuredOutputsParams = None  # type: ignore[assignment]
        STRUCTURED_OUTPUTS_AVAILABLE = False
    VLLM_AVAILABLE = True
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]
    StructuredOutputsParams = None  # type: ignore[assignment]
    STRUCTURED_OUTPUTS_AVAILABLE = False
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Please install vllm package.")

try:
    from pydantic import BaseModel

    try:
        from pydantic import RootModel  # type: ignore
    except Exception:
        RootModel = None  # type: ignore[assignment]
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = object  # type: ignore[assignment]
    RootModel = None  # type: ignore[assignment]
    PYDANTIC_AVAILABLE = False

_TREEBANK_TOKENIZER = TreebankWordTokenizer()

# Uniform tournament weights (similarity scoring disabled in this pipeline)
_UNIFORM_SIMILARITY = 1.0

TargetKey = Tuple[str, str, int, int]  # (sentence, target_word, start_i, end_i)


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Return token text with character spans using one shared tokenizer."""
    spans = list(_TREEBANK_TOKENIZER.span_tokenize(text))
    return [(text[s:e], s, e) for s, e in spans]


def tokenize_words(text: str) -> List[str]:
    """Return tokens using the same tokenizer used for replacement spans."""
    return [tok for tok, _, _ in tokenize_with_spans(text)]


def apply_replacements(sentence: str, replacements: List[Tuple[int, int, str, str]]) -> str:
    """Apply word/phrase replacements while preserving original formatting."""
    token_items = tokenize_with_spans(sentence)
    if not token_items:
        return sentence

    updated_sentence = sentence
    for start_pos, end_pos, target, replacement in sorted(
        replacements, key=lambda x: x[0], reverse=True
    ):
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


if PYDANTIC_AVAILABLE:
    if RootModel is not None:

        class SynonymMap(RootModel[Dict[str, List[str]]]):  # type: ignore[misc]
            """JSON schema: {word: [syn1, syn2, ...]}."""

            pass

    else:

        class SynonymMap(BaseModel):  # type: ignore[misc]
            """JSON schema: {word: [syn1, syn2, ...]} (Pydantic v1)."""

            __root__: Dict[str, List[str]]


def _get_synonym_json_schema() -> Optional[Dict[str, Any]]:
    """Build JSON schema for vLLM StructuredOutputsParams, or None to fall back."""
    if not PYDANTIC_AVAILABLE:
        return None

    try:
        if hasattr(SynonymMap, "model_json_schema"):  # type: ignore[attr-defined]
            return SynonymMap.model_json_schema()  # type: ignore[call-arg]
        if hasattr(SynonymMap, "schema"):  # type: ignore[attr-defined]
            return SynonymMap.schema()  # type: ignore[call-arg]
    except Exception as e:
        print(f"Error building synonym JSON schema: {e}")
        return None

    return None


def filter_bad_candidates(synonyms: List[str], target_word: str) -> List[str]:
    """Filter obviously bad synonym candidates; allow short multi-word phrases."""
    if not synonyms:
        return []

    filtered = []
    target_lower = target_word.strip().lower()

    for synonym in synonyms:
        if not synonym or not isinstance(synonym, str):
            continue

        synonym_stripped = synonym.strip()
        synonym_lower = synonym_stripped.lower()

        if synonym_lower == target_lower:
            continue
        if len(synonym_stripped) < 1:
            continue
        # Skip unreasonably long strings (likely generation errors)
        if len(synonym) > 50:
            continue

        normalized = synonym_stripped.replace("-", "").replace("'", "").replace(" ", "")
        if (not normalized.isalnum()) and len(synonym_stripped) < 3:
            continue

        filtered.append(synonym)

    return filtered if filtered else synonyms


# Shared detect/generate prompt prefix (enables vLLM prefix/KV cache reuse when batching)
_DETECT_GENERATE_SYSTEM = {
    "role": "system",
    "content": (
        "You are an expert linguist specializing in context-aware lexical substitution. "
        "Given exactly ONE English sentence, output ONLY a valid JSON object and no extra text. "
        "Your job is to identify contextually WATERMARKABLE words and generate high-quality synonym candidates. "
        "A word is WATERMARKABLE only if it can be replaced in that exact sentence by a natural synonym "
        "without changing meaning, factual content, tone, tense, number, degree, or register. "
        "Be conservative: when unsure, exclude the word. "
        "The input may be a medical multiple-choice explanation: apply the watermark by minimally modifying "
        "eligible ordinary natural-language words only; preserve numbers, lab values, dosages, units, "
        "chemical and drug names, gene/protein names, anatomical terms, disease names, "
        "highly technical medical terminology, option letters (A/B/C/D), and the final answer. "
        "Never output invalid JSON, explanations, comments, markdown, or duplicate candidates."
    ),
}

_DETECT_GENERATE_USER_PREFIX = """Task: For the given sentence, identify WATERMARKABLE words and generate synonym candidates.

Definition (WATERMARKABLE):
- Select only content words that can be naturally replaced in THIS exact sentence without changing meaning.
- Allowed POS classes: nouns, verbs, adjectives, adverbs.
- Return ONLY words that appear literally in the sentence as keys (exact surface form, case preserved).
- Never select capitalized words as watermarkable targets; treat them as entity/protected tokens.

Core decision rule:
A word is WATERMARKABLE only if BOTH are true:
1. It is not part of a protected span.
2. It can be replaced directly in this sentence by a natural synonym without changing meaning, factual content, tone, register, tense, number, or degree.

Protected-span rule:
- A protected span is any phrase that functions as a specific name, official label, fixed title, branded term, identifier, or other identity-bearing expression.
- If replacing one word would damage the identity of the phrase, exclude the whole span and all words inside it.
- Never output partial fragments of protected spans, even if an individual word looks like a common noun or adjective by itself.
- Any capitalized word should be treated as an entity/protected token and excluded from output keys.

Examples of protected spans:
- Person / place / organization names
- Product, software, platform, model, or hardware names
- Degree names, job titles, document titles, event names
- Fixed institutional or branded phrases
- Identifiers, codes, versions, dates, times, quantities, and model numbers
- Drug / chemical / compound names (e.g., vancomycin, Pseudomonas aeruginosa, TNF, IL-1)
- Gene, protein, pathway, and biomarker names
- Anatomical structures, disease names, syndrome names, and procedure names
- Lab values, dosages, concentrations, scores, staging systems, and units
- Multiple-choice option letters and the final answer choice (A/B/C/D)

Medical reasoning constraints (when the sentence is a MedMCQA-style explanation):
- The input is a medical question explanation / clinical reasoning trace.
- Apply the watermark by minimally modifying eligible ordinary natural-language words only.
- Preserve: numbers, lab values, dosages, units, chemical/drug names, gene/protein names,
  anatomical terms, disease names, highly technical medical terms, diagnostic criteria,
  causal/clinical relationships, option letters, and the final answer.
- Do not add or remove clinical reasoning steps.
- Do not change medical meaning, diagnosis, or the selected answer choice.
- Only modify everyday connective or descriptive wording where necessary for watermarking.

Candidate requirements:
- For each selected key, output 1-6 synonym candidates, best first.
- Candidates must be direct substitutes in context, not merely related words.
- Preserve meaning, factuality, tone, register, tense, number, and degree.
- Prefer single-word candidates; allow short multi-word candidates only if they are common and natural in context.
- Do NOT include the original word itself.
- Do NOT include duplicates.
- Do NOT include antonyms, broader categories, narrower categories, or loosely related words.
- If no strong synonym fits naturally in the sentence, omit the key entirely.

Quality filter:
- Keep only candidates that could replace the word in the sentence with minimal or no grammatical adjustment.
- If a candidate sounds awkward, changes the collocation, or weakens medical/technical meaning, exclude it.
- Prefer precision over recall: when unsure, exclude the word.
- Output fewer keys if necessary; do not force coverage.

Output format:
- Return ONLY one valid JSON object.
- Keys = selected words from the sentence.
- Values = arrays of synonym candidates.

Few-shot examples:

Example 1:
Sentence: On Tuesday, Dr. Sarah Chen from Stanford University delivered a clear presentation about climate policy, and the audience responded warmly to her practical recommendations.
Output: {"clear":["concise","understandable","lucid"],"presentation":["talk","lecture","briefing"],"responded":["reacted","replied"],"warmly":["enthusiastically","favorably"],"practical":["useful","realistic","workable"],"recommendations":["suggestions","proposals","advice"]}

Example 2:
Sentence: After the meeting at Microsoft headquarters in Seattle, the engineers quickly resolved the urgent problem and submitted a detailed report before noon.
Output: {"meeting":["session","discussion"],"engineers":["developers","specialists"],"quickly":["rapidly","promptly","swiftly"],"resolved":["solved","fixed","addressed"],"urgent":["pressing","critical","immediate"],"problem":["issue","difficulty","matter"],"submitted":["sent","filed","delivered"],"detailed":["thorough","comprehensive"],"report":["summary","document"]}

Example 3:
Sentence: IBM officials said the new cluster is designed to offer improved performance and scalability compared to traditional servers, as well as simplified management and lower total cost of ownership.
Output: {"new":["recent","latest","modern"],"cluster":["system","setup","platform"],"improved":["enhanced","optimized","upgraded"],"performance":["efficiency","speed","effectiveness"],"scalability":["expandability","adaptability"],"traditional":["conventional","standard","legacy"],"simplified":["streamlined","easier"],"management":["administration","oversight"],"lower":["reduced","decreased"]}

Example 4:
Sentence: The cluster includes IBM's System Storage DR550 and DR850 disk arrays, which offer scalable storage capacity and performance for high-end computing environments.
Output: {"cluster":["system","setup","configuration"],"includes":["contains","comprises","features"],"disk":["drive"],"arrays":["sets","units"],"scalable":["expandable","adaptable"],"storage":["capacity space","data storage"],"capacity":["volume","size"],"performance":["speed","efficiency"],"computing":["processing"],"environments":["settings","contexts"]}

Example 5:
Sentence: Ms. Rinaldi holds a Master of Library Science degree from Rutgers University and a Bachelor of Arts degree from Seton Hall University. She is a member of the New Jersey Library Association and has presented at various library conferences on topics such as marketing, programming, and collection development.
Output: {"holds":["possesses","has"],"degree":["credential","qualification"],"member":["participant","affiliate"],"presented":["spoke","lectured"],"various":["several","different"],"library":["research","information"],"conferences":["meetings","events"],"topics":["subjects","issues"],"marketing":["promotion","outreach"],"development":["growth","expansion"]}

Contrastive notes from the examples:
- In Example 4, words inside the product name span are excluded, but ordinary descriptive words outside that span may still be selected.
- In Example 5, words inside degree names, university names, and association names are excluded, but "library" in "library conferences" may be selected because there it functions as an ordinary descriptive word rather than part of a protected span.
- A common-looking word may still be excluded if it belongs to a protected span in that sentence.
- Output fewer keys if necessary; precision is more important than recall.

Now process this sentence:

Sentence:
"""

_DETECT_GENERATE_USER_SUFFIX = """

Output (JSON object only, no markdown or explanation):"""


def _get_sentence_tokens(
    sentence: str,
    token_cache: Optional[Dict[str, List[str]]],
) -> List[str]:
    if token_cache is not None and sentence in token_cache:
        return token_cache[sentence]
    tokens = tokenize_words(sentence)
    if token_cache is not None:
        token_cache[sentence] = tokens
    return tokens


def _normalize_word_candidates(obj: Any) -> Dict[str, List[str]]:
    """Parse a JSON object into {word: [candidates...]}."""
    if not isinstance(obj, dict):
        return {}

    result: Dict[str, List[str]] = {}
    for word, cands in obj.items():
        if not isinstance(word, str):
            continue
        word = word.strip()
        if not word:
            continue
        cleaned: List[str] = []
        if isinstance(cands, list):
            for c in cands:
                if not c:
                    continue
                c_str = str(c).strip()
                if c_str:
                    cleaned.append(c_str)
        if cleaned:
            result[word] = cleaned
    return result


def build_sampling_params_for_synonyms(
    max_tokens: Optional[int] = None,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Tuple[Any, bool]:
    """Public wrapper for watermark detect/generate SamplingParams."""
    return _build_sampling_params_for_synonyms(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def _build_sampling_params_for_synonyms(
    max_tokens: Optional[int] = None,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Tuple[Any, bool]:
    """
    SamplingParams for synonym JSON generation.
    Prefer structured outputs when available; otherwise greedy decode (temperature=0).
    """
    if max_tokens is None:
        max_tokens = _DETECT_MAX_TOKENS

    if (not _DISABLE_STRUCTURED_OUTPUTS) and VLLM_AVAILABLE and STRUCTURED_OUTPUTS_AVAILABLE:
        json_schema = _get_synonym_json_schema()
        if json_schema is not None and StructuredOutputsParams is not None:
            try:
                return SamplingParams(
                    temperature=float(temperature),
                    max_tokens=max_tokens,
                    top_p=float(top_p),
                    structured_outputs=StructuredOutputsParams(json=json_schema),
                ), True
            except TypeError:
                pass  # older vLLM without structured_outputs= or top_p

    return SamplingParams(
        max_tokens=max_tokens,
        temperature=float(temperature),
        top_p=float(top_p),
    ), False


def _parse_debug_enabled() -> bool:
    return os.environ.get("WM_PARSE_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y"}


def _extract_json_object_slice(decoded: str) -> Optional[str]:
    """Pull the outermost {...} block, skipping markdown fences when present."""
    text = decoded.strip()
    if not text:
        return None

    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and "}" in part:
                text = part
                break

    start_idx = text.find("{")
    if start_idx == -1:
        return None

    bracket_count = 0
    end_idx = start_idx
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            bracket_count += 1
        elif text[i] == "}":
            bracket_count -= 1
            if bracket_count == 0:
                end_idx = i + 1
                break

    if bracket_count != 0:
        return None
    return text[start_idx:end_idx]


def _parse_detect_and_generate_output(decoded: str, strict_json_only: bool = False) -> Dict[str, List[str]]:
    """
    Parse LLM output into {word: [candidates...]}.
    On failure returns {} and skips watermarking for that sentence (no console spam).
    Set WM_PARSE_DEBUG=1 to print parse failures.
    """
    text = (decoded or "").strip()
    if not text:
        return {}

    errors: List[str] = []

    if strict_json_only:
        try:
            return _normalize_word_candidates(json.loads(text))
        except json.JSONDecodeError as e:
            errors.append(str(e))

    json_slice = _extract_json_object_slice(text)
    if json_slice:
        try:
            return _normalize_word_candidates(json.loads(json_slice))
        except json.JSONDecodeError as e:
            errors.append(str(e))

    if _parse_debug_enabled() and errors:
        print(f"[wm-parse] detect-and-generate JSON failed: {errors[-1]}")
    return {}


def _map_candidates_to_targets(
    sentence: str,
    word_to_candidates: Dict[str, List[str]],
    tokens: List[str],
    Top_K: int,
) -> Dict[TargetKey, List[str]]:
    """Map LLM JSON keys to token positions; keep targets with >= 2 filtered candidates."""
    results: Dict[TargetKey, List[str]] = {}
    for word, candidates in word_to_candidates.items():
        for i, tok in enumerate(tokens):
            if tok.lower() != word.lower():
                continue
            filtered = filter_bad_candidates(candidates, tok)
            if len(filtered) >= 2:
                results[(sentence, tok, i, i)] = filtered[:Top_K]
    return results


def _make_sampling_record(
    target: str,
    alternatives_list: List[str],
    randomized_word: str,
) -> Dict[str, Any]:
    """Log entry for one replacement; similarity field kept for schema compat (always None)."""
    return {
        "word": target,
        "alternatives": alternatives_list,
        "alternatives_with_similarity": [
            {"word": alt, "similarity": None} for alt in alternatives_list
        ],
        "randomized_word": randomized_word,
    }


def _build_detect_generate_prompt(sentence: str) -> str:
    user_content = _DETECT_GENERATE_USER_PREFIX + sentence + _DETECT_GENERATE_USER_SUFFIX
    messages = [_DETECT_GENERATE_SYSTEM, {"role": "user", "content": user_content}]
    return messages


def llm_detect_and_generate_candidates_batch(
    sentences: List[str],
    llm: Any,
    tokenizer: Any,
    Top_K: int = 15,
    batch_size: int = 8,
    token_cache: Optional[Dict[str, List[str]]] = None,
    *,
    wm_temperature: float = 0.0,
    wm_top_p: float = 1.0,
    wm_detect_max_tokens: Optional[int] = None,
) -> Dict[TargetKey, List[str]]:
    """
    Detect watermarkable words and generate synonyms for multiple sentences.
    Shared prompt prefix improves throughput via vLLM prefix/KV-cache reuse.
    """
    if not VLLM_AVAILABLE or llm is None or not sentences:
        return {}

    valid_sentences = [s for s in sentences if s and is_valid_sentence(s, token_cache=token_cache)]
    if not valid_sentences:
        return {}

    all_results: Dict[TargetKey, List[str]] = {}

    for i in range(0, len(valid_sentences), batch_size):
        batch_sentences = valid_sentences[i : i + batch_size]
        prompts = []
        for sentence in batch_sentences:
            messages = _build_detect_generate_prompt(sentence)
            prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        sampling_params, structured_used = _build_sampling_params_for_synonyms(
            max_tokens=wm_detect_max_tokens,
            temperature=wm_temperature,
            top_p=wm_top_p,
        )
        outputs = llm.generate(prompts, sampling_params=sampling_params)

        for sentence, output in zip(batch_sentences, outputs):
            decoded = output.outputs[0].text.strip() if output.outputs else ""
            word_to_candidates = _parse_detect_and_generate_output(
                decoded, strict_json_only=structured_used
            )
            if not word_to_candidates:
                continue

            tokens = _get_sentence_tokens(sentence, token_cache)
            all_results.update(
                _map_candidates_to_targets(sentence, word_to_candidates, tokens, Top_K)
            )

    return all_results


def parallel_tournament_sampling(
    target_results: Dict[TargetKey, List[str]],
    secret_key: str,
    m: int,
    c: int,
    h: int,
    alpha: float,
    token_cache: Optional[Dict[str, List[str]]] = None,
) -> Dict[TargetKey, Optional[str]]:
    """
    Run tournament sampling per target. Similarity is uniform (_UNIFORM_SIMILARITY);
    alpha is passed through for API compatibility with Generation.py CLI.
    """
    results: Dict[TargetKey, Optional[str]] = {}
    if not target_results:
        return results

    from Tournament_randomization import tournament_select_word

    sentence_token_cache = token_cache if token_cache is not None else {}

    def process_single_tournament(item: Tuple[TargetKey, List[str]]) -> Tuple[TargetKey, Optional[str]]:
        (sentence, target, start_index, end_index), alternatives = item
        if not alternatives:
            return (sentence, target, start_index, end_index), None

        similarity = [_UNIFORM_SIMILARITY] * len(alternatives)

        if sentence not in sentence_token_cache:
            sentence_token_cache[sentence] = tokenize_words(sentence)

        context_tokens = sentence_token_cache[sentence]
        left_context = context_tokens[max(0, start_index - h) : start_index]

        randomized_word = tournament_select_word(
            target,
            alternatives,
            similarity,
            context=left_context,
            key=secret_key,
            m=m,
            c=c,
            alpha=alpha,
        )
        return (sentence, target, start_index, end_index), randomized_word

    use_threads = os.environ.get("WM_TOURNAMENT_USE_THREADS", "0").strip().lower() in {
        "1", "true", "yes", "y",
    }
    thread_threshold = int(os.environ.get("WM_TOURNAMENT_THREAD_THRESHOLD", "100000"))

    if (not use_threads) or len(target_results) <= thread_threshold:
        for item in target_results.items():
            key, result = process_single_tournament(item)
            results[key] = result
        return results

    max_workers = min(8, len(target_results))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_tournament, item): item
            for item in target_results.items()
        }
        for future in as_completed(futures):
            key, result = future.result()
            results[key] = result

    return results


def is_valid_sentence(
    sentence: str,
    token_cache: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """Skip incomplete fragments, punctuation-only strings, and non-text noise."""
    if not sentence or not isinstance(sentence, str):
        return False

    sentence = sentence.strip()

    if len(sentence) == 0:
        return False
    if len(sentence) < 3:
        return False
    if sentence[0] in string.punctuation:
        return False
    if all(c in string.punctuation or c.isspace() for c in sentence):
        return False
    if not any(c.isalnum() for c in sentence):
        return False

    tokens = _get_sentence_tokens(sentence, token_cache)
    if not tokens:
        return False

    if not any(any(ch.isalnum() for ch in tok) for tok in tokens):
        return False

    has_word = any(any(ch.isalpha() for ch in tok) for tok in tokens)
    if not has_word and len(tokens) < 3:
        return False

    return True


def build_watermark_chat_messages(sentence: str) -> List[Dict[str, str]]:
    """Chat messages for one sentence detect/generate prompt."""
    user_content = _DETECT_GENERATE_USER_PREFIX + sentence + _DETECT_GENERATE_USER_SUFFIX
    return [_DETECT_GENERATE_SYSTEM, {"role": "user", "content": user_content}]


def finalize_watermarked_sentence(
    sentence: str,
    decoded_llm_output: str,
    Top_K: int,
    secret_key: str,
    m: int,
    c: int,
    h: int,
    alpha: float,
    *,
    structured_used: bool = False,
    token_cache: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    CPU post-processing for one watermark LLM output: parse, tournament, apply replacements.
    Returns (watermarked_sentence, sampling_log).
    """
    if not sentence or not is_valid_sentence(sentence, token_cache=token_cache):
        return sentence, []

    word_to_candidates = _parse_detect_and_generate_output(
        decoded_llm_output, strict_json_only=structured_used
    )
    if not word_to_candidates:
        return sentence, []

    cache = token_cache if token_cache is not None else {}
    tokens = _get_sentence_tokens(sentence, cache)
    target_results = _map_candidates_to_targets(sentence, word_to_candidates, tokens, Top_K)
    if not target_results:
        return sentence, []

    tournament_results = parallel_tournament_sampling(
        target_results,
        secret_key,
        m,
        c,
        h,
        alpha,
        token_cache=cache,
    )

    replacements: List[Tuple[int, int, str, str]] = []
    sampling_log: List[Dict[str, Any]] = []
    for (sent, target, start_i, end_i), randomized_word in tournament_results.items():
        if not randomized_word:
            continue
        alternatives_list = target_results.get((sent, target, start_i, end_i), [])
        sampling_log.append(
            _make_sampling_record(target, alternatives_list, randomized_word)
        )
        replacements.append((start_i, end_i, target, randomized_word))

    if not replacements:
        return sentence, sampling_log
    return apply_replacements(sentence, replacements), sampling_log


def whole_context_process_sentences_batch(
    sentences: List[str],
    llm: Any,
    tokenizer: Any,
    Top_K: int,
    secret_key: str,
    m: int,
    c: int,
    h: int,
    alpha: float,
    output_name: str,
    batch_size: int = 8,
    *,
    wm_temperature: float = 0.0,
    wm_top_p: float = 1.0,
    wm_detect_max_tokens: Optional[int] = None,
) -> List[Tuple[List[Tuple[int, int, str, str]], List[Dict[str, Any]]]]:
    """
    End-to-end watermarking for a list of sentences (detect/generate + tournament).
    Returns [(replacements, sampling_log), ...] in input order.
    output_name is reserved for caller-side artifact naming (unused here).
    """
    if not sentences:
        return []

    token_cache: Dict[str, List[str]] = {}

    if VLLM_AVAILABLE and llm is not None:
        batch_results = llm_detect_and_generate_candidates_batch(
            sentences,
            llm,
            tokenizer,
            Top_K=Top_K,
            batch_size=batch_size,
            token_cache=token_cache,
            wm_temperature=wm_temperature,
            wm_top_p=wm_top_p,
            wm_detect_max_tokens=wm_detect_max_tokens,
        )
    else:
        batch_results = {}

    results_per_sentence = []
    for sentence in sentences:
        sentence_results = {
            k: v
            for k, v in batch_results.items()
            if k[0] == sentence and v and len(v) >= 2
        }
        tournament_results = parallel_tournament_sampling(
            sentence_results,
            secret_key,
            m,
            c,
            h,
            alpha,
            token_cache=token_cache,
        )

        replacements: List[Tuple[int, int, str, str]] = []
        sampling_log: List[Dict[str, Any]] = []
        for (sent, target, start_i, end_i), randomized_word in tournament_results.items():
            if not randomized_word:
                continue
            alternatives_list = sentence_results.get((sent, target, start_i, end_i), [])
            sampling_log.append(
                _make_sampling_record(target, alternatives_list, randomized_word)
            )
            replacements.append((start_i, end_i, target, randomized_word))

        results_per_sentence.append((replacements, sampling_log))

    return results_per_sentence
