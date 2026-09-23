"""
Tournament sampling for watermark synonym selection.

Given alternatives and similarity scores, softmax-weighted draws seed the first
bracket (M = c^m picks). Each round groups competitors (size c), ranks them with
per-round HMAC Bernoulli scores, and advances winners until one index remains.
Ties are broken with a deterministic seed derived from key, context, round, and group.
"""

import hashlib
import hmac
import math
import random
from typing import List, Tuple


def _softmax(xs: List[float], alpha: float = 1.0) -> List[float]:
    # Temperature-scaled softmax; subtract max for numerical stability
    if not xs:
        return []
    m = max(xs)
    zs = [math.exp(alpha * (x - m)) for x in xs]
    s = sum(zs)
    return [z / s for z in zs]


def _hash_bytes(key: str, payload: str) -> bytes:
    return hmac.new(key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()


def _hash_to_uniform01(key: str, payload: str) -> float:
    # Map 8 bytes to [0, 1) as a 64-bit integer / 2**64
    b = _hash_bytes(key, payload)[:8]
    n = int.from_bytes(b, "big", signed=False)
    return n / 2**64


def _ctx_str(context: List[str]) -> str:
    # Use last token as left context (can widen to last-4 if needed)
    toks = context or []
    return " ".join(toks[-1:]).lower()


def _per_round_score(key: str, ctx: str, candidate: str, round_idx: int) -> int:
    # Deterministic Bernoulli(0.5) score from HMAC(key, context, candidate, round)
    payload = f"score::{ctx}::{candidate}::r{round_idx}"
    uniform_score = _hash_to_uniform01(key, payload)
    return 1 if uniform_score >= 0.5 else 0


def _sample_categorical_from_uniform(u: float, probs: List[float]) -> int:
    # Inverse-CDF sampling from probs using uniform u in [0, 1)
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if u < c:
            return i
    return len(probs) - 1  # fallback when float sums miss 1.0


def _draw_M_candidates(key: str, ctx: str, probs: List[float], c: int, m: int) -> List[int]:
    """
    Draw M = c^m candidate indices (0..len(probs)-1) with replacement from probs,
    using a deterministic uniform stream keyed by (key, ctx).
    """
    picks = []
    for draw_idx in range(c ** m):
        u = _hash_to_uniform01(key, f"draw::{ctx}::m{m}::c{c}::i{draw_idx}")
        picks.append(_sample_categorical_from_uniform(u, probs))
    return picks


def _run_tournament_round(
    key: str,
    ctx: str,
    picks: List[int],
    candidates: List[str],
    round_idx: int,
    group_size: int,
) -> List[int]:
    """
    Split picks into consecutive groups of size group_size; per group, advance the
    candidate with highest per-round score. Ties use a deterministic random choice.
    """
    assert len(picks) % group_size == 0, "Group partition must divide evenly."
    winners = []

    for g in range(0, len(picks), group_size):
        group = picks[g:g + group_size]
        group_num = g // group_size + 1

        scored: List[Tuple[int, int]] = []
        for idx in group:
            s = _per_round_score(key, ctx, candidates[idx], round_idx)
            scored.append((s, idx))

        max_s = max(s for s, _ in scored)
        tied = [idx for (s, idx) in scored if s == max_s]

        if len(tied) == 1:
            winners.append(tied[0])
        else:
            # Deterministic tie-break (does not touch global RNG state)
            seed_string = f"{key}:{ctx}:r{round_idx}:g{group_num}"
            seed_value = sum(ord(ch) * (i + 1) for i, ch in enumerate(seed_string))
            winners.append(random.Random(seed_value).choice(tied))

    return winners


def tournament_randomize(
    original: str,
    alternatives: List[str],
    similarity: List[float],
    context: List[str],
    key: str,
    m: int = 2,
    c: int = 2,
    alpha: float = 1.0,
) -> int:
    """
    Tournament sampling among alternatives (K = len(alternatives)).
    Returns winner index (0..K-1) into alternatives.

    original is kept for API compatibility with callers (not used in selection).
    """
    assert len(alternatives) == len(similarity) > 0, "Need at least one alternative with a similarity score."
    assert m >= 1, "Tournament rounds m must be >= 1"
    assert c >= 2, "Number of competitors per match c must be >= 2"

    ctx = _ctx_str(context)
    probs = _softmax(similarity, alpha=alpha)

    # Round 1: M = c^m draws from softmax probs, then m elimination rounds (group size c)
    current = _draw_M_candidates(key, ctx, probs, c, m)
    for r in range(1, m + 1):
        current = _run_tournament_round(
            key=key,
            ctx=ctx,
            picks=current,
            candidates=alternatives,
            round_idx=r,
            group_size=c,
        )

    assert len(current) == 1, f"Expected a single winner after {m} rounds, got {len(current)}"
    return current[0]


def tournament_select_word(
    original: str,
    alternatives: List[str],
    similarity: List[float],
    context: List[str],
    key: str,
    m: int = 2,
    c: int = 2,
    alpha: float = 1.0,
) -> str:
    """Return the selected alternative word (convenience wrapper around tournament_randomize)."""
    idx = tournament_randomize(original, alternatives, similarity, context, key, m, c, alpha)
    return alternatives[idx]
