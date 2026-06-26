"""Lightweight retrieval over approved regulatory patterns.

This is deliberately dependency-free.  It gives small models and deterministic
rankers nearest historical patterns without requiring a vector database.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from backend.core.pattern_library import iter_active_patterns


TOKEN_RE = re.compile(r"[a-z0-9]+")


def retrieve_regulatory_patterns(
    text: str,
    regulator: str | None = None,
    pattern_type: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return similar approved/candidate patterns for a clause or field."""
    query_tokens = _token_counts(text)
    scored: list[tuple[float, dict[str, Any]]] = []

    for pattern in iter_active_patterns(pattern_type=pattern_type, regulator=regulator):
        pattern_text = " ".join([
            str(pattern.get("pattern_text") or ""),
            str(pattern.get("normalized_value") or ""),
            " ".join(example.get("evidence_span", "") for example in pattern.get("examples", []) if isinstance(example, dict)),
        ])
        if not pattern_text.strip():
            continue
        score = (
            0.55 * _cosine(query_tokens, _token_counts(pattern_text))
            + 0.30 * SequenceMatcher(None, text.lower(), pattern_text.lower()).ratio()
            + 0.15 * float(pattern.get("confidence", 0.0))
        )
        if score > 0:
            enriched = dict(pattern)
            enriched["similarity_score"] = round(score, 4)
            scored.append((score, enriched))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def _token_counts(text: str) -> Counter:
    return Counter(TOKEN_RE.findall((text or "").lower()))


def _cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
