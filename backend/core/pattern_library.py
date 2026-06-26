"""Self-improving regulatory pattern library.

The library is intentionally JSON-backed so reviewer-approved phrases can be
added without code changes.  Runtime code only trusts patterns whose status is
approved or candidate.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.core.logger import get_logger

logger = get_logger(__name__)

_RULES_ROOT = Path(__file__).resolve().parent.parent.parent / "rules"
_PATTERN_LIBRARY_PATH = _RULES_ROOT / "pattern_library.json"


@lru_cache(maxsize=1)
def load_pattern_library() -> dict[str, Any]:
    try:
        return json.loads(_PATTERN_LIBRARY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Pattern library load failed", path=str(_PATTERN_LIBRARY_PATH), error=str(exc))
        return {"version": "0.0.0", "patterns": []}


def iter_active_patterns(pattern_type: str | None = None, regulator: str | None = None) -> list[dict[str, Any]]:
    library = load_pattern_library()
    patterns = []
    for pattern in library.get("patterns", []):
        if pattern.get("status") not in {"approved", "candidate"}:
            continue
        if pattern_type and pattern.get("pattern_type") != pattern_type:
            continue
        if regulator and pattern.get("regulator") not in {regulator, "ANY"}:
            continue
        patterns.append(pattern)
    return patterns
