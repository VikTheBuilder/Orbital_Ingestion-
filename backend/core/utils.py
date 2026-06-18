"""
ORBITAL Core Utilities
Shared helpers used across the pipeline — consolidated to eliminate duplication.
"""

import re
from typing import Optional

# ── Month lookup table ────────────────────────────────────────────────────────

_MONTH_MAP: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# ── Date Parsing ──────────────────────────────────────────────────────────────

def parse_date_to_iso(value: str) -> Optional[str]:
    """
    Parse a date string to ISO 8601 (YYYY-MM-DD).
    Handles:
      - DD.MM.YYYY / DD/MM/YYYY / DD-MM-YYYY
      - Month DD, YYYY  (e.g. April 29, 2026)
      - DD Month YYYY   (e.g. 29 April 2026)
    Returns None on failure — never raises.
    """
    if not value:
        return None

    value = value.strip()
    # Normalise separators
    normalised = value.replace("/", ".").replace("-", ".")

    # DD.MM.YYYY
    numeric = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", normalised)
    if numeric:
        d, mo, y = numeric.groups()
        try:
            return f"{y}-{int(mo):02d}-{int(d):02d}"
        except ValueError:
            pass

    # Month DD, YYYY
    mf = re.match(
        r"^(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2}),?\s+(\d{4})$",
        value, re.IGNORECASE,
    )
    if mf:
        mn, d, y = mf.groups()
        mo = _MONTH_MAP.get(mn.lower())
        if mo:
            return f"{y}-{mo:02d}-{int(d):02d}"

    # DD Month YYYY
    df = re.match(
        r"^(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{4})$",
        value, re.IGNORECASE,
    )
    if df:
        d, mn, y = df.groups()
        mo = _MONTH_MAP.get(mn.lower())
        if mo:
            return f"{y}-{mo:02d}-{int(d):02d}"

    return None


def parse_financial_year(value: str) -> Optional[str]:
    """Parse an Indian financial year reference to an ISO start date (April 1).

    Handles:
      - "2022-23", "2022-2023"
      - "FY 2022-23", "FY2022-23"
      - "financial year 2022-23"
      - "financial year 2022-23 onwards"

    Indian financial years run April 1 – March 31, so FY 2022-23 → 2022-04-01.
    Returns None on failure — never raises.
    """
    if not value:
        return None

    m = re.search(
        r"(?:financial\s+year|FY)\s*(\d{4})\s*[-–]\s*(\d{2,4})",
        value, re.IGNORECASE,
    )
    if m:
        start_year = int(m.group(1))
        return f"{start_year}-04-01"

    # Bare "2022-23" pattern (four digits, dash, two digits)
    m2 = re.match(r"^\s*(\d{4})\s*[-–]\s*(\d{2})\s*$", value.strip())
    if m2:
        start_year = int(m2.group(1))
        return f"{start_year}-04-01"

    return None


def extract_first_iso_date(text: str) -> Optional[str]:
    """Return the first parseable date found in text, as ISO 8601."""
    patterns = [
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result = parse_date_to_iso(m.group(0))
            if result:
                return result
    return None


# ── Deduplication helper ───────────────────────────────────────────────────────

def unique_ordered(items: list[str]) -> list[str]:
    """Return a deduplicated list preserving original order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
