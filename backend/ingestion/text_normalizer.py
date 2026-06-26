"""
ORBITAL Text Normalizer
Central module for all text pre-processing before obligation extraction.

Design principle: every normalization step must be format-agnostic — it
should handle OCR text, clean digital PDF text, mixed Hindi/English text,
amendment documents, master directions, and circulars equally well.

This module is the single source of truth for:
  - Raw text cleaning (OCR artefacts, ligatures, line-break rejoining)
  - Section text preparation (strip noise, signatory tails, page numbers)
  - Atomic unit splitting (the correct boundary for one obligation)
  - Action phrase extraction (format-agnostic, predicate-preserving)
  - Boilerplate / non-obligation detection
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional


# ── Constants ────────────────────────────────────────────────────────────────

# Minimum meaningful text length
MIN_UNIT_LENGTH = 20

# Action phrase limits
ACTION_MAX_WORDS = 60        # Soft cap — above this the action is a raw clause dump
ACTION_MAX_CHARS = 350       # Hard cap — anything longer is flagged verbatim

# Verbatim threshold — only applied to LONG actions (>= 150 chars)
# Short clean passive sentences (e.g. "shall be subjected to quarterly audit")
# are legitimately high-similarity to source text and should NOT be penalised.
VERBATIM_THRESHOLD = 0.85
VERBATIM_MIN_LENGTH = 150    # Don't apply verbatim check below this length

# Boilerplate phrases that are never obligations regardless of "shall"
BOILERPLATE_PHRASES: tuple[str, ...] = (
    "shall be called",
    "shall be known as",
    "shall be cited as",
    "shall be titled",
    "is hereby issued",
    "come into force",
    "come into effect",
    "shall come into force",
    "shall come into effect",
    "these directions shall",
    "these amendment directions shall",
    "these regulations shall",
    "these guidelines shall",
    "this circular shall",
    "hereby issues the following",
    "in exercise of the powers",
    "being satisfied that it is necessary",
    "expedient in the public interest",
)

# Quotation lead-in phrases — content following these is background law, not new directive
QUOTE_LEADINS: tuple[str, ...] = (
    "provides as under",
    "provides as follows",
    "reads as under",
    "reads as follows",
    "states as under",
    "states as follows",
    "states that",
    "stipulates as under",
    "stipulates as follows",
    "is reproduced below",
    "is extracted below",
    "is quoted below",
    "are reproduced below",
    "are extracted below",
    "is set out below",
    "as under:",
    "as follows:",
    "as hereunder:",
)

# Operational restart phrases inside quoted blocks — these signal a NEW directive
# within or immediately after a quoted section
OPERATIONAL_RESTARTS: tuple[str, ...] = (
    "accordingly",
    "it has now been decided",
    "it is hereby directed",
    "it is advised",
    "all regulated entities shall",
    "banks shall",
    "insurers shall",
    "with effect from",
    "in view of the above",
)

# Modal verbs that identify obligation boundaries
MODAL_PATTERN = re.compile(
    r"\b(shall|must|is required to|are required to|shall not|must not|may)\b",
    re.IGNORECASE,
)

# Actor phrases that duplicate the obligation's actor field — strip these as prefix
ACTOR_PREFIX_PATTERN = re.compile(
    r"^(?:a\s+)?(?:bank|banks|the\s+bank|the\s+banks|insurer|insurers|the\s+insurer|"
    r"regulated\s+entit(?:y|ies)|res|nbfc(?:s)?|the\s+company|companies|"
    r"frb(?:s)?|reinsurer(?:s)?)\s+",
    re.IGNORECASE,
)

# Signatory patterns at end of document sections
SIGNATORY_PATTERN = re.compile(
    r"\s*\(?\b[A-Z][A-Za-z .']{3,}\)?\s*[\n,]?\s*"
    r"(?:Chief\s+General\s+Manager|General\s+Manager|Deputy\s+Governor|"
    r"Governor|Director|Executive\s+Director|Principal\s+Chief\s+General\s+Manager|"
    r"Chief\s+Executive\s+Officer|Secretary|Joint\s+Secretary|Under\s+Secretary)"
    r".*$",
    re.IGNORECASE | re.DOTALL,
)

# Trailing "Yours faithfully" / "Yours sincerely"
YOURS_PATTERN = re.compile(
    r"\s*Yours\s+(?:faithfully|sincerely),?\s*$",
    re.IGNORECASE,
)

# Formula / mathematical expression — not an obligation predicate
FORMULA_PATTERN = re.compile(
    r"[A-Z][a-z]?\s*=\s*[A-Z][a-z]?\s*[-+*/]|"
    r"\bEPt\b|\bNPt\b|\bCRAR\b|\bLCR\b|\bNSFR\b|"
    r"(?:\d+\s*[-+*/]\s*\d+\s*[=<>])|"
    r"(?:\bwhere\s*:\s*\n)",
    re.IGNORECASE,
)

# Page-number noise lines
PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,3}\s*$")

# ── Public API ────────────────────────────────────────────────────────────────

def clean_section_text(text: str) -> str:
    """
    Prepare raw section text for obligation extraction.

    Applies (in order):
      1. Unicode normalization (ligatures, NBSP, smart quotes → plain)
      2. Line-ending normalization
      3. OCR line-break artefact repair (word.\nword → word word)
      4. Strip signatory tails
      5. Strip page-number noise lines
      6. Collapse excessive whitespace
    """
    if not text:
        return ""

    # 1. Unicode normalization
    t = text
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\u00a0", " ")          # NBSP → space
    t = t.replace("\ufb01", "fi").replace("\ufb02", "fl")   # ligatures
    t = t.replace("\u2019", "'").replace("\u2018", "'")     # smart single quotes
    # Preserve curly double quotes — they delimit quoted blocks in Indian regs
    # but normalize to plain ASCII for downstream regex compatibility
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u201e", '"').replace("\u201f", '"')

    # 2. OCR line-break artefact repair
    # Pattern A: short prep/article before period at line end (OCR hard wrap)
    #   "services at the.\nearli\nest" → "services at the earliest"
    t = re.sub(
        r"\b(at|by|of|the|a|an|from|to|for|in|on|with|under|into|per|and|or|its|their|any)\.\s*\n\s*([a-z])",
        r"\1 \2",
        t,
        flags=re.IGNORECASE,
    )
    # Pattern B: any period + newline + lowercase continuation (general OCR wrap)
    t = re.sub(r"\.\s*\n\s*([a-z])", r" \1", t)
    # Pattern C: hyphenated line break — "regula-\ntory" → "regulatory"
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
    # Pattern D: space before newline swallowed — "the \nbank" → "the bank"
    t = re.sub(r"(\w) \s*\n\s*(\w)", r"\1 \2", t)

    # 3. Strip signatory tail
    t = SIGNATORY_PATTERN.sub("", t)
    t = YOURS_PATTERN.sub("", t)

    # 4. Strip page-number noise lines
    lines = [line for line in t.split("\n") if not PAGE_NUMBER_PATTERN.match(line)]
    t = "\n".join(lines)

    # 5. Collapse excess whitespace (preserve paragraph breaks as \n\n)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


def split_into_obligation_units(text: str, clause_type: str = "other") -> list[str]:
    """
    Split a cleaned section text into the smallest meaningful obligation units.

    Strategy (in order, first applicable wins):
      A. Amendment replacement block  — extract operative text from inside "shall be replaced by: ..."
      B. Enumerated list              — split on (a)/(i)/1. markers, prepend parent context
      C. Modal verb boundaries        — split at ". Banks shall" / ". The insurer shall" etc.
      D. Paragraph boundaries         — split on double newline
      E. Fallback                     — return whole text as single unit

    Each returned unit is guaranteed to:
      - Be >= MIN_UNIT_LENGTH characters
      - Contain at least one modal verb (for obligation units)
      - Not be a boilerplate or page-number line
    """
    cleaned = clean_section_text(text)
    if not cleaned:
        return []

    # A. Amendment replacement block
    replacement_units = _extract_replacement_block_units(cleaned)
    if replacement_units:
        return replacement_units

    units: list[str] = []

    # B + C + D: enumerate, then split on modal/paragraph boundaries
    for block in re.split(r"\n{2,}", cleaned):
        block = block.strip()
        if not block:
            continue
        for sub in _split_enumerated_block(block):
            for part in _split_on_modal_boundaries(sub):
                part = re.sub(r"\s+", " ", part).strip(" ;")
                if len(part) >= MIN_UNIT_LENGTH:
                    units.append(part)

    return units if units else ([cleaned] if len(cleaned) >= MIN_UNIT_LENGTH else [])


def extract_action_phrase(unit: str, trigger_rule: Optional[dict] = None) -> str:
    """
    Extract a clean, self-contained action phrase from an obligation unit.

    Design principle: PRESERVE the full predicate. Indian regulatory text is
    written as complete sentences — the action IS the sentence minus boilerplate
    actor prefix and modal-only prefix. We do NOT chop at the trigger verb.

    Steps:
      1. Strip leading clause number
      2. Strip leading actor phrase (broad — handles "A bank", "Banks", "it", "Persons...")
      3. Strip leading modal-only prefix ONLY when a real verb follows
      4. Rewrite "not / that / to / also" openings
      5. Ensure starts with imperative verb (capitalised)
      6. Ensure ends with period
    """
    if not unit:
        return "Review the clause manually."

    clean = re.sub(r"\s+", " ", unit).strip()

    # 1. Strip leading clause number
    clean = re.sub(r"^[A-Za-z]?\d+(?:\.\d+)*[A-Z]?\.\s*", "", clean)
    clean = re.sub(r"^\(?[a-zivxlcdm]+\)\s*", "", clean, flags=re.IGNORECASE)

    # 2. Strip leading actor phrase — broad list covering Indian regulatory patterns
    clean = ACTOR_PREFIX_PATTERN.sub("", clean)
    # Also strip pronoun-based subjects ("it", "they", "he", "she") that refer
    # to the actor already identified in the actor field
    clean = re.sub(
        r"^(?:it|they|he|she|such\s+(?:bank|entity|insurer|person))\s+",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    # Strip "at its discretion, " and "at their discretion, " — pure boilerplate qualifier
    clean = re.sub(r"^at\s+(?:its|their|his|her)\s+discretion\s*,?\s*", "", clean, flags=re.IGNORECASE)

    # 3. Strip leading modal-only prefix when a real predicate follows
    modal_prefix = re.match(
        r"^(shall\s+also|must\s+also|shall|must|may|is\s+required\s+to|"
        r"are\s+required\s+to)\s+(?=\S)",
        clean,
        re.IGNORECASE,
    )
    if modal_prefix:
        remainder = clean[modal_prefix.end():]
        # Keep: "shall not" (prohibition), "shall be" (passive), compound predicates
        # Strip only when a clear action verb follows directly
        if (
            len(remainder) >= 8
            and not re.match(
                r"^(?:not\s+)?(?:be\s+)?(?:the\s+|a\s+|an\s+|this\s+|that\s+|such\s+|called\s+|known\s+)",
                remainder,
                re.IGNORECASE,
            )
        ):
            clean = remainder.strip()

    # 4. Rewrite leading fragments
    clean = re.sub(r"^not\s+", "Refrain from ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^that\s+", "Ensure ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^to\s+(?=\S)", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^also\s+", "", clean, flags=re.IGNORECASE)
    # "For continuing X, ..." → keep as-is (it's already a full predicate)

    clean = re.sub(r"\s+", " ", clean).strip(" .;:")

    if not clean or len(clean) < 5:
        return "Review the clause manually."

    result = clean[0].upper() + clean[1:]
    if not result.endswith("."):
        result += "."

    return result


def is_boilerplate(text: str) -> bool:
    """
    Return True if the text is a boilerplate/administrative clause that should
    never produce an obligation, regardless of whether it contains "shall".

    Covers:
      - Naming clauses ("shall be called")
      - Effective-date clauses ("shall come into force")
      - Legislative authority paragraphs ("in exercise of the powers")
      - Pure signatory lines
    """
    lower = text.lower().strip()
    if not lower:
        return True
    return any(phrase in lower for phrase in BOILERPLATE_PHRASES)


def is_verbatim_action(action: str, source_text: str) -> bool:
    """
    Return True when the action is just a copy of the source text.
    Only applies the check for LONG actions (>= VERBATIM_MIN_LENGTH chars)
    to avoid falsely flagging clean short passive sentences.
    """
    if len(action) < VERBATIM_MIN_LENGTH:
        return False
    ratio = SequenceMatcher(None, action.lower(), source_text[:len(action)].lower()).ratio()
    return ratio > VERBATIM_THRESHOLD


def is_quoted_content(text: str, clause_type: str = "other") -> bool:
    """
    Return True when the text is a quotation of existing law, not a new directive.
    Checks clause_type first, then text content.
    """
    if clause_type == "quoted_reference":
        return True

    lower = text.lower()

    # Operational restart overrides the quote-leadin check
    if any(restart in lower for restart in OPERATIONAL_RESTARTS):
        return False

    # Text starts with a quote-leadin phrase
    if any(lead in lower for lead in QUOTE_LEADINS):
        return True

    # Text is wrapped in double quotes
    body = re.sub(r'^(?:\([a-z]+\)\s*|\d+(?:\.\d+)*[A-Z]?\.\s*)', "", text.strip())
    if body and body[0] == '"' and body.rstrip()[-1:] == '"':
        return True

    # Reference-only (as per / vide / pursuant to) without a new directive
    has_ref = re.search(r"\b(?:as per|in terms of|under|vide|pursuant to)\s+(?:section|regulation|para|circular)", lower)
    has_directive = re.search(r"\b(?:shall|must|required to|is advised to|are advised to)\b", lower)
    return bool(has_ref and not has_directive)


def count_action_words(action: str) -> int:
    """Count words in an action phrase."""
    return len(action.split())


def action_quality_score(action: str, source_text: str) -> tuple[float, str]:
    """
    Return a (quality_score, reason) tuple for an extracted action.

    quality_score ranges 0.0–1.0:
      1.0 = perfect
      0.7 = acceptable
      0.5 = needs review (may still pass confidence gate)
      0.0 = must re-summarize

    Key principle: short-to-medium sentences (< VERBATIM_MIN_LENGTH chars) are
    NEVER penalized for verbatim ratio — the full sentence IS the obligation.
    Only genuinely oversized blobs (>= VERBATIM_MIN_LENGTH) are checked.
    """
    if not action or action.strip().lower() in ("review the clause manually.", "review the clause manually"):
        return 0.0, "empty_or_fallback"

    word_count = count_action_words(action)
    char_len = len(action)

    # Too short — vague
    if char_len < MIN_UNIT_LENGTH:
        return 0.2, "too_short"

    # Contains enumeration markers — likely merged multiple obligations
    if re.search(r"(?:^|\s)(?:\([a-z]\)|\([ivxlcdm]+\)|[a-z]\)|[ivxlcdm]+\))\s+", action, re.IGNORECASE):
        return 0.3, "contains_enumeration_markers"

    # Formula content — mathematical expression, not an obligation
    if FORMULA_PATTERN.search(action):
        return 0.45, "contains_formula"

    # Verbatim check — ONLY for genuinely long blobs
    if char_len >= VERBATIM_MIN_LENGTH and is_verbatim_action(action, source_text):
        return 0.5, "verbatim_long"

    # Oversized but not verbatim — still acceptable, just flag it
    if word_count > ACTION_MAX_WORDS:
        return 0.6, "too_long"

    # Good: 4–60 words, not verbatim
    if 4 <= word_count <= 60:
        return 1.0, "good"

    return 0.7, "acceptable"


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_replacement_block_units(text: str) -> list[str]:
    """
    For amendment sections: "Paragraph X shall be replaced by: '...content...'"
    Extract the operative obligation clauses from inside the replacement content.

    Returns list of atomic units, or [] if the pattern is not found.
    """
    frame = re.search(
        r'shall be replaced by\s*[:\-]\s*\n?\s*"?',
        text,
        re.IGNORECASE,
    )
    if not frame:
        return []

    inner = text[frame.end():]
    # Strip outer quote wrapper if present
    inner = inner.lstrip('"').rstrip('"\n').strip()
    # Strip signatory tail that may appear inside
    inner = SIGNATORY_PATTERN.sub("", inner).strip()
    inner = YOURS_PATTERN.sub("", inner).strip()

    if not inner or len(inner) < MIN_UNIT_LENGTH:
        return []

    # Normalize OCR artefacts within the inner block
    inner = re.sub(r"\.\s*\n\s*([a-z])", r" \1", inner)
    inner = re.sub(r"[ \t]+", " ", inner)

    units: list[str] = []
    for block in re.split(r"\n{2,}", inner):
        block = block.strip()
        if not block or PAGE_NUMBER_PATTERN.match(block):
            continue
        for sub in _split_enumerated_block(block):
            for part in _split_on_modal_boundaries(sub):
                part = re.sub(r"\s+", " ", part).strip(" ;")
                # Only keep units with a modal verb — skip math/formula lines
                if (
                    len(part) >= MIN_UNIT_LENGTH
                    and MODAL_PATTERN.search(part)
                    and not FORMULA_PATTERN.search(part)
                ):
                    units.append(part)

    return units


def _split_enumerated_block(block: str) -> list[str]:
    """
    Split a text block on enumeration markers: (a), (i), 1., A., -, *.

    When a parent clause introduces a list (ends with ":"), we prepend the
    parent context to each child so each unit is self-contained.
    """
    if not block:
        return []

    # Strip leading clause number from the block itself
    block = re.sub(r"^\s*[A-Za-z]?\d+(?:\.\d+)*[A-Z]?\.\s+", "", block).strip()

    # Marker patterns (newline-anchored and inline)
    marker_re = re.compile(
        r"(?:(?:^|\n)\s*(?:\(?[a-z]\)|\(?[ivxlcdm]+\)|\d+\.|[A-Z]\.|[*\-•])\s+)",
        re.IGNORECASE,
    )
    inline_marker_re = re.compile(
        r"\s+(?=(?:\([a-z]\)|\([ivxlcdm]+\)|[a-z]\)|[ivxlcdm]+\))\s+)",
        re.IGNORECASE,
    )

    pieces = marker_re.split(block)
    if len(pieces) == 1:
        pieces = inline_marker_re.split(block)
    if len(pieces) == 1:
        return [block]

    parent_text = pieces[0].strip(" :;\n")
    children = [p.strip(" ;\n") for p in pieces[1:] if len(p.strip()) >= 8]

    if not children:
        return [block]

    # Keep the parent if it contains its own modal verb and is long enough
    parent_units = _keep_parent_if_operative(parent_text)

    # Build child context prefix from parent if it ends with ":"
    context = _parent_context_for_children(parent_text)
    if context:
        child_units = [
            re.sub(r"\s+", " ", f"{context} {_strip_enumeration_token(c)}").strip()
            for c in children
        ]
    else:
        child_units = [_strip_enumeration_token(c) for c in children]

    return parent_units + child_units


def _keep_parent_if_operative(parent: str) -> list[str]:
    """Return the parent clause as a unit only if it contains a real obligation."""
    if not parent:
        return []
    modal = re.search(
        r"\b(shall|must|may|is required to|are required to|shall not|must not)\b",
        parent,
        re.IGNORECASE,
    )
    if not modal:
        return []
    # Check the tail after the modal has substance (not just "shall:" or "shall ensure that:")
    tail = parent[modal.end():].strip(" :;.")
    if len(tail) < 8:
        return []
    # Reject pure lead-ins like "Banks shall ensure the following:"
    if re.search(r"(?:the\s+following|as\s+under|as\s+follows)\s*:?\s*$", parent, re.IGNORECASE):
        return []
    return [re.sub(r"\s+", " ", parent).strip()]


def _parent_context_for_children(parent: str) -> str:
    """Extract an actor+modal context string to prepend to child clauses."""
    if not parent:
        return ""
    cleaned = re.sub(r"^[A-Za-z]?\d+(?:\.\d+)*[A-Z]?\.\s*", "", parent)
    match = re.search(
        r"([^.]{0,220}?\b(?:shall|must|is required to|are required to|may|shall not|must not)\b[^.;:]*(?:that)?)\s*:?\s*$",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip(" :;")
    return ""


def _strip_enumeration_token(text: str) -> str:
    """Remove leading enumeration token from a child clause."""
    text = re.sub(
        r"^\s*(?:\([a-z]+\)|\([ivxlcdm]+\)|[a-z]\)|[ivxlcdm]+\)|\d+\.)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip(" ;.")


def _split_on_modal_boundaries(text: str) -> list[str]:
    """
    Split a text unit at sentence boundaries before a new actor+modal phrase.

    Pattern: ". Banks shall" / ". During the period, it shall" etc.
    Only splits at real sentence ends (after ". " or "; "), not mid-sentence.
    """
    parts = re.split(
        r"(?<=[.;])\s+(?="
        r"(?:All\s|The\s|A\s|An\s|Banks?\s|Insurers?\s|Regulated\s+entities|REs\s|"
        r"During\s+the\s+period|Where\s|If\s|Provided\s+that\s|In\s+case\s+of\s|"
        r"In\s+such\s+cases?\s*,|For\s+continuing\s|Subject\s+to\s|"
        r"Such\s+(?:bank|entity|insurer|person)s?\s)"
        r".{0,120}\b(?:shall|must|may|is\s+required\s+to|are\s+required\s+to)\b"
        r")",
        text,
        flags=re.IGNORECASE,
    )
    return parts if len(parts) > 1 else [text]
