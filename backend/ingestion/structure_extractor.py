"""
ORBITAL Structure Extractor
Extracts regulatory-document metadata and clause-level sections from OCR text.
Metadata extraction (circular numbers, dates, cross-references, amends) is
delegated to the JSON Rule Engine where rules exist.
"""

import hashlib
import re
from pathlib import Path
from typing import Optional

from backend.core.logger import get_logger
from backend.core.rule_engine import get_rule_engine
from backend.core.utils import parse_date_to_iso, parse_financial_year, extract_first_iso_date
from backend.ingestion.schemas import DocumentStructureSchema, SectionSchema

logger = get_logger(__name__)

SOURCE_VALUES = {"RBI", "SEBI", "CERT-In", "NPCI", "IRDAI", "DPDP", "FIU-IND", "IBA", "OTHER"}


def extract_structure(
    full_text: str,
    pages: list,
    source: str,
    pdf_path: str,
    analysis_hints: Optional[dict] = None,
) -> DocumentStructureSchema:
    """Parse OCR text into the project document schema."""
    try:
        logger.info("Structure extraction started", source=source, pdf_path=pdf_path)
        re_engine = get_rule_engine()

        normalized_text = _normalize_text(full_text)
        detected_source = _extract_source(normalized_text, source, analysis_hints)

        # ── Rule Engine: metadata extraction ──
        circular_number = (
            re_engine.extract_circular_number(normalized_text)
            or _extract_circular_number(normalized_text)
        )
        reference_number = (
            re_engine.extract_reference_number(normalized_text)
            or _extract_reference_number(normalized_text, circular_number)
        )
        issue_date = (
            re_engine.extract_header_date("\n".join(normalized_text.split("\n")[:30]))
            or _extract_header_date(normalized_text)
        )
        effective_date = (
            re_engine.extract_effective_date(normalized_text)
            or _extract_effective_date(normalized_text)
        )
        amends = (
            re_engine.detect_amends(normalized_text)
            or _extract_amends(normalized_text)
        )

        title = _extract_title(normalized_text, pdf_path)
        issued_by = _extract_issued_by(normalized_text)
        language = _detect_language(normalized_text)

        # ── Rule Engine: cross-references ──
        cross_references = re_engine.extract_cross_references(normalized_text)
        if not cross_references:
            cross_references = _extract_cross_references(normalized_text)

        sections = _extract_sections(normalized_text, pages, re_engine)
        tables = _extract_tables(normalized_text)
        annexures = _extract_annexures(sections)
        doc_id = _generate_doc_id(detected_source, circular_number, title)

        logger.info(
            "Structure extraction complete",
            doc_id=doc_id,
            sections=len(sections),
            tables=len(tables),
            annexures=len(annexures),
            cross_references=len(cross_references),
        )

        return DocumentStructureSchema(
            doc_id=doc_id,
            source=detected_source,
            title=title,
            circular_number=circular_number,
            reference_number=reference_number,
            date=issue_date,
            effective_date=effective_date,
            issued_by=issued_by,
            amends=amends,
            language=language,
            total_pages=len(pages) if pages else 0,
            sections=sections,
            tables=tables,
            annexures=annexures,
            cross_references=cross_references,
            obligations=[],
            analysis=analysis_hints or None,
        )

    except Exception as e:
        logger.error("Structure extraction failed", error=str(e), pdf_path=pdf_path)
        fallback_title = Path(pdf_path).stem if pdf_path else "unknown_document"
        fallback_source = source if source in SOURCE_VALUES else "OTHER"
        return DocumentStructureSchema(
            doc_id=f"{_safe_source_prefix(fallback_source)}-{hashlib.md5(fallback_title.encode()).hexdigest()[:12]}",
            source=fallback_source,
            title=fallback_title,
            issued_by="",
            analysis=analysis_hints or None,
        )


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_source(text: str, fallback: str, analysis_hints: Optional[dict] = None) -> str:
    lower = text.lower()
    if "reserve bank of india" in lower or "rbi/" in lower:
        return "RBI"
    if "securities and exchange board of india" in lower or "sebi/" in lower:
        return "SEBI"
    if "cert-in" in lower or "cert in" in lower:
        return "CERT-In"
    if "national payments corporation of india" in lower or "npci" in lower:
        return "NPCI"
    if (
        "insurance regulatory and development authority of india" in lower
        or "insurance regulatory and development authority" in lower
        or "irdai" in lower
        or "irda" in lower
    ):
        return "IRDAI"
    if "digital personal data protection" in lower or "dpdp" in lower:
        return "DPDP"
    if "financial intelligence unit - india" in lower or "financial intelligence unit" in lower or "fiu-ind" in lower or "fiu" in lower:
        return "FIU-IND"
    if "indian banks' association" in lower or "indian banks association" in lower or "\niba" in lower:
        return "IBA"
    if analysis_hints:
        inferred = _infer_source_from_analysis(analysis_hints)
        if inferred != "OTHER":
            return inferred
    return fallback if fallback in SOURCE_VALUES else "OTHER"


def _infer_source_from_analysis(analysis_hints: dict) -> str:
    primary_actor = str(analysis_hints.get("primary_actor") or "").lower()
    reasoning = str(analysis_hints.get("reasoning") or "").lower()
    combined = f"{primary_actor} {reasoning}"
    if "reserve bank" in combined or "rbi" in combined:
        return "RBI"
    if "sebi" in combined or "securities and exchange board" in combined:
        return "SEBI"
    if "cert-in" in combined or "cyber" in combined and "government" in combined:
        return "CERT-In"
    if "national payments corporation" in combined or "npci" in combined:
        return "NPCI"
    if "insurance regulatory" in combined or "irdai" in combined or "irda" in combined or "insurer" in combined:
        return "IRDAI"
    if "digital personal data" in combined or "dpdp" in combined:
        return "DPDP"
    if "financial intelligence unit" in combined or "fiu-ind" in combined:
        return "FIU-IND"
    if "indian banks" in combined or "iba" in combined:
        return "IBA"
    return "OTHER"


def _extract_circular_number(text: str) -> Optional[str]:
    patterns = [
        r"\bRBI/\d{4}-\d{2}/\d+\b",
        r"\bSEBI/[A-Z0-9/.-]+\b",
        r"\bCERT-In/[A-Z0-9/.-]+\b",
        r"\bNPCI/[A-Z0-9/.-]+\b",
        r"\bIRDAI/[A-Z0-9/.-]+\b",
        r"\bDPDP/[A-Z0-9/.-]+\b",
        r"\bFIU-IND/[A-Z0-9/.-]+\b",
        r"\bIBA/[A-Z0-9/.-]+\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _extract_reference_number(text: str, circular_number: Optional[str]) -> Optional[str]:
    for line in text.split("\n")[:25]:
        stripped = line.strip()
        if not stripped or stripped == circular_number:
            continue
        if re.search(r"\d{4}-\d{2}$", stripped):
            continue
        if re.match(r"^[A-Z][A-Z0-9()./-]{6,}$", stripped):
            return stripped
    match = re.search(r"\b[A-Z]{2,}(?:\.[A-Z0-9-]+)+/[0-9.]+/\d{4}-\d{2}\b", text)
    return match.group(0) if match else None


def _extract_header_date(text: str) -> Optional[str]:
    header = "\n".join(text.split("\n")[:30])
    return extract_first_iso_date(header)


def _extract_effective_date(text: str) -> Optional[str]:
    patterns = [
        r"(?:come into force|come into effect|shall come into force|shall come into effect|with effect from|effective from)\s+(?:on\s+|from\s+)?([A-Za-z]+ \d{1,2},? \d{4})",
        r"(?:come into force|come into effect|shall come into force|shall come into effect|with effect from|effective from)\s+(?:on\s+|from\s+)?(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_date_to_iso(match.group(1))

    # Financial year pattern: "effective from financial year 2022-23"
    fy_match = re.search(
        r"(?:effective\s+from|with\s+effect\s+from|applicable\s+from|come into (?:force|effect)\s+from)"
        r"\s+(?:the\s+)?(?:financial\s+year|FY)\s*(\d{4}\s*[-–]\s*\d{2,4})",
        text, re.IGNORECASE,
    )
    if fy_match:
        return parse_financial_year(fy_match.group(1))

    if re.search(r"with immediate effect", text, re.IGNORECASE):
        return _extract_header_date(text)
    return None


def _extract_title(text: str, pdf_path: str) -> str:
    lines = [line.strip(" -:\t") for line in text.split("\n") if line.strip()]
    candidates = []

    for idx, line in enumerate(lines[:40]):
        lower = line.lower()
        if line.startswith(("RBI/", "SEBI/", "CERT-In/", "NPCI/", "IRDAI/", "DPDP/", "FIU-IND/", "IBA/")):
            continue
        if lower.startswith(("date", "dated", "tel", "fax", "www.", "madam", "dear sir", "dear madam")):
            continue
        if re.search(
            r"(master direction|master circular|direction|directions|amendment directions|guidelines|framework|policy|scheme|circular|regulations|instruction|implementation)",
            lower,
        ):
            combined = line
            if idx + 1 < len(lines) and len(lines[idx + 1]) < 120 and not re.match(r"^\d", lines[idx + 1]):
                next_lower = lines[idx + 1].lower()
                if not next_lower.startswith(("please refer", "dated", "madam", "dear")):
                    combined = f"{line} {lines[idx + 1]}".strip()
            candidates.append(combined)

    if candidates:
        return max(candidates, key=len)

    for line in lines[:20]:
        if 20 <= len(line) <= 220 and any(ch.isalpha() for ch in line):
            return line

    return Path(pdf_path).stem if pdf_path else "Unknown Document"


def _extract_issued_by(text: str) -> str:
    tail_lines = [line.strip() for line in text.split("\n")[-25:] if line.strip()]
    for idx, line in enumerate(tail_lines):
        if re.match(r"^\([^)]+\)$", line) and idx + 1 < len(tail_lines):
            designation = tail_lines[idx + 1]
            if re.search(r"(manager|director|officer|secretary|general manager|chief)", designation, re.IGNORECASE):
                return f"{line.strip('()')} - {designation}"
    for idx, line in enumerate(tail_lines):
        if re.match(r"^[A-Z][A-Za-z .'-]{3,}$", line) and idx + 1 < len(tail_lines):
            designation = tail_lines[idx + 1]
            if re.search(r"(manager|director|officer|secretary|general manager|chief)", designation, re.IGNORECASE):
                return f"{line} - {designation}"
    return ""


def _extract_amends(text: str) -> Optional[str]:
    patterns = [
        r"Please refer to\s+(.{20,250}?(?:Directions|Circular|Master Direction|Guidelines).{0,100})\.",
        r"amend(?:s|ment to)?\s+(.{20,250}?(?:Directions|Circular|Master Direction|Guidelines).{0,100})\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return None


def _detect_language(text: str) -> str:
    if not text:
        return "en"
    devanagari = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
    latin = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    total = max(1, devanagari + latin)
    ratio = devanagari / total
    if ratio > 0.70:
        return "hi"
    if devanagari > 0 and latin > 0:
        return "mixed"
    return "en"


def _extract_sections(text: str, pages: list, re_engine=None) -> list[SectionSchema]:
    lines = text.split("\n")
    sections: list[dict] = []
    page_map = _build_page_map(pages)
    current_parent_heading = ""

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _is_noise_line(stripped):
            continue

        clause_match = re.match(r"^(?P<id>\d+(?:\.\d+)*[A-Z]?)\.\s*(?P<body>.+)?$", stripped)
        roman_match = re.match(r"^\((?P<id>[ivxlcdm]+)\)\s*(?P<body>.+)?$", stripped, re.IGNORECASE)
        section_match = re.match(r"^(Section\s+\d+[A-Z]?)[:.\s-]*(?P<body>.*)$", stripped, re.IGNORECASE)
        annex_match = re.match(r"^(Annex(?:ure)?|Appendix|Schedule)\s+([A-Z0-9IVX-]+)[:.\s-]*(?P<body>.*)$", stripped, re.IGNORECASE)

        if clause_match:
            clause_id = clause_match.group("id")
            body = (clause_match.group("body") or "").strip()
            if re.fullmatch(r"\d{4}", clause_id):
                continue
            level = clause_id.count(".") + 1
            heading = body if body else current_parent_heading or clause_id
            sections.append({"id": clause_id, "heading": heading, "start_idx": idx, "level": level})
            if level == 1 and body:
                current_parent_heading = body
            continue

        if roman_match and sections:
            clause_id = roman_match.group("id").lower()
            body = (roman_match.group("body") or "").strip()
            heading = body if body else current_parent_heading or clause_id
            sections.append({"id": clause_id, "heading": heading, "start_idx": idx, "level": 3})
            continue

        if section_match:
            clause_id = section_match.group(1).strip()
            body = (section_match.group("body") or "").strip()
            heading = body if body else clause_id
            sections.append({"id": clause_id, "heading": heading, "start_idx": idx, "level": 1})
            current_parent_heading = heading
            continue

        if annex_match:
            clause_id = f"{annex_match.group(1)} {annex_match.group(2)}".strip()
            body = (annex_match.group("body") or "").strip()
            heading = f"{clause_id} {body}".strip()
            sections.append({"id": clause_id, "heading": heading, "start_idx": idx, "level": 1})
            current_parent_heading = heading
            continue

    result: list[SectionSchema] = []
    for i, section in enumerate(sections):
        end_idx = sections[i + 1]["start_idx"] if i + 1 < len(sections) else len(lines)
        text_block = "\n".join(lines[section["start_idx"]:end_idx]).strip()
        char_offset = sum(len(lines[k]) + 1 for k in range(section["start_idx"]))
        result.append(
            SectionSchema(
                id=section["id"],
                heading=section["heading"],
                text=text_block,
                page_number=_get_page_number(char_offset, page_map),
                clause_type=_finalize_clause_type(
                    re_engine.classify_clause_type(text_block)
                    if re_engine else _classify_clause_type(text_block),
                    text_block,
                ),
                level=section["level"],
            )
        )

    # ── Post-pass: propagate "quoted_reference" from parent lead-in sections
    # to their immediate child sub-clauses.  When a parent section's text ends
    # with a quotation lead-in phrase ("provides as under", "reads as follows",
    # etc.), all contiguous children at a higher nesting level are background
    # quotations of existing law, not new directives.
    _propagate_quoted_reference(result)

    if not result and text.strip():
        result.append(
            SectionSchema(
                id="S1",
                heading=_extract_title(text, ""),
                text=text.strip(),
                page_number=1 if pages else None,
                clause_type=_finalize_clause_type(
                    re_engine.classify_clause_type(text)
                    if re_engine else _classify_clause_type(text),
                    text,
                ),
            )
        )

    return result


def _is_noise_line(line: str) -> bool:
    if re.fullmatch(r"\d+", line):
        return True
    if re.fullmatch(r"[_\-. ]{5,}", line):
        return True
    return False


def _is_quoted_block(text: str) -> bool:
    """Return True if the text block appears to be a quotation of existing law.

    Checks for:
      - Text wrapped in smart/curly quotes (\u201c...\u201d) or regular quotes
      - Text starting with a quotation mark after a clause number
    """
    # Strip leading clause numbering (e.g. "(i) ", "3.1 ", "121A. ")
    body = re.sub(r"^(?:\([a-z]+\)\s*|\d+(?:\.\d+)*[A-Z]?\.\s*)", "", text.strip())
    if not body:
        return False
    # Smart / curly quotes wrapping the whole block
    if (body[0] in '\u201c\u201e"' and body.rstrip()[-1] in '\u201d\u201f"'):
        return True
    return False


# ── Quotation lead-in phrases ──────────────────────────────────────────────────
# These phrases in a parent section indicate that everything that follows
# (until the next same-level section) is a quotation of existing law, not a
# new directive.  Must be checked in lowered text.

_QUOTATION_LEADINS = [
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
]


def _propagate_quoted_reference(sections: list) -> None:
    """Mark child sub-clauses as 'quoted_reference' when their parent section
    ends with a quotation lead-in phrase.

    This handles the common regulatory pattern where a paragraph cites existing
    law and then the sub-clauses (i), (ii), (iii) etc. are the *quoted text*
    of that existing law, not new obligations.

    Example structure:
        Section 1 (level 1): "...Regulations, 2002 provides as under for
                              recognition of premium: \u201c2. Premium"
          Section (i) (level 3): "Premium shall be recognized..."  ← quoted
          Section (ii) (level 3): "\u201cPremium received in Advance\u201d..." ← quoted
          Section (iii) (level 3): "\u201cUnallocated premium\u201d ..."    ← quoted
        Section 2 (level 1): "The Authority has carried out..."   ← NOT quoted
    """
    i = 0
    while i < len(sections):
        section = sections[i]
        lower_text = section.text.lower()

        # Check if this section's text ends with a quotation lead-in
        has_leadin = any(phrase in lower_text for phrase in _QUOTATION_LEADINS)

        if has_leadin:
            parent_level = section.level
            # Also tag the parent itself as quoted_reference if its only
            # purpose is introducing the quotation (no obligation content)
            if section.clause_type not in ("obligation", "penalty"):
                section.clause_type = "quoted_reference"

            # Walk forward: every contiguous child at a deeper level
            # is part of the quotation block
            j = i + 1
            while j < len(sections) and sections[j].level > parent_level:
                sections[j].clause_type = "quoted_reference"
                j += 1
            i = j
        else:
            i += 1


def _finalize_clause_type(clause_type: str, text: str) -> str:
    """Post-process the raw clause type with deterministic override rules.

    The rule-engine classifier may label boilerplate sections as 'obligation'
    because they contain 'shall'.  These overrides take precedence:
      - "come into force / effect" → always effective_date
      - "shall be called / shall be known as" → other (naming clause)
      - quoted-reference propagation is handled separately in _propagate_quoted_reference
    """
    lower = text.lower()
    # Effective-date boilerplate
    if re.search(r"come into (?:force|effect)|with effect from|shall come into (?:force|effect)", lower):
        return "effective_date"
    # Naming / citation boilerplate
    if re.search(r"shall be (?:called|known as|cited as|titled)", lower) and len(text) < 300:
        return "other"
    return clause_type


def _classify_clause_type(text: str) -> str:
    lower = text.lower()
    # ── Quoted reference: text wrapped in curly/smart quotes ──
    stripped = text.strip()
    if _is_quoted_block(stripped):
        return "quoted_reference"
    # ── Quoted reference: parent section with quotation lead-in ──
    if any(phrase in lower for phrase in _QUOTATION_LEADINS):
        return "quoted_reference"
    if "come into force" in lower or "come into effect" in lower or "with effect from" in lower:
        return "effective_date"
    if re.search(r"(please refer|as per|vide|in terms of|dated\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}|circular no\.|directions, \d{4})", lower):
        return "cross_reference"
    if "penalty" in lower or "punishable" in lower or "fine" in lower:
        return "penalty"
    if "means" in lower or "shall mean" in lower or "for the purpose of these directions" in lower:
        return "definition"
    if " shall " in f" {lower} ":
        return "obligation"
    if " may " in f" {lower} ":
        return "permission"
    return "other"


def _build_page_map(pages: list) -> list[tuple[int, int]]:
    page_map = []
    offset = 0
    for page in pages:
        page_map.append((offset, page.get("page_number", 1)))
        offset += page.get("char_count", 0) + 2
    return page_map


def _get_page_number(char_offset: int, page_map: list[tuple[int, int]]) -> Optional[int]:
    if not page_map:
        return None
    page_number = page_map[0][1]
    for offset, candidate in page_map:
        if char_offset >= offset:
            page_number = candidate
        else:
            break
    return page_number


def _extract_tables(text: str) -> list[dict]:
    tables = []
    lines = text.split("\n")
    table_lines = []

    for line in lines:
        if line.count("\t") >= 2 or line.count("|") >= 2:
            table_lines.append(line)
        else:
            if len(table_lines) >= 3:
                tables.append({"raw_text": "\n".join(table_lines), "row_count": len(table_lines)})
            table_lines = []

    if len(table_lines) >= 3:
        tables.append({"raw_text": "\n".join(table_lines), "row_count": len(table_lines)})

    return tables


def _extract_annexures(sections: list[SectionSchema]) -> list[str]:
    keywords = ["annex", "annexure", "schedule", "appendix"]
    return [section.text for section in sections if any(word in section.heading.lower() for word in keywords)]


def _extract_cross_references(text: str) -> list[str]:
    matches = re.findall(
        r"(?:Please refer to|as per|vide|in terms of)\s+(.{10,180}?(?:Circular|Directions|Guidelines|Master Direction).{0,60})",
        text,
        re.IGNORECASE,
    )
    unique = []
    seen = set()
    for match in matches:
        value = re.sub(r"\s+", " ", match).strip()
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _generate_doc_id(source: str, circular_number: Optional[str], title: str) -> str:
    prefix = _safe_source_prefix(source)
    if circular_number:
        return f"{prefix}-{circular_number}".replace("/", "-")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title.strip()).strip("-").lower()
    if not slug:
        slug = hashlib.md5(title.encode()).hexdigest()[:12]
    return slug[:80]


def _safe_source_prefix(source: str) -> str:
    return source if source in SOURCE_VALUES else "OTHER"
