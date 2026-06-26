"""
Compliance JSON repair helper.

Takes an extracted document JSON plus a validation JSON, applies only the
repairs supported by the validator findings, and regenerates validation.
"""

from __future__ import annotations

import copy
import json
import re
from difflib import SequenceMatcher
from typing import Any

from backend.core.logger import get_logger
from backend.core.rule_engine import get_rule_engine
from backend.ingestion.schemas import DocumentStructureSchema, ObligationSchema
from backend.ingestion.validator import validate_extraction

logger = get_logger(__name__)


FIELD_ORDER = [
    "actor",
    "action",
    "deadline",
    "domain",
    "departments",
    "severity",
    "confidence",
    "effective_date",
]


def repair_extracted_json(extracted_json: dict[str, Any], validation_json: dict[str, Any]) -> dict[str, Any]:
    """Repair extracted JSON using validation findings, then rerun validation."""
    repaired = copy.deepcopy(extracted_json)
    repair_summary = {
        "obligations_added": [],
        "obligations_removed": [],
        "fields_corrected": [],
        "confidence_changes": [],
    }

    obligations = [copy.deepcopy(item) for item in repaired.get("obligations", []) if isinstance(item, dict)]
    validation = validation_json or {}
    validation_missed = validation.get("missed_obligations", []) or []
    validation_incorrect = validation.get("incorrect_extractions", []) or []

    obligations_by_id = {ob.get("id"): ob for ob in obligations if ob.get("id")}
    obligations_by_clause = {}
    for ob in obligations:
        obligations_by_clause.setdefault(str(ob.get("clause_number", "")), []).append(ob)

    # Step 1: missed obligations.
    for finding in validation_missed:
        if not isinstance(finding, dict):
            continue
        raw_text = str(finding.get("raw_text") or "").strip()
        clause_number = str(finding.get("clause_number") or "").strip()
        if not raw_text:
            continue
        if _already_covered(raw_text, obligations, clause_number):
            continue
        new_ob = _build_obligation_from_missed_text(
            raw_text=raw_text,
            clause_number=clause_number or _next_clause_number(repaired),
            doc=repaired,
            existing=obligations,
        )
        if new_ob:
            obligations.append(new_ob)
            repair_summary["obligations_added"].append(new_ob["id"])

    # Step 2: incorrect extractions.
    for finding in validation_incorrect:
        if not isinstance(finding, dict):
            continue
        obligation_id = str(finding.get("obligation_id") or "").strip()
        field = str(finding.get("field") or "").strip()
        if not obligation_id or not field:
            continue
        obligation = obligations_by_id.get(obligation_id)
        if not obligation:
            continue

        changed = _repair_obligation_field(obligation, finding, repaired)
        if changed:
            repair_summary["fields_corrected"].extend(changed)
        if field == "confidence":
            old_conf = float(obligation.get("confidence", 0.0) or 0.0)
            new_conf = min(0.94, max(0.50, old_conf - 0.07))
            obligation["confidence"] = round(new_conf, 4)
            repair_summary["confidence_changes"].append(
                {"obligation_id": obligation_id, "from": old_conf, "to": obligation["confidence"]}
            )

    # Step 3: remove clearly invalid obligations.
    kept = []
    for obligation in obligations:
        if _is_invalid_obligation(obligation):
            repair_summary["obligations_removed"].append(obligation.get("id"))
            continue
        kept.append(obligation)
    obligations = kept

    # Step 4: split obvious merged obligations.
    split_obligations: list[dict[str, Any]] = []
    for obligation in obligations:
        parts = _split_merged_obligation(obligation)
        if len(parts) == 1:
            split_obligations.extend(parts)
            continue
        repair_summary["obligations_removed"].append(obligation.get("id"))
        split_obligations.extend(parts)
        repair_summary["obligations_added"].extend([part["id"] for part in parts])
    obligations = split_obligations

    # Step 5: repair effective date.
    if not repaired.get("effective_date"):
        repaired["effective_date"] = _repair_effective_date(repaired, validation)
        if repaired["effective_date"]:
            repair_summary["fields_corrected"].append("effective_date")

    # Re-number IDs only for newly created obligations to keep traceability stable.
    obligations = _deduplicate_obligations(obligations)
    repaired["obligations"] = obligations

    # Step 6: regenerate validation.
    regenerated_validation = _regenerate_validation(repaired)

    return {
        "repaired_json": repaired,
        "repair_summary": repair_summary,
        "validation": _serialize_validation(regenerated_validation),
    }


def repair_extracted_json_from_files(extracted_path: str, validation_path: str) -> dict[str, Any]:
    with open(extracted_path, "r", encoding="utf-8") as f:
        extracted = json.load(f)
    with open(validation_path, "r", encoding="utf-8") as f:
        validation = json.load(f)
    return repair_extracted_json(extracted, validation)


def _regenerate_validation(repaired: dict[str, Any]):
    try:
        doc = DocumentStructureSchema(**repaired)
        raw_text = "\n".join(section.get("text", "") for section in repaired.get("sections", []) if isinstance(section, dict))
        if not raw_text:
            raw_text = _concat_obligation_texts(repaired.get("obligations", []))
        return validate_extraction(raw_text=raw_text, obligations=doc.obligations, doc_effective_date=doc.effective_date)
    except Exception as exc:
        logger.warning("Repair validation regeneration failed", error=str(exc))
        return validate_extraction(raw_text="", obligations=[], doc_effective_date=None)


def _serialize_validation(validation) -> dict[str, Any]:
    if hasattr(validation, "model_dump"):
        return validation.model_dump(mode="json")
    return dict(validation or {})


def _repair_obligation_field(obligation: dict[str, Any], finding: dict[str, Any], repaired_doc: dict[str, Any]) -> list[str]:
    field = str(finding.get("field") or "").strip()
    current = finding.get("current_value")
    corrected = finding.get("correct_value")
    reason = str(finding.get("reason") or "").lower()
    changed: list[str] = []

    if field == "action":
        source_text = str(obligation.get("action") or "")
        normalized = _normalize_action(source_text)
        if normalized and normalized != source_text:
            obligation["action"] = normalized
            changed.append("action")
        obligation["notes"] = _strip_validation_tags(obligation.get("notes"), ["action_quality"])

    elif field == "actor":
        actor = _normalize_actor(str(corrected or current or obligation.get("actor") or ""))
        if actor and actor != obligation.get("actor"):
            obligation["actor"] = actor
            changed.append("actor")
        obligation["notes"] = _strip_validation_tags(obligation.get("notes"), ["actor_conflict"])

    elif field == "deadline" or field.startswith("deadline."):
        deadline_text = _extract_deadline_text(str(corrected or current or ""), obligation.get("action", ""))
        if deadline_text:
            deadline = obligation.setdefault("deadline", {})
            deadline["text"] = deadline_text
            if not deadline.get("duration"):
                duration = _extract_duration(deadline_text)
                if duration:
                    deadline["duration"] = duration
            if not deadline.get("urgency") or deadline.get("urgency") == "ongoing":
                deadline["urgency"] = "short_term" if deadline.get("duration") else "triggered"
            changed.append("deadline")
        obligation["notes"] = _strip_validation_tags(obligation.get("notes"), ["deadline_conflict"])

    elif field == "domain":
        domain = _infer_domain(obligation, corrected, reason)
        if domain and domain != obligation.get("domain"):
            obligation["domain"] = domain
            changed.append("domain")
            obligation["departments"] = _departments_for_domain(domain, obligation.get("actor", ""), obligation.get("action", ""))
            changed.append("departments")
        obligation["notes"] = _strip_validation_tags(obligation.get("notes"), ["domain_conflict"])

    elif field == "departments":
        departments = _departments_for_domain(obligation.get("domain", "Other"), obligation.get("actor", ""), obligation.get("action", ""))
        if departments and departments != obligation.get("departments"):
            obligation["departments"] = departments
            changed.append("departments")
        obligation["notes"] = _strip_validation_tags(obligation.get("notes"), ["departments_conflict"])

    elif field == "severity":
        severity = _infer_severity(obligation)
        if severity and severity != obligation.get("severity"):
            obligation["severity"] = severity
            changed.append("severity")
        obligation["notes"] = _strip_validation_tags(obligation.get("notes"), ["severity_conflict"])

    elif field == "confidence":
        old_conf = float(obligation.get("confidence", 0.0) or 0.0)
        obligation["confidence"] = round(max(0.50, min(0.94, old_conf - 0.05)), 4)
        changed.append("confidence")

    if changed:
        obligation["notes"] = _append_note(obligation.get("notes"), f"[repaired:{','.join(changed)}]")
    return changed


def _build_obligation_from_missed_text(
    raw_text: str,
    clause_number: str,
    doc: dict[str, Any],
    existing: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _is_obligation_like(raw_text):
        return None
    actor = _infer_actor(raw_text, doc)
    domain = _infer_domain({"action": raw_text, "domain": "Other"}, None, raw_text.lower())
    departments = _departments_for_domain(domain, actor, raw_text)
    # Pass None as candidate so we only extract deadline from explicit "within X days" phrases
    deadline_text = _extract_deadline_text(None, raw_text)
    deadline = {
        "text": deadline_text or "ongoing",
        "absolute_date": None,
        "duration": _extract_duration(deadline_text or ""),
        "urgency": "short_term" if deadline_text else "ongoing",
    }
    action = _normalize_action(raw_text)
    new_id = _next_obligation_id(doc, clause_number, existing)
    return {
        "id": new_id,
        "section_id": clause_number,
        "clause_number": clause_number,
        "actor": actor,
        "action": action,
        "obligation_type": "mandatory",
        "trigger": _extract_trigger(raw_text),
        "deadline": deadline,
        "domain": domain,
        "departments": departments,
        "severity": _infer_severity({"action": raw_text, "deadline": deadline}),
        "severity_reason": "Derived from validation finding and source clause wording.",
        "evidence_required": _evidence_for(domain, raw_text),
        "penalty_if_missed": None,
        "fine_exposure_inr": None,
        "cross_references": _extract_cross_refs(raw_text),
        "confidence": 0.84,
        "notes": "[repaired:missed_obligation]",
    }


def _already_covered(raw_text: str, obligations: list[dict[str, Any]], clause_number: str) -> bool:
    for obligation in obligations:
        if clause_number and str(obligation.get("clause_number", "")) == clause_number:
            if SequenceMatcher(None, raw_text.lower(), str(obligation.get("action", "")).lower()).ratio() > 0.58:
                return True
        if SequenceMatcher(None, raw_text.lower(), str(obligation.get("action", "")).lower()).ratio() > 0.72:
            return True
    return False


def _is_invalid_obligation(obligation: dict[str, Any]) -> bool:
    action = str(obligation.get("action") or "").strip().lower()
    if not action:
        return True
    # Boilerplate directive/naming clauses — not actionable obligations
    BOILERPLATE_PHRASES = (
        "shall be called",
        "shall be known as",
        "shall be titled",
        "is hereby issued",
        "come into force",
        "come into effect",
        "shall come into force",
        "shall come into effect",
        "these directions shall",
        "these amendment directions shall",
        "these regulations shall",
        "this circular shall",
    )
    if any(phrase in action for phrase in BOILERPLATE_PHRASES):
        return True
    if "as under" in action and "shall" not in action and "must" not in action:
        return True
    return False


def _split_merged_obligation(obligation: dict[str, Any]) -> list[dict[str, Any]]:
    action = str(obligation.get("action") or "")
    if not _looks_like_merged_action(action):
        return [obligation]
    parts = _split_action_text(action)
    if len(parts) <= 1:
        return [obligation]
    split: list[dict[str, Any]] = []
    for idx, part in enumerate(parts, 1):
        clone = copy.deepcopy(obligation)
        clone["id"] = f"{obligation.get('id', 'ob')}-S{idx}"
        clone["action"] = part
        clone["confidence"] = round(max(0.50, float(clone.get("confidence", 0.8)) - 0.04 * (idx - 1)), 4)
        clone["notes"] = _append_note(clone.get("notes"), "[split:merged_obligation]")
        split.append(clone)
    return split


def _looks_like_merged_action(action: str) -> bool:
    if re.search(r"\(\s*[a-eivxlcdm]+\s*\)|\b[a-e]\)", action, re.IGNORECASE):
        return True
    if action.count(";") >= 1 and len(re.findall(r"\b(?:shall|must|required to)\b", action, re.IGNORECASE)) >= 2:
        return True
    return False


def _split_action_text(action: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", action).strip(" ;.")
    if not cleaned:
        return []
    parts = re.split(r"\s*;\s+", cleaned)
    results: list[str] = []
    for part in parts:
        stripped = re.sub(r"^\s*(?:\([a-z]+\)|\([ivxlcdm]+\)|[a-z]\)|[ivxlcdm]+\)|\d+\.)\s*", "", part, flags=re.IGNORECASE)
        stripped = re.sub(r"\s+", " ", stripped).strip(" ;.")
        if stripped:
            results.append(_normalize_action(stripped))
    return results


def _clone_obligation_for_parts(obligation: dict[str, Any], parts: list[str]) -> list[dict[str, Any]]:
    clones: list[dict[str, Any]] = []
    for idx, part in enumerate(parts, 1):
        clone = copy.deepcopy(obligation)
        clone["id"] = f"{obligation.get('id', 'ob')}-S{idx}"
        clone["action"] = part
        clones.append(clone)
    return clones


def _normalize_action(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip(" ;.")
    cleaned = re.sub(r"^\s*(?:where under|accordingly|however|therefore|thus)\s*,?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(?:the\s+)?(?:regulated entities?|reinsurers?|reinsurer|banks?|nbfcs?|insurance companies?)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(?:shall|must|may|is required to|are required to)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(?:ensure that|submit|file|report|disclose|maintain|obtain|update|review|publish|provide)\s+", lambda m: m.group(0).capitalize(), cleaned, flags=re.IGNORECASE)
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    return cleaned


def _normalize_actor(text: str) -> str:
    lower = text.lower()
    if not lower:
        return ""
    if "reinsurer" in lower or "frb" in lower:
        return "FRBs/Reinsurers" if "frb" in lower or "reinsurer" in lower else text
    if "nbfc" in lower:
        return "NBFC"
    if "bank" in lower and "commercial" not in lower:
        return "Bank"
    if "commercial bank" in lower:
        return "Commercial Bank"
    if "regulatory authority" in lower or "irda" in lower:
        return "IRDAI"
    if "regulated entity" in lower:
        return "Regulated Entity"
    return text.strip()


def _infer_actor(raw_text: str, doc: dict[str, Any]) -> str:
    analysis = doc.get("analysis") or {}
    primary = str(analysis.get("primary_actor") or "").strip()
    if primary:
        return _normalize_actor(primary)
    return _normalize_actor(raw_text)


def _infer_domain(obligation: dict[str, Any], corrected: Any, reason: str) -> str:
    current = str(obligation.get("domain") or "").strip()
    action = str(obligation.get("action") or "")
    haystack = f"{action} {current} {corrected or ''} {reason}".lower()
    if any(term in haystack for term in ["audit", "disclosure", "report", "return", "statement"]):
        return "ReportingAudit"
    if any(term in haystack for term in ["board", "approval", "governance", "policy"]):
        return "Governance"
    if any(term in haystack for term in ["premium", "insurance", "reinsur", "claim"]):
        return "ReportingAudit"
    return current or "Other"


def _departments_for_domain(domain: str, actor: str, text: str) -> list[str]:
    re_engine = get_rule_engine()
    departments = re_engine.get_departments(domain=domain, sub_type=None, actor=actor, text=text)
    return departments or ["Compliance"]


def _infer_severity(obligation: dict[str, Any]) -> str:
    text = f"{obligation.get('action', '')} {obligation.get('deadline', {}).get('text', '')}".lower()
    if any(term in text for term in ["within 7 days", "immediately", "forthwith", "urgent"]):
        return "high"
    if any(term in text for term in ["within 15 days", "within 30 days", "report"]):
        return "high"
    if any(term in text for term in ["board", "approval", "disclosure", "annual report"]):
        return "medium"
    return "medium"


def _repair_effective_date(repaired: dict[str, Any], validation: dict[str, Any]) -> str | None:
    analysis = repaired.get("analysis") or {}
    candidate = str(analysis.get("likely_effective_date_text") or "").strip()
    if candidate:
        return candidate
    return validation.get("missing_effective_date")


def _extract_deadline_text(candidate: str, text: str) -> str | None:
    """Extract a short, meaningful deadline phrase from text.

    The candidate value may be a validator suggestion (e.g. "within 30 days")
    or "null" / the full clause text (in which case we ignore it and parse
    the source text directly).
    """
    candidate = str(candidate or "").strip()

    # A non-trivial candidate that explicitly contains deadline language
    if (
        candidate
        and candidate.lower() not in ("null", "none", "ongoing", "n/a")
        and len(candidate) < 120  # not a full clause
        and re.search(r"\b(?:within|by|days|months|weeks|years|before|after|from)\b", candidate, re.IGNORECASE)
        and re.search(r"\d", candidate)  # must have a number (30 days, etc.) or a date
    ):
        return re.sub(r"\s+", " ", candidate).strip(" .;")

    # Parse the source text for a deadline phrase
    match = re.search(
        r"(within\s+\d+\s+(?:days|months|weeks)(?:\s+(?:from|of|after)\s+[^.;]+)?)",
        text,
        re.IGNORECASE,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip(" .;")
    return None


def _extract_duration(text: str) -> str | None:
    match = re.search(r"\b(\d+\s+(?:days|months))\b", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _extract_trigger(text: str) -> str:
    match = re.search(
        r"(if[^.;]+|where[^.;]+|provided that[^.;]+|subject to[^.;]+|in case of[^.;]+)",
        text,
        re.IGNORECASE,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip(" .;")
    return "always"


def _extract_cross_refs(text: str) -> list[str]:
    refs = re.findall(r"\bpara\s+\d+(?:\([a-z]\))?|\bsection\s+\d+(?:\s*\(\d+\))?", text, re.IGNORECASE)
    return _unique([re.sub(r"\s+", " ", ref).strip() for ref in refs])


def _evidence_for(domain: str, text: str) -> list[str]:
    mapping = {
        "ReportingAudit": ["Submission acknowledgement", "Audit trail or return copy"],
        "Governance": ["Board note or approval", "Policy record"],
        "Other": ["Supporting source clause"],
    }
    evidence = list(mapping.get(domain, mapping["Other"]))
    lower = text.lower()
    if "annual report" in lower:
        evidence.append("Annual report extract")
    if "disclosure" in lower:
        evidence.append("Disclosure record")
    return _unique(evidence)


def _next_clause_number(repaired: dict[str, Any]) -> str:
    numbers = []
    for ob in repaired.get("obligations", []):
        if isinstance(ob, dict):
            clause = str(ob.get("clause_number") or "")
            if clause.isdigit():
                numbers.append(int(clause))
    return str(max(numbers, default=0) + 1)


def _next_obligation_id(doc: dict[str, Any], clause_number: str, existing: list[dict[str, Any]]) -> str:
    section = str(clause_number or doc.get("doc_id", "X"))
    prefix = f"{section}-OB"
    existing_numbers = []
    for obligation in existing:
        oid = str(obligation.get("id") or "")
        if oid.startswith(prefix):
            tail = oid[len(prefix):]
            try:
                match = re.match(r"(\d+)", tail)
                if match:
                    existing_numbers.append(int(match.group(1)))
            except Exception:
                continue
    return f"{prefix}{max(existing_numbers, default=0) + 1}"


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _concat_obligation_texts(obligations: list[dict[str, Any]]) -> str:
    return "\n".join(str(ob.get("action", "")) for ob in obligations if isinstance(ob, dict))


def _append_note(notes: str | None, addition: str) -> str:
    return f"{notes} {addition}".strip() if notes else addition


def _strip_validation_tags(notes: str | None, tags: list[str]) -> str | None:
    if not notes:
        return notes
    cleaned = str(notes)
    for tag in tags:
        cleaned = re.sub(rf"\s*\[{re.escape(tag)}:[^\]]+\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\s*\[{re.escape(tag)}\s*-\s*needs review\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\s*\[{re.escape(tag)}\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _is_obligation_like(text: str) -> bool:
    lower = str(text or "").lower()
    return bool(re.search(r"\b(shall|must|required|obligated|within\s+\d+\s+days|shall ensure|shall submit|shall report|shall disclose)\b", lower))


def _deduplicate_obligations(obligations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for obligation in obligations:
        duplicate = False
        for existing in unique:
            same_clause = str(existing.get("clause_number")) == str(obligation.get("clause_number"))
            similarity = SequenceMatcher(None, str(existing.get("action", "")).lower(), str(obligation.get("action", "")).lower()).ratio()
            if same_clause and similarity > 0.90 and existing.get("actor") == obligation.get("actor"):
                duplicate = True
                break
        if not duplicate:
            unique.append(obligation)
    return unique
