"""
ORBITAL Extraction Validator
Reviews extracted obligations against original regulatory text.
Department mapping and validation rule checks are delegated to the JSON Rule Engine.
"""

import json
import re
from difflib import SequenceMatcher

from backend.core.llm_client import llm
from backend.core.logger import get_logger
from backend.core.rule_engine import get_rule_engine
from backend.core.utils import parse_date_to_iso
from backend.ingestion.schemas import (
    IncorrectExtractionSchema,
    ObligationSchema,
    ValidationFindingSchema,
    ValidationResultSchema,
)

logger = get_logger(__name__)

CLAUSE_LINE_RE = re.compile(
    r"^(?P<clause>[A-Za-z]?\d+(?:\.\d+)*[A-Z]?|\([ivxlcdm]+\))\.\s*(?P<body>.+)$",
    re.IGNORECASE,
)


def validate_extraction(raw_text: str, obligations: list[ObligationSchema] | list[dict], doc_effective_date: str | None = None) -> ValidationResultSchema:
    """Validate extracted obligations against raw document text."""
    try:
        normalized_obligations = [_coerce_obligation(item) for item in obligations]
        normalized_obligations = [item for item in normalized_obligations if item is not None]

        llm_result = _validate_with_llm(raw_text, normalized_obligations)
        heuristic_result = _validate_with_heuristics(raw_text, normalized_obligations)

        merged = _merge_results(llm_result, heuristic_result)

        # If document already extracted effective date, suppress validator false positives
        if doc_effective_date:
            merged.missing_effective_date = None
            merged.missed_obligations = [
                m for m in merged.missed_obligations
                if not (
                    "effective date" in m.reason_missed.lower() 
                    or "come into force" in m.raw_text.lower() 
                    or "effect from" in m.raw_text.lower()
                    or "effective from" in m.raw_text.lower()
                )
            ]

        return merged
    except Exception as e:
        logger.error("Extraction validation failed", error=str(e))
        return ValidationResultSchema(
            overall_confidence=0.0,
            validation_notes="Validation failed during processing.",
        )


def _validate_with_llm(raw_text: str, obligations: list[ObligationSchema]) -> ValidationResultSchema:
    content = json.dumps(
        {
            "raw_text": raw_text,
            "obligations_json": [item.model_dump(mode="json") for item in obligations],
        },
        ensure_ascii=False,
    )
    result = llm.validate_extraction(content)
    if isinstance(result, dict):
        try:
            return ValidationResultSchema(**result)
        except Exception:
            pass
    return ValidationResultSchema(
        overall_confidence=0.0,
        validation_notes="LLM validation unavailable; used heuristic checks.",
    )


def _validate_with_heuristics(raw_text: str, obligations: list[ObligationSchema]) -> ValidationResultSchema:
    missed = []
    incorrect = []

    clause_lines = _extract_clause_lines(raw_text)
    effective_date = _extract_effective_date(raw_text)
    extracted_effective_date = any(
        obligation.deadline.absolute_date == effective_date
        or obligation.deadline.text.lower().startswith("with effect from")
        or "come into force" in obligation.action.lower()
        for obligation in obligations
        if effective_date
    )

    for clause_number, sentence in clause_lines:
        lower = sentence.lower()
        if " shall " in f" {lower} " and not _has_matching_obligation(sentence, clause_number, obligations):
            missed.append(
                ValidationFindingSchema(
                    clause_number=clause_number,
                    raw_text=sentence,
                    reason_missed="Contains a mandatory 'shall' clause but no matching extracted obligation was found.",
                )
            )

    for obligation in obligations:
        incorrect.extend(_issues_from_notes(obligation))

        if obligation.obligation_type == "mandatory" and _looks_discretionary(obligation):
            incorrect.append(
                IncorrectExtractionSchema(
                    obligation_id=obligation.id,
                    field="obligation_type",
                    current_value=obligation.obligation_type,
                    correct_value="discretionary",
                    reason="The clause uses discretionary language such as 'may' or 'at its discretion'.",
                )
            )

        deadline_text = obligation.deadline.text or ""
        if "within" in obligation.action.lower() and deadline_text == "ongoing":
            incorrect.append(
                IncorrectExtractionSchema(
                    obligation_id=obligation.id,
                    field="deadline.text",
                    current_value=deadline_text,
                    correct_value="deadline present in clause",
                    reason="The action still contains a deadline phrase, indicating the deadline was not captured cleanly.",
                )
            )

        if _action_too_verbatim_or_vague(obligation):
            incorrect.append(
                IncorrectExtractionSchema(
                    obligation_id=obligation.id,
                    field="action",
                    current_value=obligation.action,
                    correct_value="Summarized action needed",
                    reason="The action appears too vague or too close to raw clause wording.",
                )
            )

        expected_departments = _expected_departments(obligation)
        if expected_departments and sorted(expected_departments) != sorted(obligation.departments):
            incorrect.append(
                IncorrectExtractionSchema(
                    obligation_id=obligation.id,
                    field="departments",
                    current_value=", ".join(obligation.departments),
                    correct_value=", ".join(expected_departments),
                    reason="Department mapping does not align with the operational domain cues in the clause.",
                )
            )

    incorrect.extend(_find_duplicates(obligations))
    incorrect.extend(_apply_validation_rule_checks(obligations))

    notes = "Heuristic + rule-engine validation reviewed mandatory clauses, effective dates, deadlines, department mapping, duplicate extraction, and JSON validation rules."
    confidence = 0.82 if raw_text.strip() and obligations else 0.55

    return ValidationResultSchema(
        missed_obligations=missed,
        incorrect_extractions=incorrect,
        missing_effective_date=effective_date if effective_date and not extracted_effective_date else None,
        overall_confidence=confidence,
        validation_notes=notes,
    )


def _extract_clause_lines(raw_text: str) -> list[tuple[str, str]]:
    results = []
    for line in raw_text.splitlines():
        stripped = re.sub(r"\s+", " ", line).strip()
        if len(stripped) < 15:
            continue
        match = CLAUSE_LINE_RE.match(stripped)
        if match:
            results.append((match.group("clause"), stripped))
    return results


def _has_matching_obligation(sentence: str, clause_number: str, obligations: list[ObligationSchema]) -> bool:
    for obligation in obligations:
        if obligation.clause_number == clause_number:
            return True
        similarity = SequenceMatcher(None, sentence.lower(), obligation.action.lower()).ratio()
        if similarity > 0.45:
            return True
    return False


def _looks_discretionary(obligation: ObligationSchema) -> bool:
    trigger = (obligation.trigger or "").lower()
    action = obligation.action.lower()
    notes = (obligation.notes or "").lower()
    return any(token in f"{trigger} {action} {notes}" for token in ["may", "can", "at its discretion"])


def _action_too_verbatim_or_vague(obligation: ObligationSchema) -> bool:
    action = obligation.action.strip().lower()
    if len(action) < 18:
        return True
    vague_prefixes = ["be ", "do ", "review manually", "review the clause manually"]
    if any(action.startswith(prefix) for prefix in vague_prefixes):
        return True
    return False


def _issues_from_notes(obligation: ObligationSchema) -> list[IncorrectExtractionSchema]:
    """Promote candidate-engine note tags into deterministic review findings."""
    notes = obligation.notes or ""
    if not notes:
        return []

    issues: list[IncorrectExtractionSchema] = []
    field_mapping = {
        "actor_conflict": "actor",
        "domain_conflict": "domain",
        "departments_conflict": "departments",
        "severity_conflict": "severity",
        "deadline_conflict": "deadline",
    }

    for tag, field_name in field_mapping.items():
        match = re.search(rf"\[{tag}:\s*([^\]]+)\]", notes)
        if not match:
            continue
        issues.append(
            IncorrectExtractionSchema(
                obligation_id=obligation.id,
                field=field_name,
                current_value=_current_value_for_field(obligation, field_name),
                correct_value="review competing candidates",
                reason=f"The candidate engine found conflicting {field_name} signals: {match.group(1)}.",
            )
        )

    if "[action_quality: verbatim - needs review]" in notes.lower():
        issues.append(
            IncorrectExtractionSchema(
                obligation_id=obligation.id,
                field="action",
                current_value=obligation.action,
                correct_value="Summarized action needed",
                reason="The candidate engine marked the action as too close to source text.",
            )
        )

    return issues


def _current_value_for_field(obligation: ObligationSchema, field_name: str) -> str:
    if field_name == "actor":
        return obligation.actor
    if field_name == "domain":
        return obligation.domain
    if field_name == "departments":
        return ", ".join(obligation.departments)
    if field_name == "severity":
        return obligation.severity
    if field_name == "deadline":
        return obligation.deadline.text
    return ""


def _expected_departments(obligation: ObligationSchema) -> list[str]:
    """Use the rule engine's full department mapping instead of the partial hardcoded dict."""
    re_engine = get_rule_engine()
    return re_engine.get_departments(
        domain=obligation.domain,
        sub_type=None,
        actor=obligation.actor,
        text=obligation.action,
    )


def _find_duplicates(obligations: list[ObligationSchema]) -> list[IncorrectExtractionSchema]:
    duplicates = []
    for i, left in enumerate(obligations):
        for right in obligations[i + 1:]:
            if left.section_id != right.section_id:
                continue
            similarity = SequenceMatcher(None, left.action.lower(), right.action.lower()).ratio()
            if similarity > 0.88 and left.actor == right.actor:
                duplicates.append(
                    IncorrectExtractionSchema(
                        obligation_id=right.id,
                        field="duplicate",
                        current_value=right.action,
                        correct_value=left.action,
                        reason="This obligation substantially duplicates another extraction in the same clause and should likely be merged.",
                    )
                )
    return duplicates


def _apply_validation_rule_checks(obligations: list[ObligationSchema]) -> list[IncorrectExtractionSchema]:
    """
    Run deterministic checks from validations/validation_rules.json.
    Only implements checks that are fully deterministic (no external text needed).
    """
    re_engine = get_rule_engine()
    issues: list[IncorrectExtractionSchema] = []

    for ob in obligations:
        # VAL-003: deadline phrase left in action text
        if re.search(r"\bwithin\s+\d+\s+(?:days|months)", ob.action, re.IGNORECASE):
            if ob.deadline.text == "ongoing":
                issues.append(IncorrectExtractionSchema(
                    obligation_id=ob.id,
                    field="deadline.text",
                    current_value=ob.deadline.text,
                    correct_value="deadline present in clause",
                    reason="(VAL-003) Deadline phrase found in action text but deadline was not extracted.",
                ))

        # VAL-004: discretionary marked mandatory
        action_lower = ob.action.lower()
        if ob.obligation_type == "mandatory" and re.search(r"\bmay\b|at its discretion", action_lower):
            issues.append(IncorrectExtractionSchema(
                obligation_id=ob.id,
                field="obligation_type",
                current_value=ob.obligation_type,
                correct_value="discretionary",
                reason="(VAL-004) Action text uses discretionary language but obligation_type is mandatory.",
            ))

        # VAL-006: vague action
        if len(ob.action.strip()) < 20:
            issues.append(IncorrectExtractionSchema(
                obligation_id=ob.id,
                field="action",
                current_value=ob.action,
                correct_value="Summarized action needed",
                reason="(VAL-006) Action text is too short — likely vague or incomplete.",
            ))

        # VAL-015: prohibition with low/medium severity
        if ob.obligation_type == "mandatory" and "shall not" in ob.action.lower():
            if ob.severity in ("low", "medium"):
                issues.append(IncorrectExtractionSchema(
                    obligation_id=ob.id,
                    field="severity",
                    current_value=ob.severity,
                    correct_value="high",
                    reason="(VAL-015) Prohibition clauses should be at least high severity.",
                ))

        # VAL-018: confidence below threshold
        if ob.confidence < 0.65:
            issues.append(IncorrectExtractionSchema(
                obligation_id=ob.id,
                field="confidence",
                current_value=str(round(ob.confidence, 3)),
                correct_value=">= 0.65",
                reason="(VAL-018) Obligation confidence is below the minimum threshold — review or drop.",
            ))

    return issues


def _extract_effective_date(raw_text: str) -> str | None:
    patterns = [
        r"(?:come into force|come into effect|with effect from)\s+(?:on\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        r"(?:come into force|come into effect|with effect from)\s+(?:on\s+)?(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            return parse_date_to_iso(match.group(1))
    return None


def _parse_date_to_iso(value: str) -> str | None:
    """Thin shim — delegates to the shared utility."""
    return parse_date_to_iso(value)


def _coerce_obligation(item) -> ObligationSchema | None:
    if isinstance(item, ObligationSchema):
        return item
    if isinstance(item, dict):
        try:
            return ObligationSchema(**item)
        except Exception:
            return None
    return None


def _merge_results(llm_result: ValidationResultSchema, heuristic_result: ValidationResultSchema) -> ValidationResultSchema:
    missed = list(heuristic_result.missed_obligations)
    seen_missed = {(item.clause_number, item.raw_text) for item in missed}
    for item in llm_result.missed_obligations:
        key = (item.clause_number, item.raw_text)
        if key not in seen_missed:
            missed.append(item)
            seen_missed.add(key)

    incorrect = list(heuristic_result.incorrect_extractions)
    seen_incorrect = {(item.obligation_id, item.field, item.reason) for item in incorrect}
    for item in llm_result.incorrect_extractions:
        key = (item.obligation_id, item.field, item.reason)
        if key not in seen_incorrect:
            incorrect.append(item)
            seen_incorrect.add(key)

    missing_effective_date = heuristic_result.missing_effective_date or llm_result.missing_effective_date
    overall_confidence = max(heuristic_result.overall_confidence, llm_result.overall_confidence)
    notes = heuristic_result.validation_notes
    if llm_result.validation_notes and llm_result.validation_notes not in notes:
        notes = f"{notes} {llm_result.validation_notes}".strip()
    fix_summary = _compose_fix_summary(incorrect)
    if fix_summary:
        notes = f"{notes} {fix_summary}".strip()

    return ValidationResultSchema(
        missed_obligations=missed,
        incorrect_extractions=incorrect,
        missing_effective_date=missing_effective_date,
        overall_confidence=overall_confidence,
        validation_notes=notes,
    )


def _compose_fix_summary(incorrect: list[IncorrectExtractionSchema]) -> str:
    """Summarise the main corrective action for the validator output."""
    if not incorrect:
        return ""

    fixes: list[str] = []
    for item in incorrect[:4]:
        field = item.field.replace(".", " ")
        fixes.append(f"fix {field} for {item.obligation_id}")
    return f"Fix summary: {', '.join(fixes)}."
