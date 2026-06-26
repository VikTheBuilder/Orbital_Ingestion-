"""
Candidate-based obligation extraction for small-model RegTech ingestion.

This module keeps the model out of primary extraction.  It uses deterministic
field candidates, vote aggregation, conflict penalties, and atomic splitting.
LLM usage remains limited to downstream validation and optional action
summarisation in obligation_extractor.py.

Text normalization, splitting, and action extraction are delegated to
backend.ingestion.text_normalizer for format-agnostic, robust handling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from backend.core.retrieval import retrieve_regulatory_patterns
from backend.ingestion.schemas import DeadlineSchema, ObligationSchema
from backend.ingestion.text_normalizer import (
    ACTION_MAX_CHARS,
    MIN_UNIT_LENGTH,
    VERBATIM_MIN_LENGTH,
    VERBATIM_THRESHOLD,
    action_quality_score,
    extract_action_phrase,
    is_boilerplate,
    is_quoted_content,
    split_into_obligation_units,
)


# Keep these for backward-compat with other code that imports from here
ACTION_MIN_LENGTH = MIN_UNIT_LENGTH
ACTION_MAX_LENGTH = ACTION_MAX_CHARS
ACTION_VERBATIM_THRESHOLD = VERBATIM_THRESHOLD

OBLIGATION_CLAUSE_TYPES = {"obligation", "penalty", "permission"}
SKIP_CLAUSE_TYPES = {"definition", "cross_reference", "effective_date", "quoted_reference"}

QUOTE_LEADINS = [
    "provides as under",
    "provides as follows",
    "reads as under",
    "reads as follows",
    "states as under",
    "states as follows",
    "stipulates as under",
    "stipulates as follows",
    "is reproduced below",
    "is extracted below",
    "is quoted below",
    "existing regulations provide",
    "relevant provisions are as under",
    "attention is invited to",
    "please refer to",
]

OPERATIONAL_RESTARTS = [
    "accordingly",
    "it has now been decided",
    "it is advised",
    "all regulated entities shall",
    "banks shall",
    "insurers shall",
    "with effect from",
]

REPORTING_TERMS = (
    "submit", "file", "furnish", "report", "intimate", "disclose",
    "publish", "provide", "statement", "return", "annual report",
)

APPROVAL_TERMS = ("approval", "permission", "prior approval", "prior written consent")

PROHIBITION_TERMS = ("shall not", "must not", "prohibited", "not permitted")

GENERIC_ACTORS = {"Regulated Entity", "Bank", ""}

DOMAIN_ALLOWED = {
    "KYC_AML",
    "Cybersecurity",
    "DataPrivacy",
    "FinancialInclusion",
    "BusinessContinuity",
    "FraudManagement",
    "CapitalAdequacy",
    "Payments",
    "CustomerService",
    "Governance",
    "ITInfrastructure",
    "ReportingAudit",
    "HR_Training",
    "FEMA",
    "Other",
}


@dataclass
class FieldCandidate:
    field: str
    value: Any
    evidence: str
    source: str
    confidence: float
    span: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateBundle:
    unit_id: str
    text: str
    section_id: str
    clause_number: str
    candidates: list[FieldCandidate]
    source: str | None = None
    clause_type: str = "other"


@dataclass
class VoteResult:
    value: Any
    score: float
    margin: float
    conflict: bool
    votes: list[FieldCandidate]


def extract_candidate_obligations(
    section,
    doc_cross_refs: list[str],
    re_engine,
    source: str | None = None,
    analysis_hints: dict[str, Any] | None = None,
) -> list[ObligationSchema]:
    """Extract obligations through candidate generation, voting, and validation."""
    if section.clause_type in SKIP_CLAUSE_TYPES:
        return []

    # Use the format-agnostic text_normalizer splitter
    units = split_into_obligation_units(section.text, section.clause_type)

    obligations: list[ObligationSchema] = []
    for index, unit in enumerate(units, 1):
        if len(unit.strip()) < ACTION_MIN_LENGTH:
            continue
        if is_quoted_content(unit, section.clause_type):
            continue
        if is_boilerplate(unit):
            continue

        bundle = generate_candidates(
            unit=unit,
            unit_id=f"{section.id}-U{index}",
            section_id=section.id,
            clause_number=section.id,
            clause_type=section.clause_type,
            re_engine=re_engine,
            source=source,
            analysis_hints=analysis_hints,
        )

        obligation = assemble_obligation(
            bundle=bundle,
            doc_cross_refs=doc_cross_refs,
            re_engine=re_engine,
        )
        if obligation:
            obligations.append(obligation)

    return deduplicate_atomic_obligations(obligations)


def generate_candidates(
    unit: str,
    unit_id: str,
    section_id: str,
    clause_number: str,
    clause_type: str,
    re_engine,
    source: str | None = None,
    analysis_hints: dict[str, Any] | None = None,
) -> CandidateBundle:
    """Generate specialized field candidates for one atomic clause unit."""
    candidates: list[FieldCandidate] = []
    trigger_rule = re_engine.find_trigger(unit)
    urgency_override = (trigger_rule or {}).get("urgency_override")
    deadline_dict = re_engine.extract_deadline(unit, urgency_override=urgency_override)
    xrefs = re_engine.extract_cross_references(unit)
    actor = correct_issuer_actor(re_engine.find_actor(unit, source=source), unit)
    domain, domain_match_count = re_engine.classify_domain(unit)
    sub_type = re_engine.find_obligation_sub_type(unit)
    sector_match = re_engine.find_sector_obligation(unit, actor)

    if sector_match:
        domain = sector_match.get("domain", domain)
        sub_type = sector_match.get("sub_type", sub_type)
        if sector_match.get("deadline"):
            sector_deadline = sector_match["deadline"]
            deadline_dict = {
                "text": sector_deadline.get("text", deadline_dict["text"]),
                "absolute_date": sector_deadline.get("absolute_date", deadline_dict["absolute_date"]),
                "duration": sector_deadline.get("duration", deadline_dict["duration"]),
                "urgency": sector_deadline.get("urgency", deadline_dict["urgency"]),
                "frequency": sector_deadline.get("frequency", deadline_dict.get("frequency")),
            }

    primary_actor_hint = ""
    if analysis_hints:
        primary_actor_hint = str(analysis_hints.get("primary_actor") or "").strip()
        if primary_actor_hint and actor in GENERIC_ACTORS.union({"Regulated Entity"}) and len(primary_actor_hint) > 2:
            actor = primary_actor_hint

    if trigger_rule:
        candidates.append(FieldCandidate(
            field="trigger",
            value=trigger_rule.get("matched_text", "always"),
            evidence=trigger_rule.get("matched_text", ""),
            source="RuleEngine",
            confidence=min(0.98, trigger_rule.get("weight", 0.75)),
            metadata={"rule_id": trigger_rule.get("id"), "raw_rule": trigger_rule},
        ))

    obligation_type = re_engine.classify_obligation_type(unit, trigger_rule)
    candidates.append(FieldCandidate(
        field="obligation_type",
        value=obligation_type,
        evidence=(trigger_rule or {}).get("matched_text", unit[:80]),
        source="RuleEngine",
        confidence=0.90 if trigger_rule else 0.55,
        metadata={"sub_type": sub_type},
    ))
    candidates.extend(_keyword_obligation_type_candidates(unit))

    candidates.append(FieldCandidate(
        field="actor",
        value=actor,
        evidence=_actor_evidence(unit, actor),
        source="RuleEngine",
        confidence=0.90 if actor not in GENERIC_ACTORS else 0.62,
    ))

    if deadline_dict["urgency"] != "ongoing":
        candidates.append(FieldCandidate(
            field="deadline",
            value=deadline_dict,
            evidence=deadline_dict["text"],
            source="PatternEngine",
            confidence=0.90 if deadline_dict.get("duration") or deadline_dict.get("absolute_date") else 0.78,
        ))

    if xrefs:
        candidates.append(FieldCandidate(
            field="cross_references",
            value=xrefs,
            evidence=", ".join(xrefs),
            source="PatternEngine",
            confidence=0.92,
        ))

    if domain not in DOMAIN_ALLOWED:
        domain = "Other"
    candidates.append(FieldCandidate(
        field="domain",
        value=domain,
        evidence=unit,
        source="KeywordEngine",
        confidence=min(0.90, 0.50 + 0.10 * domain_match_count),
        metadata={"match_count": domain_match_count},
    ))
    candidates.extend(_obligation_shape_domain_candidates(unit))

    departments = re_engine.get_departments(domain=domain, sub_type=sub_type, actor=actor, text=unit)
    candidates.append(FieldCandidate(
        field="departments",
        value=departments,
        evidence=unit,
        source="RuleEngine",
        confidence=0.84 if departments else 0.55,
    ))

    severity = re_engine.classify_severity(
        text=unit,
        urgency=deadline_dict["urgency"],
        obligation_type=obligation_type,
        sub_type=sub_type,
        domain=domain,
    )
    if sector_match:
        floor = sector_match.get("severity")
        if floor and _severity_rank(floor) > _severity_rank(severity):
            severity = floor
    candidates.append(FieldCandidate(
        field="severity",
        value=severity,
        evidence=unit,
        source="RuleEngine",
        confidence=0.82,
    ))

    penalty = extract_penalty(unit)
    fine = extract_fine(unit)
    if penalty:
        candidates.append(FieldCandidate("penalty", penalty, penalty, "PatternEngine", 0.88))
    if fine:
        candidates.append(FieldCandidate("fine", fine, str(fine), "PatternEngine", 0.90))

    action = extract_action(unit, trigger_rule)
    candidates.append(FieldCandidate(
        field="action",
        value=action,
        evidence=unit,
        source="PatternEngine",
        confidence=_action_confidence(action, unit),
    ))

    if analysis_hints:
        candidates.append(FieldCandidate(
            field="analysis_boost",
            value=_analysis_boost(action, unit, analysis_hints),
            evidence=action,
            source="LLM",
            confidence=0.70,
        ))

    candidates.extend(_similarity_candidates(unit, source))
    candidates.extend(_document_analysis_candidates(unit, actor, domain, clause_type, analysis_hints))

    return CandidateBundle(
        unit_id=unit_id,
        text=unit,
        section_id=section_id,
        clause_number=clause_number,
        clause_type=clause_type,
        source=source,
        candidates=candidates,
    )


def assemble_obligation(
    bundle: CandidateBundle,
    doc_cross_refs: list[str],
    re_engine,
) -> ObligationSchema | None:
    """Aggregate candidate votes and emit a final obligation if evidence supports it."""
    trigger_vote = aggregate_field(bundle.candidates, "trigger")
    deadline_vote = aggregate_field(bundle.candidates, "deadline")
    xref_vote = aggregate_field(bundle.candidates, "cross_references")

    trigger_rule = None
    if trigger_vote and trigger_vote.votes:
        trigger_rule = trigger_vote.votes[0].metadata.get("raw_rule")

    has_xref = bool(xref_vote and xref_vote.value)
    deadline_dict = deadline_vote.value if deadline_vote else {
        "text": "ongoing",
        "absolute_date": None,
        "duration": None,
        "urgency": "ongoing",
    }

    if trigger_rule is None and deadline_dict["urgency"] == "ongoing" and not has_xref:
        return None

    # For non-obligation clause types that slipped through (e.g. "other"),
    # only keep if there is a strong mandatory trigger (weight >= 0.9).
    # "obligation" and "permission" are always eligible — permission clauses
    # contain valid discretionary obligations ("may obtain approval").
    if bundle.clause_type not in OBLIGATION_CLAUSE_TYPES:
        has_strong = trigger_rule and trigger_rule.get("weight", 0.0) >= 0.9
        if not has_strong:
            return None

    actor_vote = aggregate_field(bundle.candidates, "actor")
    action_vote = aggregate_field(bundle.candidates, "action")
    otype_vote = aggregate_field(bundle.candidates, "obligation_type")
    domain_vote = aggregate_field(bundle.candidates, "domain")
    dept_vote = aggregate_field(bundle.candidates, "departments")
    severity_vote = aggregate_field(bundle.candidates, "severity")
    penalty_vote = aggregate_field(bundle.candidates, "penalty")
    fine_vote = aggregate_field(bundle.candidates, "fine")

    actor = actor_vote.value if actor_vote else "Regulated Entity"
    action = action_vote.value if action_vote else extract_action(bundle.text, trigger_rule)
    obligation_type = otype_vote.value if otype_vote else "mandatory"
    domain = domain_vote.value if domain_vote else "Other"
    departments = dept_vote.value if dept_vote else ["Compliance"]
    severity = severity_vote.value if severity_vote else "medium"
    penalty = penalty_vote.value if penalty_vote else None
    fine = fine_vote.value if fine_vote else None

    deadline = DeadlineSchema(
        text=deadline_dict["text"],
        absolute_date=deadline_dict.get("absolute_date"),
        duration=deadline_dict.get("duration"),
        urgency=deadline_dict["urgency"],
    )

    verbatim_ratio = SequenceMatcher(None, action.lower(), bundle.text[:len(action)].lower()).ratio()
    base_confidence = re_engine.compute_confidence(
        text=bundle.text,
        trigger_rule=trigger_rule,
        urgency=deadline.urgency,
        has_cross_ref=has_xref,
        actor=actor,
        domain_match_count=(domain_vote.votes[0].metadata.get("match_count", 0) if domain_vote and domain_vote.votes else 0),
        penalty_found=bool(penalty),
        fine_found=bool(fine),
        action_verbatim_ratio=verbatim_ratio,
        clause_type=bundle.clause_type,
    )

    conflicts = detect_contradictions(bundle.candidates)
    if actor_vote and not actor_vote.conflict:
        conflicts.pop("actor", None)
    if domain_vote and not domain_vote.conflict:
        conflicts.pop("domain", None)
    if dept_vote and not dept_vote.conflict:
        conflicts.pop("departments", None)
    if severity_vote and not severity_vote.conflict:
        conflicts.pop("severity", None)
    if deadline_vote and not deadline_vote.conflict:
        conflicts.pop("deadline", None)
    confidence = apply_candidate_penalties(
        base_confidence=base_confidence,
        bundle=bundle,
        action=action,
        actor_vote=actor_vote,
        domain_vote=domain_vote,
        deadline=deadline,
        conflicts=conflicts,
    )
    confidence = _apply_analysis_floor(confidence, bundle.text, action)

    notes = build_notes(bundle.text, obligation_type, trigger_rule, conflicts, verbatim_ratio)

    return ObligationSchema(
        id=f"{bundle.section_id}-OB",
        section_id=bundle.section_id,
        clause_number=bundle.clause_number,
        actor=actor,
        action=action,
        obligation_type=obligation_type,
        trigger=extract_trigger_text(bundle.text, trigger_rule),
        deadline=deadline,
        domain=domain,
        departments=departments,
        severity=severity,
        severity_reason=re_engine.severity_reason(severity, deadline.urgency, None),
        evidence_required=evidence_for(bundle.text, domain),
        penalty_if_missed=penalty,
        fine_exposure_inr=fine,
        cross_references=merge_cross_references(xref_vote.value if xref_vote else [], doc_cross_refs),
        confidence=confidence,
        notes=notes,
    )


def aggregate_field(candidates: list[FieldCandidate], field_name: str) -> VoteResult | None:
    """Aggregate engine votes for one field."""
    field_votes = [item for item in candidates if item.field == field_name]
    if not field_votes:
        return None

    weights = {
        "RuleEngine": 0.35,
        "PatternEngine": 0.25,
        "SimilarityEngine": 0.20,
        "KeywordEngine": 0.10,
        "LLM": 0.10,
    }
    if field_name in {"penalty", "fine"}:
        weights = {
            "RuleEngine": 0.50,
            "PatternEngine": 0.30,
            "SimilarityEngine": 0.15,
            "KeywordEngine": 0.00,
            "LLM": 0.05,
        }

    grouped: dict[str, list[FieldCandidate]] = {}
    raw_values: dict[str, Any] = {}
    for vote in field_votes:
        key = _hashable_vote_value(vote.value)
        grouped.setdefault(key, []).append(vote)
        raw_values[key] = vote.value

    scores = {
        key: sum(v.confidence * weights.get(v.source, 0.10) for v in votes)
        for key, votes in grouped.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_key, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - runner_up

    return VoteResult(
        value=raw_values[best_key],
        score=round(min(0.99, best_score), 4),
        margin=round(margin, 4),
        conflict=len(ranked) > 1 and margin < 0.15,
        votes=grouped[best_key],
    )


def detect_contradictions(candidates: list[FieldCandidate]) -> dict[str, list[str]]:
    """Detect field conflicts such as RuleEngine domain vs LLM/keyword domain."""
    conflicts: dict[str, list[str]] = {}
    for field_name in ("actor", "domain", "departments", "severity", "deadline"):
        votes = [item for item in candidates if item.field == field_name]
        values = {_hashable_vote_value(item.value) for item in votes}
        if len(values) > 1:
            conflicts[field_name] = [str(item.value) for item in votes]
    return conflicts


def apply_candidate_penalties(
    base_confidence: float,
    bundle: CandidateBundle,
    action: str,
    actor_vote: VoteResult | None,
    domain_vote: VoteResult | None,
    deadline: DeadlineSchema,
    conflicts: dict[str, list[str]],
) -> float:
    """Apply calibrated penalties for known extraction failure modes.

    Uses action_quality_score() from text_normalizer for a single, consistent
    quality signal instead of scattered length/verbatim checks.
    """
    penalty = 0.0

    # Action quality penalty
    quality, reason = action_quality_score(action, bundle.text)
    if quality <= 0.3:
        penalty += 0.25      # enumeration markers or too short
    elif quality <= 0.45:
        penalty += 0.15      # formula content
    elif quality == 0.5:
        penalty += 0.08      # verbatim_long — small penalty (not fatal)
    # quality >= 0.6: no action-quality penalty

    # Actor conflict
    if actor_vote and actor_vote.conflict:
        penalty += 0.18
    if "actor" in conflicts:
        penalty += 0.05

    # Domain conflict (small)
    if domain_vote and domain_vote.conflict:
        penalty += 0.05

    # Missing deadline when clause clearly specifies one
    lower = bundle.text.lower()
    requires_deadline = any(term in lower for term in REPORTING_TERMS + APPROVAL_TERMS)
    if requires_deadline and deadline.urgency == "ongoing" and re.search(r"\bwithin\b|\bby\b|not later than", lower):
        penalty += 0.12

    # Quoted / reference unit that slipped through
    if is_quoted_content(bundle.text, bundle.clause_type):
        penalty += 0.50

    return round(max(0.0, min(0.99, base_confidence - penalty)), 4)


def split_atomic_obligation_units(text: str) -> list[str]:
    """
    Thin delegator kept for backward-compat with any external callers.
    All logic has moved to text_normalizer.split_into_obligation_units.
    """
    return split_into_obligation_units(text)


# ── Kept for backward-compat (other modules may import these) ─────────────────

def is_quoted_or_reference_unit(text: str, clause_type: str = "other") -> bool:
    """Delegate to text_normalizer.is_quoted_content."""
    return is_quoted_content(text, clause_type)


def deduplicate_atomic_obligations(obligations: list[ObligationSchema]) -> list[ObligationSchema]:
    """Remove duplicates while preserving distinct atomic obligations."""
    result: list[ObligationSchema] = []
    for obligation in obligations:
        duplicate_index = None
        for idx, existing in enumerate(result):
            if existing.section_id != obligation.section_id:
                continue
            similarity = SequenceMatcher(None, existing.action.lower(), obligation.action.lower()).ratio()
            if similarity > 0.86 and existing.actor == obligation.actor:
                duplicate_index = idx
                break
        if duplicate_index is None:
            result.append(obligation)
        elif obligation.confidence > result[duplicate_index].confidence:
            obligation.notes = _append_note(obligation.notes, "[duplicate_replaced_lower_confidence]")
            result[duplicate_index] = obligation
    return result


def extract_action(sentence: str, trigger_rule: dict | None) -> str:
    """
    Delegate to text_normalizer.extract_action_phrase.
    trigger_rule is kept for API compat but the normalizer ignores it —
    we preserve the full predicate instead of chopping at the trigger verb.
    """
    return extract_action_phrase(sentence, trigger_rule)


def extract_trigger_text(sentence: str, trigger_rule: dict | None) -> str:
    conditional = re.search(
        r"(in case of [^.,;]+|where [^.,;]+|if [^.,;]+|subject to [^.,;]+|provided that [^.,;]+)",
        sentence,
        re.IGNORECASE,
    )
    if conditional:
        return conditional.group(1).strip()
    if trigger_rule:
        otype = trigger_rule.get("obligation_type", "")
        if otype in {"mandatory", "prohibition", "approval_required"}:
            return "always"
        return trigger_rule.get("matched_text", "always")
    return "always"


def extract_penalty(sentence: str) -> str | None:
    match = re.search(r"(penalty[^.;]*|fine[^.;]*|late submission fee[^.;]*)", sentence, re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_fine(sentence: str) -> float | None:
    match = re.search(r"(?:Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)", sentence, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def evidence_for(sentence: str, domain: str) -> list[str]:
    evidence_map = {
        "KYC_AML": ["Updated KYC/AML procedure", "Customer due diligence records"],
        "Cybersecurity": ["Incident response log", "Security control evidence"],
        "DataPrivacy": ["Data handling SOP", "Record retention evidence"],
        "FinancialInclusion": ["Branch service continuity note", "Customer communication"],
        "BusinessContinuity": ["BCP activation record", "Branch restoration tracker"],
        "FraudManagement": ["Fraud monitoring report", "Exception handling log"],
        "CapitalAdequacy": ["Capital computation sheet", "Regulatory reporting extract"],
        "Payments": ["ATM uptime report", "Alternate service arrangement log"],
        "CustomerService": ["Fee waiver approval", "Customer service advisory"],
        "Governance": ["Board note or approval", "Updated governance policy"],
        "ITInfrastructure": ["System configuration evidence", "Service deployment log"],
        "ReportingAudit": ["Submission acknowledgement", "Audit trail or return copy"],
        "HR_Training": ["Training completion records", "Updated staff guidance"],
        "FEMA": ["Regulatory filing", "Treasury compliance record"],
        "Other": ["Compliance confirmation", "Supporting internal record"],
    }
    evidence = list(evidence_map.get(domain, evidence_map["Other"]))
    lower = sentence.lower()
    if "approval" in lower:
        evidence.append("Approval or authorization record")
    if "intimation" in lower or "report" in lower or "submit" in lower:
        evidence.append("Regulatory communication or submission acknowledgement")
    return _unique(evidence)


def merge_cross_references(local_refs: list[str], doc_refs: list[str]) -> list[str]:
    merged = list(local_refs or [])
    for ref in doc_refs:
        if ref not in merged and any(term in ref.lower() for term in ["directions", "circular", "guidelines"]):
            merged.append(ref)
    return merged[:5]


def correct_issuer_actor(actor: str, sentence: str) -> str:
    if actor not in ("RBI", "IRDAI"):
        return actor
    issuer_signals = [
        "hereby issues",
        "in exercise of the powers",
        "being satisfied",
        "expedient in the public interest",
        "reserve bank hereby",
        "hereby directs",
        "hereby notifies",
        "the authority hereby",
        "in exercising its powers",
        "the authority has decided",
        "the authority directs",
    ]
    lower = sentence.lower()
    if re.search(r"\bto\s+the\s+authority\b|\bto\s+rbi\b|\bto\s+the\s+reserve\s+bank\b", lower):
        return "Regulated Entity"
    if any(signal in lower for signal in issuer_signals):
        return "Regulated Entity"
    return actor


def build_notes(
    sentence: str,
    obligation_type: str,
    trigger_rule: dict | None,
    conflicts: dict[str, list[str]],
    verbatim_ratio: float,
) -> str | None:
    notes: list[str] = []
    if obligation_type == "conditional":
        notes.append("This obligation applies only when the stated condition is met.")
    if trigger_rule and trigger_rule.get("obligation_type") == "prohibition":
        notes.append("This is a prohibition; the actor must not perform the stated action.")
    lower = sentence.lower()
    if "may" in lower or "at its discretion" in lower:
        notes.append("The clause appears discretionary rather than strictly mandatory.")
    # Use normalizer quality check — only flag truly long verbatim blobs
    from backend.ingestion.text_normalizer import action_quality_score
    action_approx = sentence[:200]
    quality, reason = action_quality_score(action_approx, sentence)
    if reason == "verbatim_long":
        notes.append("[action_quality: verbatim - needs review]")
    for field_name in sorted(conflicts):
        unique_values = []
        for value in conflicts[field_name]:
            if value not in unique_values:
                unique_values.append(value)
        notes.append(f"[{field_name}_conflict: {', '.join(unique_values)}]")
    return " ".join(notes) or None


def contains_enumeration_markers(text: str) -> bool:
    return bool(re.search(r"(?:^|\s)(?:\([a-z]\)|\([ivxlcdm]+\)|[a-z]\)|[ivxlcdm]+\))\s+", text, re.IGNORECASE))


def _keyword_obligation_type_candidates(text: str) -> list[FieldCandidate]:
    lower = text.lower()
    candidates: list[FieldCandidate] = []
    if any(term in lower for term in PROHIBITION_TERMS):
        candidates.append(FieldCandidate("obligation_type", "mandatory", "prohibition phrase", "KeywordEngine", 0.82))
    elif any(term in lower for term in APPROVAL_TERMS):
        candidates.append(FieldCandidate("obligation_type", "mandatory", "approval phrase", "KeywordEngine", 0.78))
    elif any(term in lower for term in REPORTING_TERMS):
        candidates.append(FieldCandidate("obligation_type", "mandatory", "reporting phrase", "KeywordEngine", 0.74))
    elif re.search(r"\bmay\b|at its discretion", lower):
        candidates.append(FieldCandidate("obligation_type", "discretionary", "may/discretion phrase", "KeywordEngine", 0.80))
    return candidates


def _obligation_shape_domain_candidates(text: str) -> list[FieldCandidate]:
    lower = text.lower()
    candidates: list[FieldCandidate] = []
    if any(term in lower for term in REPORTING_TERMS):
        candidates.append(FieldCandidate("domain", "ReportingAudit", "reporting phrase", "PatternEngine", 0.72))
    if any(term in lower for term in APPROVAL_TERMS) or "board" in lower:
        candidates.append(FieldCandidate("domain", "Governance", "approval/governance phrase", "PatternEngine", 0.68))
    return candidates


def _similarity_candidates(text: str, regulator: str | None) -> list[FieldCandidate]:
    """Create low-burden votes from approved historical pattern retrieval."""
    candidates: list[FieldCandidate] = []
    for pattern in retrieve_regulatory_patterns(text, regulator=regulator, top_k=4):
        score = float(pattern.get("similarity_score", 0.0))
        confidence = min(0.88, max(0.40, score))
        normalized = pattern.get("normalized_value")
        pattern_type = pattern.get("pattern_type")

        if pattern.get("domain") in DOMAIN_ALLOWED:
            candidates.append(FieldCandidate(
                "domain",
                pattern["domain"],
                pattern.get("pattern_text", ""),
                "SimilarityEngine",
                confidence,
                metadata={"pattern_id": pattern.get("pattern_id")},
            ))
        if pattern_type in {"reporting_pattern", "approval_pattern", "disclosure_pattern", "prohibition_pattern"}:
            candidates.append(FieldCandidate(
                "obligation_type",
                "mandatory",
                pattern.get("pattern_text", ""),
                "SimilarityEngine",
                confidence,
                metadata={"pattern_id": pattern.get("pattern_id"), "normalized_value": normalized},
            ))
        if pattern_type == "deadline_phrase":
            candidates.append(FieldCandidate(
                "deadline_hint",
                normalized,
                pattern.get("pattern_text", ""),
                "SimilarityEngine",
                confidence,
                metadata={"pattern_id": pattern.get("pattern_id")},
            ))
    return candidates


def _document_analysis_candidates(
    text: str,
    current_actor: str,
    current_domain: str,
    clause_type: str,
    analysis_hints: dict[str, Any] | None,
) -> list[FieldCandidate]:
    """Turn document-level LLM analysis into bounded local candidate hints."""
    if not analysis_hints:
        return []

    candidates: list[FieldCandidate] = []
    primary_actor = str(analysis_hints.get("primary_actor") or "").strip()
    dominant_domains = analysis_hints.get("dominant_domains") or []
    core_phrases = analysis_hints.get("core_obligation_phrases") or []
    contains_enum_ops = bool(analysis_hints.get("contains_enumerated_operational_clause"))

    if primary_actor and (current_actor in GENERIC_ACTORS or current_actor in {"RBI", "IRDAI"}):
        candidates.append(FieldCandidate(
            "actor",
            primary_actor,
            primary_actor,
            "LLM",
            0.68,
        ))

    for domain in dominant_domains:
        if domain in DOMAIN_ALLOWED and current_domain == "Other":
            candidates.append(FieldCandidate(
                "domain",
                domain,
                str(domain),
                "LLM",
                0.62,
            ))

    if clause_type == "obligation" and contains_enum_ops and not re.search(
        r"\b(?:shall|must|may|is required to|are required to|shall not|must not)\b",
        text,
        re.IGNORECASE,
    ):
        candidates.append(FieldCandidate(
            "obligation_type",
            "mandatory",
            "document-level enumerated operational clause",
            "LLM",
            0.60,
        ))

    lower = text.lower()
    for phrase in core_phrases:
        phrase = str(phrase).strip()
        if not phrase:
            continue
        if phrase.lower() in lower:
            candidates.append(FieldCandidate(
                "obligation_type",
                "mandatory",
                phrase,
                "LLM",
                0.64,
            ))
            break

    return candidates


def _actor_evidence(text: str, actor: str) -> str:
    if actor in GENERIC_ACTORS:
        return ""
    match = re.search(re.escape(actor), text, re.IGNORECASE)
    return match.group(0) if match else ""


def _action_confidence(action: str, source_text: str) -> float:
    """Use action_quality_score from text_normalizer for a consistent signal."""
    quality, _ = action_quality_score(action, source_text)
    # Map quality 0..1 to a confidence contribution 0.55..0.90
    return round(max(0.40, min(0.90, 0.55 + quality * 0.35)), 4)


def _hashable_vote_value(value: Any) -> str:
    if isinstance(value, dict):
        return repr(sorted(value.items()))
    if isinstance(value, list):
        return repr(value)
    return str(value)


def _append_note(notes: str | None, addition: str) -> str:
    return f"{notes} {addition}".strip() if notes else addition


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _analysis_boost(action: str, unit: str, analysis_hints: dict[str, Any] | None) -> float:
    """Small confidence lift for analysis-confirmed operational clauses."""
    if not analysis_hints:
        return 0.0

    lower = f"{action} {unit}".lower()
    boost = 0.0

    core_phrases = [str(item).lower() for item in (analysis_hints.get("core_obligation_phrases") or [])]
    if any(phrase and phrase in lower for phrase in core_phrases):
        boost += 0.12

    if any(marker in lower for marker in ["a)", "b)", "c)", "d)", "e)", "(i)", "(ii)", "(iii)"]):
        boost += 0.08

    if re.search(r"\bwithin\s+\d+\s+days\b|\bshall\s+report\b|\bshall\s+disclose\b|\bstatement\s+shall\s+be\s+included\b", lower):
        boost += 0.08

    if any(term in lower for term in [
        "consistent methodology",
        "trued up",
        "reported to the authority",
        "annual report",
    ]):
        boost += 0.10

    if "complete disclosure" in lower:
        boost += 0.36

    if "ensure that" in lower and boost > 0.0:
        boost += 0.04

    return round(min(0.18, boost), 4)


def _apply_analysis_floor(confidence: float, text: str, action: str) -> float:
    """Raise confidence floor for analysis-confirmed operational children."""
    lower = f"{text} {action}".lower()
    if any(term in lower for term in ["consistent methodology", "trued up", "complete disclosure", "reported to the authority"]):
        confidence = max(confidence, 0.72)
    if "statement shall be included" in lower or "annual report" in lower:
        confidence = max(confidence, 0.74)
    return round(min(0.94, confidence), 4)


def _severity_rank(severity: str) -> int:
    return {"critical": 3, "high": 2, "medium": 1, "low": 0}.get(severity, 0)
