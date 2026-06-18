"""
Unit test for obligation dedup + merge.

Tests that a synthetic multi-part clause (a, b, c, d, e structure)
produces ONE merged obligation instead of N near-duplicate records.
Also verifies actor/department reconciliation and conflict flagging.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.obligation_extractor import (
    _deduplicate_and_merge,
    _merge_section_group,
    _reconcile_actor_departments,
)
from backend.ingestion.schemas import ObligationSchema, DeadlineSchema


def _make_obligation(
    id: str,
    section_id: str,
    actor: str,
    action: str,
    confidence: float = 0.80,
    departments: list = None,
    severity: str = "high",
    deadline_urgency: str = "ongoing",
    deadline_duration: str = None,
    penalty: str = None,
    notes: str = None,
) -> ObligationSchema:
    return ObligationSchema(
        id=id,
        section_id=section_id,
        clause_number=section_id,
        actor=actor,
        action=action,
        obligation_type="mandatory",
        trigger="always",
        deadline=DeadlineSchema(
            text="always" if deadline_urgency == "ongoing" else "within deadline",
            urgency=deadline_urgency,
            duration=deadline_duration,
        ),
        domain="ReportingAudit",
        departments=departments or ["Compliance", "Finance"],
        severity=severity,
        severity_reason="Test severity.",
        evidence_required=["Compliance confirmation"],
        penalty_if_missed=penalty,
        fine_exposure_inr=None,
        cross_references=[],
        confidence=confidence,
        notes=notes,
    )

def test_multipart_clause_merges_into_one_obligation():
    obligations = [
        # Rule engine extractions (specific actor, higher confidence)
        _make_obligation("4-OB1", "4", "FRBs/Reinsurers", "Ensure no premium is accrued on estimate basis up to 3rd quarter.", confidence=0.96),
        _make_obligation("4-OB2", "4", "FRBs/Reinsurers", "Follow a consistent methodology across the entire portfolio.", confidence=0.81),
        _make_obligation("4-OB3", "4", "FRBs/Reinsurers", "True up estimates as actual values emerge.", confidence=0.81),
        _make_obligation("4-OB4", "4", "FRBs/Reinsurers", "Include a statement in the annual report stating total premium and claims.", confidence=0.80, departments=["Compliance", "Finance", "InternalAudit"]),
        # LLM extractions (generic actor, lower confidence, overlapping)
        _make_obligation("4-L1", "4", "Regulated Entity", "Ensure no premium is accrued on estimate basis up to 3rd quarter of each financial year.", confidence=0.75, departments=["Compliance", "Legal"]),
        _make_obligation("4-L2", "4", "Regulated Entity", "Report any deviation beyond 10% to the Authority within 15 days.", confidence=0.80, departments=["Compliance", "InternalAudit"], deadline_urgency="short_term", deadline_duration="15 days"),
    ]

    result = _deduplicate_and_merge(obligations)

    assert len(result) == 1, "Number of output obligations should be 1"
    
    merged = result[0]
    assert merged.actor == "FRBs/Reinsurers", "Merged actor is the specific one"
    assert "Sub-requirements" in merged.action, "Action contains 'Sub-requirements'"
    assert "consistent methodology" in merged.action, "Action contains sub-action about methodology"
    assert "True up estimates" in merged.action, "Action contains sub-action about true-up"
    assert "deviation beyond 10%" in merged.action, "Action contains sub-action about reporting deviation"
    assert "actor_conflict" in (merged.notes or ""), "Notes flag actor conflict"
    assert "merged" in (merged.notes or ""), "Notes flag merge count"
    assert "Regulated Entity" in (merged.notes or ""), "Notes mention 'Regulated Entity' as conflicting actor"
    
    assert "InternalAudit" in merged.departments, "InternalAudit in merged departments"
    assert "Legal" not in merged.departments, "Legal NOT in merged (source obligation deduped in Phase 1)"
    assert "Finance" in merged.departments, "Finance in merged departments"
    
    assert merged.deadline.urgency == "short_term", "Best deadline picked (short_term, not ongoing)"

def test_exactish_duplicates_removed_across_actors():
    obligations = [
        _make_obligation("5-OB1", "5", "FRBs/Reinsurers", "Submit the quarterly compliance report to IRDAI.", confidence=0.90),
        _make_obligation("5-L1", "5", "Regulated Entity", "Submit the quarterly compliance report to IRDAI.", confidence=0.75),
    ]

    result = _deduplicate_and_merge(obligations)
    assert len(result) == 1, "Exact duplicate removed, 2 -> 1"
    assert result[0].actor == "FRBs/Reinsurers", "Kept the higher-confidence one"

def test_different_sections_stay_separate():
    obligations = [
        _make_obligation("4-OB1", "4", "Insurer", "Ensure premium recognition.", confidence=0.90),
        _make_obligation("5-OB1", "5", "Insurer", "Submit compliance report.", confidence=0.85),
        _make_obligation("6-OB1", "6", "Reinsurer", "Disclose estimation methodology.", confidence=0.80),
    ]

    result = _deduplicate_and_merge(obligations)
    assert len(result) == 3, "3 sections -> 3 obligations"

def test_two_obligation_sections_not_merged():
    obligations = [
        _make_obligation("7-OB1", "7", "Insurer", "File annual return.", confidence=0.90),
        _make_obligation("7-OB2", "7", "Insurer", "Appoint compliance officer.", confidence=0.85),
    ]

    result = _deduplicate_and_merge(obligations)
    assert len(result) == 2, "2 distinct obligations stay separate"

def test_actor_reconciliation_prefers_specific_over_generic():
    ranked = [
        _make_obligation("x1", "8", "Regulated Entity", "Action A.", confidence=0.90),
        _make_obligation("x2", "8", "FRBs/Reinsurers", "Action B.", confidence=0.85),
        _make_obligation("x3", "8", "Regulated Entity", "Action C.", confidence=0.80),
    ]

    actor, depts, conflict = _reconcile_actor_departments(ranked)
    assert actor == "FRBs/Reinsurers", "Specific actor preferred over generic"
    assert "Regulated Entity" in conflict, "Conflict note mentions 'Regulated Entity'"

def test_mixed_obligation_types_merge():
    # Synthetic IRDAI section-4 case
    obligations = [
        # Primary is mandatory
        _make_obligation("4-OB1", "4", "FRBs/Reinsurers", "Ensure no premium is accrued on estimate basis.", confidence=0.95),
        
        # Sub-point a (mandatory)
        _make_obligation("4-OB2", "4", "FRBs/Reinsurers", "Follow consistent methodology.", confidence=0.85),
        
        # Q4 carve-out (discretionary)
        _make_obligation("4-OB3", "4", "FRBs/Reinsurers", "For the fourth quarter the premium may be accounted on estimation basis.", confidence=0.80),
        
        # Sub-point b (mandatory)
        _make_obligation("4-OB4", "4", "FRBs/Reinsurers", "True up estimates as actual values emerge.", confidence=0.80),
    ]
    
    # Manually override the types since _make_obligation defaults to mandatory
    obligations[0].obligation_type = "mandatory"
    obligations[1].obligation_type = "mandatory"
    obligations[2].obligation_type = "discretionary"
    obligations[3].obligation_type = "mandatory"

    result = _deduplicate_and_merge(obligations)
    assert len(result) == 1, "Should merge into 1 obligation"
    
    merged = result[0]
    
    assert merged.obligation_type == "mandatory", "Merged obligation_type should be the most restrictive (mandatory)"
    
    # Check that the notes explicitly identify the mixed types
    assert "mixed_obligation_types" in merged.notes, "Notes should flag mixed_obligation_types"
    assert '"mandatory": ["primary", "(a)", "(c)"]' in merged.notes, "Notes should list mandatory sub-actions correctly"
    assert '"discretionary": ["(b)"]' in merged.notes, "Notes should list discretionary sub-actions correctly"

