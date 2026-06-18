"""
Unit test for obligation dedup + merge.

Tests that a synthetic multi-part clause (a, b, c, d, e structure)
produces ONE merged obligation instead of N near-duplicate records.
Also verifies actor/department reconciliation and conflict flagging.
"""

import sys
import os

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


def test_merge():
    passed = 0
    failed = 0

    def check(description, actual, expected):
        nonlocal passed, failed
        if actual == expected:
            passed += 1
            print(f"  [PASS] {description} -> {actual}")
        else:
            failed += 1
            print(f"  [FAIL] {description}")
            print(f"         Expected: {expected}")
            print(f"         Actual:   {actual}")

    def check_contains(description, text, substring):
        nonlocal passed, failed
        if substring in text:
            passed += 1
            print(f"  [PASS] {description}")
        else:
            failed += 1
            print(f"  [FAIL] {description}")
            print(f"         Expected substring: {substring}")
            print(f"         In text: {text[:200]}")

    # ──────────────────────────────────────────────────────────────────────
    # Test 1: Synthetic multi-part clause (a, b, c, d, e)
    # Same section_id, same underlying directive, different sub-actions
    # ──────────────────────────────────────────────────────────────────────
    print("\n=== Test 1: Multi-part clause merges into ONE obligation ===")

    obligations = [
        # Rule engine extractions (specific actor, higher confidence)
        _make_obligation(
            "4-OB1", "4", "FRBs/Reinsurers",
            "Ensure no premium is accrued on estimate basis up to 3rd quarter.",
            confidence=0.96,
        ),
        _make_obligation(
            "4-OB2", "4", "FRBs/Reinsurers",
            "Follow a consistent methodology across the entire portfolio.",
            confidence=0.81,
        ),
        _make_obligation(
            "4-OB3", "4", "FRBs/Reinsurers",
            "True up estimates as actual values emerge.",
            confidence=0.81,
        ),
        _make_obligation(
            "4-OB4", "4", "FRBs/Reinsurers",
            "Include a statement in the annual report stating total premium and claims.",
            confidence=0.80,
            departments=["Compliance", "Finance", "InternalAudit"],
        ),
        # LLM extractions (generic actor, lower confidence, overlapping)
        _make_obligation(
            "4-L1", "4", "Regulated Entity",
            "Ensure no premium is accrued on estimate basis up to 3rd quarter of each financial year.",
            confidence=0.75,
            departments=["Compliance", "Legal"],
        ),
        _make_obligation(
            "4-L2", "4", "Regulated Entity",
            "Report any deviation beyond 10% to the Authority within 15 days.",
            confidence=0.80,
            departments=["Compliance", "InternalAudit"],
            deadline_urgency="short_term",
            deadline_duration="15 days",
        ),
    ]

    result = _deduplicate_and_merge(obligations)

    check("Number of output obligations", len(result), 1)

    if len(result) >= 1:
        merged = result[0]

        check("Merged actor is the specific one", merged.actor, "FRBs/Reinsurers")

        check_contains(
            "Action contains 'Sub-requirements'",
            merged.action, "Sub-requirements"
        )
        check_contains(
            "Action contains sub-action about methodology",
            merged.action, "consistent methodology"
        )
        check_contains(
            "Action contains sub-action about true-up",
            merged.action, "True up estimates"
        )
        check_contains(
            "Action contains sub-action about reporting deviation",
            merged.action, "deviation beyond 10%"
        )
        check_contains(
            "Notes flag actor conflict",
            merged.notes or "", "actor_conflict"
        )
        check_contains(
            "Notes flag merge count",
            merged.notes or "", "merged"
        )
        check_contains(
            "Notes mention 'Regulated Entity' as conflicting actor",
            merged.notes or "", "Regulated Entity"
        )

        # Departments should be a union
        check(
            "InternalAudit in merged departments",
            "InternalAudit" in merged.departments, True,
        )
        # Note: 4-L1 ("Regulated Entity", ["Compliance", "Legal"]) was deduped
        # away in Phase 1 because its action is >75% similar to 4-OB1, so its
        # "Legal" department is correctly NOT in the merged result.
        check(
            "Legal NOT in merged (source obligation deduped in Phase 1)",
            "Legal" not in merged.departments, True,
        )
        check(
            "Finance in merged departments",
            "Finance" in merged.departments, True,
        )

        # Deadline should prefer the specific one (15 days, short_term)
        check(
            "Best deadline picked (short_term, not ongoing)",
            merged.deadline.urgency, "short_term",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Test 2: Exact-ish duplicates are removed even with different actors
    # ──────────────────────────────────────────────────────────────────────
    print("\n=== Test 2: Exact-ish duplicates removed across actors ===")

    obligations2 = [
        _make_obligation(
            "5-OB1", "5", "FRBs/Reinsurers",
            "Submit the quarterly compliance report to IRDAI.",
            confidence=0.90,
        ),
        _make_obligation(
            "5-L1", "5", "Regulated Entity",
            "Submit the quarterly compliance report to IRDAI.",
            confidence=0.75,
        ),
    ]

    result2 = _deduplicate_and_merge(obligations2)
    check("Exact duplicate removed, 2 -> 1", len(result2), 1)
    if result2:
        check("Kept the higher-confidence one", result2[0].actor, "FRBs/Reinsurers")

    # ──────────────────────────────────────────────────────────────────────
    # Test 3: Obligations from different sections are NOT merged
    # ──────────────────────────────────────────────────────────────────────
    print("\n=== Test 3: Different sections stay separate ===")

    obligations3 = [
        _make_obligation("4-OB1", "4", "Insurer", "Ensure premium recognition.", confidence=0.90),
        _make_obligation("5-OB1", "5", "Insurer", "Submit compliance report.", confidence=0.85),
        _make_obligation("6-OB1", "6", "Reinsurer", "Disclose estimation methodology.", confidence=0.80),
    ]

    result3 = _deduplicate_and_merge(obligations3)
    check("3 sections -> 3 obligations", len(result3), 3)

    # ──────────────────────────────────────────────────────────────────────
    # Test 4: 2 obligations per section (below merge threshold) stay separate
    # ──────────────────────────────────────────────────────────────────────
    print("\n=== Test 4: 2-obligation sections are NOT merged ===")

    obligations4 = [
        _make_obligation("7-OB1", "7", "Insurer", "File annual return.", confidence=0.90),
        _make_obligation("7-OB2", "7", "Insurer", "Appoint compliance officer.", confidence=0.85),
    ]

    result4 = _deduplicate_and_merge(obligations4)
    check("2 distinct obligations stay separate", len(result4), 2)

    # ──────────────────────────────────────────────────────────────────────
    # Test 5: Actor reconciliation prefers specific over generic
    # ──────────────────────────────────────────────────────────────────────
    print("\n=== Test 5: Actor reconciliation ===")

    ranked = [
        _make_obligation("x1", "8", "Regulated Entity", "Action A.", confidence=0.90),
        _make_obligation("x2", "8", "FRBs/Reinsurers", "Action B.", confidence=0.85),
        _make_obligation("x3", "8", "Regulated Entity", "Action C.", confidence=0.80),
    ]

    actor, depts, conflict = _reconcile_actor_departments(ranked)
    check(
        "Specific actor preferred over generic",
        actor, "FRBs/Reinsurers",
    )
    check_contains(
        "Conflict note mentions 'Regulated Entity'",
        conflict, "Regulated Entity",
    )

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = test_merge()
    sys.exit(0 if success else 1)
