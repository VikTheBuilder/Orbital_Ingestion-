"""
Verification script for globally unique obligation IDs.
Tests that:
  1. Rule engine obligations get section-prefixed IDs
  2. LLM obligations always get section-prefixed IDs (not bare numbers)
  3. After the final re-numbering pass, all IDs are unique and section-prefixed
  4. Schemas allow any string ID (no format constraint)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.schemas import ObligationSchema, DeadlineSchema


def test_unique_ids():
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

    # ── Test 1: Schema accepts any string ID ──
    print("\n=== Test 1: Schema accepts any string ID format ===")
    try:
        ob = ObligationSchema(
            id="4-OB1",
            section_id="4",
            clause_number="4",
            actor="Insurer",
            action="Test action.",
            obligation_type="mandatory",
            trigger="always",
            deadline=DeadlineSchema(text="always", urgency="ongoing"),
            domain="Other",
            departments=["Compliance"],
            severity="medium",
            severity_reason="Test.",
            confidence=0.8,
        )
        check("Schema accepts '4-OB1' format", ob.id, "4-OB1")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] Schema rejected '4-OB1': {e}")

    try:
        ob2 = ObligationSchema(
            id="Section 1-OB3",
            section_id="Section 1",
            clause_number="Section 1",
            actor="Insurer",
            action="Test action.",
            obligation_type="mandatory",
            trigger="always",
            deadline=DeadlineSchema(text="always", urgency="ongoing"),
            domain="Other",
            departments=["Compliance"],
            severity="medium",
            severity_reason="Test.",
            confidence=0.8,
        )
        check("Schema accepts 'Section 1-OB3' format", ob2.id, "Section 1-OB3")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] Schema rejected 'Section 1-OB3': {e}")

    # ── Test 2: _coerce_llm_obligation always uses section prefix ──
    print("\n=== Test 2: LLM obligation IDs are section-prefixed ===")

    from backend.ingestion.obligation_extractor import _coerce_llm_obligation
    from backend.ingestion.schemas import SectionSchema

    section = SectionSchema(
        id="4", heading="Test", text="Test text.",
        clause_type="obligation", level=1,
    )

    # LLM returns bare ID "1"
    llm_item = {
        "id": "1",
        "action": "Submit quarterly compliance report.",
        "actor": "Regulated Entity",
        "obligation_type": "mandatory",
        "trigger": "always",
        "deadline": {"text": "quarterly", "urgency": "ongoing"},
        "domain": "ReportingAudit",
        "departments": ["Compliance"],
        "severity": "medium",
        "severity_reason": "Periodic reporting.",
        "confidence": 0.75,
    }

    from backend.core.rule_engine import RuleEngine
    re_engine = RuleEngine()

    coerced = _coerce_llm_obligation(llm_item, section, index=1, re_engine=re_engine)
    check(
        "LLM bare id '1' becomes section-prefixed",
        coerced.id,
        "4-L1",  # Always uses section.id-L{index}, ignoring LLM's "id" field
    )

    # LLM returns no ID
    llm_item_no_id = dict(llm_item)
    del llm_item_no_id["id"]
    coerced2 = _coerce_llm_obligation(llm_item_no_id, section, index=2, re_engine=re_engine)
    check(
        "LLM with no id gets section-prefixed id",
        coerced2.id,
        "4-L2",
    )

    # ── Test 3: Final re-numbering produces unique IDs ──
    print("\n=== Test 3: Simulate final re-numbering pass ===")

    obligations = []
    for section_id in ["4", "5", "4"]:  # note: two obligations from section 4
        obligations.append(ObligationSchema(
            id="temp",
            section_id=section_id,
            clause_number=section_id,
            actor="Insurer",
            action=f"Action for section {section_id}.",
            obligation_type="mandatory",
            trigger="always",
            deadline=DeadlineSchema(text="always", urgency="ongoing"),
            domain="Other",
            departments=["Compliance"],
            severity="medium",
            severity_reason="Test.",
            confidence=0.8,
        ))

    # Simulate the re-numbering pass from extract_obligations()
    for seq, ob in enumerate(obligations, 1):
        ob.id = f"{ob.section_id}-OB{seq}"

    ids = [ob.id for ob in obligations]
    check("All IDs unique", len(ids), len(set(ids)))
    check("First ID is section-prefixed", ids[0], "4-OB1")
    check("Second ID is section-prefixed", ids[1], "5-OB2")
    check("Third ID is section-prefixed (no collision)", ids[2], "4-OB3")

    # Verify no bare numeric IDs
    for ob_id in ids:
        if ob_id.isdigit():
            failed += 1
            print(f"  [FAIL] Found bare numeric ID: {ob_id}")
            break
    else:
        passed += 1
        print(f"  [PASS] No bare numeric IDs found")

    # ── Test 4: Validator IncorrectExtractionSchema works with new IDs ──
    print("\n=== Test 4: Validator schema works with new ID format ===")

    from backend.ingestion.schemas import IncorrectExtractionSchema
    try:
        finding = IncorrectExtractionSchema(
            obligation_id="4-OB1",
            field="action",
            current_value="old action",
            correct_value="new action",
            reason="Test reason.",
        )
        check("IncorrectExtractionSchema accepts '4-OB1'", finding.obligation_id, "4-OB1")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] IncorrectExtractionSchema rejected '4-OB1': {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = test_unique_ids()
    sys.exit(0 if success else 1)
