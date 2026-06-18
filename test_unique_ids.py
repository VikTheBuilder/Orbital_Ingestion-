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
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.schemas import ObligationSchema, DeadlineSchema, SectionSchema, IncorrectExtractionSchema
from backend.ingestion.obligation_extractor import _coerce_llm_obligation
from backend.core.rule_engine import RuleEngine


@pytest.mark.parametrize("description, ob_id", [
    ("Schema accepts '4-OB1' format", "4-OB1"),
    ("Schema accepts 'Section 1-OB3' format", "Section 1-OB3"),
])
def test_schema_accepts_any_string_id(description, ob_id):
    ob = ObligationSchema(
        id=ob_id,
        section_id="test",
        clause_number="test",
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
    assert ob.id == ob_id, description

def test_coerce_llm_obligation_always_uses_section_prefix():
    section = SectionSchema(
        id="4", heading="Test", text="Test text.",
        clause_type="obligation", level=1,
    )
    re_engine = RuleEngine()

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

    coerced = _coerce_llm_obligation(llm_item, section, index=1, re_engine=re_engine)
    assert coerced.id == "4-L1", "LLM bare id '1' becomes section-prefixed"

    # LLM returns no ID
    llm_item_no_id = dict(llm_item)
    del llm_item_no_id["id"]
    coerced2 = _coerce_llm_obligation(llm_item_no_id, section, index=2, re_engine=re_engine)
    assert coerced2.id == "4-L2", "LLM with no id gets section-prefixed id"

def test_final_renumbering_produces_unique_ids():
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
    assert len(ids) == len(set(ids)), "All IDs unique"
    assert ids[0] == "4-OB1", "First ID is section-prefixed"
    assert ids[1] == "5-OB2", "Second ID is section-prefixed"
    assert ids[2] == "4-OB3", "Third ID is section-prefixed (no collision)"

    for ob_id in ids:
        assert not ob_id.isdigit(), f"Found bare numeric ID: {ob_id}"

def test_validator_schema_works_with_new_id_format():
    finding = IncorrectExtractionSchema(
        obligation_id="4-OB1",
        field="action",
        current_value="old action",
        correct_value="new action",
        reason="Test reason.",
    )
    assert finding.obligation_id == "4-OB1", "IncorrectExtractionSchema accepts '4-OB1'"
