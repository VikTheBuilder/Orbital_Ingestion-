"""
Verification script for effective date validation fix.
Tests that if a document already has its effective_date populated,
the validator does not flag it as missing or create spurious missed obligations.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.validator import validate_extraction
from backend.ingestion.schemas import ObligationSchema, DeadlineSchema

def test_populated_doc_effective_date_suppresses_warnings():
    raw_text = "1. This circular shall come into force on April 1, 2024.\n2. The bank shall submit the report."
    
    # Simulate an extracted obligation that doesn't explicitly capture the effective date in its deadline
    obligations = [
        ObligationSchema(
            id="2-OB1",
            section_id="2",
            clause_number="2",
            actor="Bank",
            action="submit the report",
            obligation_type="mandatory",
            trigger="always",
            deadline=DeadlineSchema(text="ongoing", urgency="ongoing"),
            domain="ReportingAudit",
            departments=["Compliance"],
            severity="medium",
            severity_reason="Standard reporting",
            confidence=0.9
        )
    ]
    
    # Case 1: doc_effective_date is NOT populated (reproduce bug behavior)
    result_missing = validate_extraction(raw_text, obligations, doc_effective_date=None)
    
    assert result_missing.missing_effective_date == "2024-04-01", "Should detect missing effective date when not provided by doc_structure"
    assert any("come into force" in m.raw_text.lower() for m in result_missing.missed_obligations), "Should mistakenly flag effective date sentence as missed obligation"
    
    # Case 2: doc_effective_date IS populated (verify fix)
    result_fixed = validate_extraction(raw_text, obligations, doc_effective_date="2024-04-01")
    
    assert result_fixed.missing_effective_date is None, "Should suppress missing effective date warning"
    assert not any("come into force" in m.raw_text.lower() for m in result_fixed.missed_obligations), "Should strip effective date false positive from missed obligations"
