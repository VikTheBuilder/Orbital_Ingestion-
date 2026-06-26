import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.schemas import DeadlineSchema, ObligationSchema
from backend.ingestion.validator import _validate_with_heuristics


class ValidatorCandidateSignalTests(unittest.TestCase):
    def test_candidate_conflict_notes_become_review_findings(self):
        obligation = ObligationSchema(
            id="4-OB1",
            section_id="4",
            clause_number="4",
            actor="Regulated Entity",
            action="Maintain records for five years.",
            obligation_type="mandatory",
            trigger="always",
            deadline=DeadlineSchema(text="ongoing", urgency="ongoing"),
            domain="ReportingAudit",
            departments=["Compliance"],
            severity="medium",
            severity_reason="Test severity.",
            evidence_required=["Compliance confirmation"],
            cross_references=[],
            confidence=0.65,
            notes="[domain_conflict: Governance, ReportingAudit] [action_quality: verbatim - needs review]",
        )

        result = _validate_with_heuristics(
            raw_text="4. Regulated entities shall maintain records for five years.",
            obligations=[obligation],
        )

        fields = {item.field for item in result.incorrect_extractions}
        self.assertIn("domain", fields)
        self.assertIn("action", fields)


if __name__ == "__main__":
    unittest.main()
