import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.rule_engine import RuleEngine
from backend.ingestion.candidate_engine import (
    deduplicate_atomic_obligations,
    extract_candidate_obligations,
    is_quoted_or_reference_unit,
    split_atomic_obligation_units,
)
from backend.ingestion.schemas import DeadlineSchema, ObligationSchema, SectionSchema


class CandidateEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RuleEngine()

    def test_inline_enumerated_clause_splits_into_atomic_units(self):
        text = (
            "4. Regulated entities shall: "
            "(a) submit the compliance report within 15 days; "
            "(b) maintain records for five years; "
            "(c) obtain prior approval before launch."
        )

        units = split_atomic_obligation_units(text)

        self.assertEqual(len(units), 3)
        self.assertTrue(all(unit.startswith("Regulated entities shall") for unit in units))

    def test_candidate_extraction_keeps_atomic_obligations_separate(self):
        section = SectionSchema(
            id="4",
            heading="Operative clause",
            text=(
                "4. Regulated entities shall: "
                "(a) submit the compliance report within 15 days; "
                "(b) maintain records for five years; "
                "(c) obtain prior approval before launch."
            ),
            clause_type="obligation",
        )

        obligations = extract_candidate_obligations(section, [], self.engine, source="RBI")

        self.assertEqual(len(obligations), 3)
        self.assertEqual(
            [item.action for item in obligations],
            [
                "Submit the compliance report within 15 days.",
                "Maintain records for five years.",
                "Obtain prior approval before launch.",
            ],
        )
        self.assertEqual(obligations[0].deadline.text, "within 15 days")

    def test_quoted_reference_is_blocked_even_when_it_contains_shall(self):
        section = SectionSchema(
            id="1",
            heading="Reference",
            text="The relevant provision reads as follows: banks shall maintain records.",
            clause_type="quoted_reference",
        )

        self.assertTrue(is_quoted_or_reference_unit(section.text, section.clause_type))
        self.assertEqual(extract_candidate_obligations(section, [], self.engine, source="RBI"), [])

    def test_dedup_does_not_merge_distinct_same_section_obligations(self):
        obligations = [
            _make_obligation("x1", "8", "Submit quarterly reports."),
            _make_obligation("x2", "8", "Maintain records for five years."),
            _make_obligation("x3", "8", "Obtain prior approval before product launch."),
        ]

        result = deduplicate_atomic_obligations(obligations)

        self.assertEqual(len(result), 3)


def _make_obligation(obligation_id: str, section_id: str, action: str) -> ObligationSchema:
    return ObligationSchema(
        id=obligation_id,
        section_id=section_id,
        clause_number=section_id,
        actor="Regulated Entity",
        action=action,
        obligation_type="mandatory",
        trigger="always",
        deadline=DeadlineSchema(text="ongoing", urgency="ongoing"),
        domain="ReportingAudit",
        departments=["Compliance"],
        severity="medium",
        severity_reason="Test severity.",
        evidence_required=["Compliance confirmation"],
        cross_references=[],
        confidence=0.8,
    )


if __name__ == "__main__":
    unittest.main()
