import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.json_repair import repair_extracted_json


class JsonRepairTests(unittest.TestCase):
    def test_repair_adds_missed_obligation_and_updates_fields(self):
        extracted = {
            "doc_id": "doc-1",
            "source": "IRDAI",
            "title": "Sample circular",
            "effective_date": None,
            "sections": [
                {
                    "id": "4",
                    "heading": "Sample",
                    "text": "4. Reinsurers shall ensure that a statement shall be included in the annual report.",
                    "clause_type": "obligation",
                }
            ],
            "obligations": [
                {
                    "id": "4-OB1",
                    "section_id": "4",
                    "clause_number": "4",
                    "actor": "Reinsurer",
                    "action": "Ensure that a statement be included in the annual report.",
                    "obligation_type": "mandatory",
                    "trigger": "always",
                    "deadline": {"text": "ongoing", "absolute_date": None, "duration": None, "urgency": "ongoing"},
                    "domain": "ReportingAudit",
                    "departments": ["Compliance"],
                    "severity": "medium",
                    "severity_reason": "test",
                    "evidence_required": [],
                    "penalty_if_missed": None,
                    "fine_exposure_inr": None,
                    "cross_references": [],
                    "confidence": 0.84,
                    "notes": None,
                }
            ],
            "analysis": {"likely_effective_date_text": "2022-04-01", "primary_actor": "FRBs/Reinsurers"},
            "validation": None,
        }
        validation = {
            "missed_obligations": [
                {
                    "clause_number": "4",
                    "raw_text": "complete disclosure shall be made for three years including the current financial year.",
                    "reason_missed": "shall obligation not extracted",
                }
            ],
            "incorrect_extractions": [
                {
                    "obligation_id": "4-OB1",
                    "field": "departments",
                    "current_value": "Compliance",
                    "correct_value": "Compliance, InternalAudit, Finance",
                    "reason": "Department mapping does not align with the operational domain cues in the clause.",
                }
            ],
            "missing_effective_date": None,
            "overall_confidence": 0.75,
            "validation_notes": "test",
        }

        result = repair_extracted_json(extracted, validation)

        self.assertEqual(len(result["repaired_json"]["obligations"]), 2)
        self.assertTrue(any("complete disclosure" in ob["action"].lower() for ob in result["repaired_json"]["obligations"]))
        self.assertTrue(any("InternalAudit" in ", ".join(ob["departments"]) for ob in result["repaired_json"]["obligations"]))
        self.assertIn("validation", result)
        self.assertIn("repair_summary", result)


if __name__ == "__main__":
    unittest.main()
