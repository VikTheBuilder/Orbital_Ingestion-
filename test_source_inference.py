import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.pipeline import _detect_source_from_analysis, _detect_source_from_text


class SourceInferenceTests(unittest.TestCase):
    def test_text_detection_handles_irdai_variants(self):
        text = "This circular is issued by the Insurance Regulatory and Development Authority (IRDA) for insurers."
        self.assertEqual(_detect_source_from_text(text), "IRDAI")

    def test_analysis_detection_overrides_other_for_insurance_docs(self):
        analysis = {
            "primary_actor": "Insurance Regulatory and Development Authority (IRDA)",
            "reasoning": "This circular sets accounting and disclosure rules for insurers.",
        }
        self.assertEqual(_detect_source_from_analysis(analysis, "premium accounting circular"), "IRDAI")


if __name__ == "__main__":
    unittest.main()
