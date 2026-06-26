import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.obligation_extractor import extract_obligations
from backend.ingestion.pipeline import _heuristic_document_analysis
from backend.ingestion.schemas import DocumentStructureSchema, SectionSchema


SECTION_4_TEXT = (
    "4. Given that a significant part of the premium is being accounted on estimation basis, "
    "a need is felt to lay down guidelines to govern the accounting and disclosures of premium "
    "recognized on estimation basis in the annual report. Accordingly, the Authority, in "
    "exercising its powers given under Section 14 (2) lays down the following framework where "
    "under the FRBs/Reinsurers shall ensure that in annual financial statements no premium is "
    "accrued / accounted on estimate basis at least up to 3rd quarter of each financial year. "
    "However, for the fourth quarter ending on 31st March, where the statement of accounts has "
    "not been received in time, the premium, losses and related expenses may be accounted on "
    "estimation basis. However, in estimation of the said income and expenses, the reinsurers "
    "shall ensure that: a) a consistent methodology is followed across the entire portfolio; "
    "b) the estimates shall be trued up as actual values emerge; c) a statement shall be "
    "included in the annual report stating total premium, claims and expenses accounted for "
    "during the financial year and premium, claims and expenses accounted on estimation basis; "
    "d) complete disclosure shall be made for three years (including the current Financial Year) "
    "giving the segment wise break up of premium, claims and expenses accounted on estimation "
    "basis and its actual experience as per the attached formats - Annexure 1 and Management's "
    "comments on variation, if any, beyond 10% on a yearly basis under Notes to Accounts if the "
    "actual figures are available at the time of closing of books of accounts for the said "
    "financial year; and e) If the actual figures are not available at the time of closure of "
    "books of accounts for the financial year, any deviation beyond +/- 10% shall be reported "
    "to the Authority in the format referred in above para 4(d) within 15 days from the end of "
    "first quarter of the next financial year."
)


class DocumentAnalysisFlowTests(unittest.TestCase):
    def test_heuristic_analysis_plus_candidate_engine_recovers_operational_clauses(self):
        section = SectionSchema(
            id="4",
            heading="Framework clause",
            text=SECTION_4_TEXT,
            clause_type="obligation",
        )
        analysis = _heuristic_document_analysis(SECTION_4_TEXT, "IRDAI")
        doc = DocumentStructureSchema(
            doc_id="doc-4",
            source="IRDAI",
            title="IRDAI premium accounting",
            sections=[section],
            analysis=analysis,
        )

        obligations = extract_obligations(doc)

        self.assertGreaterEqual(len(obligations), 5)
        self.assertTrue(any("within 15 days" in ob.action.lower() for ob in obligations))


if __name__ == "__main__":
    unittest.main()
