"""
Verification for both fixes:
  1. Financial year effective date extraction
  2. Discretionary obligation type detection

Shows before/after behavior for each fix.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.utils import parse_financial_year
from backend.core.rule_engine import RuleEngine
from backend.ingestion.structure_extractor import _extract_effective_date

@pytest.fixture
def re_engine():
    return RuleEngine()

@pytest.mark.parametrize("description, text, expected", [
    ("FY 2022-23", "FY 2022-23", "2022-04-01"),
    ("FY2022-23 (no space)", "FY2022-23", "2022-04-01"),
    ("financial year 2022-23", "financial year 2022-23", "2022-04-01"),
    ("financial year 2024-2025", "financial year 2024-2025", "2024-04-01"),
    ("2022-23 (bare)", "2022-23", "2022-04-01"),
    ("random text (no match)", "random text", None),
    ("empty string", "", None),
])
def test_parse_financial_year(description, text, expected):
    assert parse_financial_year(text) == expected, description

@pytest.mark.parametrize("description, text, expected", [
    ("IRDAI circular FY clause", "5. This circular is effective from financial year 2022-23 onwards.", "2022-04-01"),
    ("Effective from FY 2025-26", "These guidelines are effective from FY 2025-26.", "2025-04-01"),
    ("With effect from financial year 2023-24", "Applicable with effect from financial year 2023-24.", "2023-04-01"),
    ("Existing: with effect from April 1, 2024", "with effect from April 1, 2024", "2024-04-01"),
    ("Existing: shall come into force on January 15, 2026", "shall come into force on January 15, 2026", "2026-01-15"),
])
def test_extract_effective_date_from_structure_extractor(description, text, expected):
    assert _extract_effective_date(text) == expected, description

@pytest.mark.parametrize("description, text, expected", [
    ("Rule engine: FY 2022-23", "This circular is effective from financial year 2022-23 onwards.", "2022-04-01"),
    ("Rule engine: FY abbreviation", "Effective from FY 2025-26.", "2025-04-01"),
    ("Rule engine: existing date (regression)", "with effect from April 1, 2024", "2024-04-01"),
])
def test_rule_engine_extract_effective_date(re_engine, description, text, expected):
    assert re_engine.extract_effective_date(text) == expected, description

@pytest.mark.parametrize("description, text, expected", [
    ("'may be accounted on estimate basis' -> discretionary", "Premium on reinsurance accepted may be accounted on estimate basis.", "discretionary"),
    ("'The insurer may appoint' -> discretionary", "The insurer may appoint an external actuary for this purpose.", "discretionary"),
    ("'at its discretion' -> discretionary", "The Authority may, at its discretion, grant extension.", "discretionary"),
    ("'at the discretion of' -> discretionary", "This shall be at the discretion of the Board.", "discretionary"),
    ("'shall ensure' -> mandatory (regression)", "The insurer shall ensure no premium is accrued on estimate basis.", "mandatory"),
    ("'shall ... may ...' mixed -> mandatory (shall dominates)", "The insurer shall submit reports, and may include supplementary notes.", "mandatory"),
    ("'if the deviation exceeds' -> conditional (regression)", "If the deviation exceeds 10%, the entity must report.", "conditional"),
])
def test_rule_engine_classify_obligation_type(re_engine, description, text, expected):
    assert re_engine.classify_obligation_type(text, None) == expected, description
