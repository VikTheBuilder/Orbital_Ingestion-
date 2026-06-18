"""
Verification script for quoted_reference detection.
Tests that:
  1. Background quotation lead-in sections are tagged as quoted_reference
  2. Child sub-clauses of quoted sections are also tagged quoted_reference
  3. Regular obligation sections are NOT affected
  4. The obligation extractor skips quoted_reference sections
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.rule_engine import RuleEngine
from backend.ingestion.structure_extractor import (
    _classify_clause_type,
    _propagate_quoted_reference,
    _is_quoted_block,
)
from backend.ingestion.schemas import SectionSchema
from backend.ingestion.obligation_extractor import SKIP_CLAUSE_TYPES

@pytest.fixture
def engine():
    return RuleEngine()

@pytest.mark.parametrize("description, text, expected", [
    (
        "Section with 'provides as under' (lead-in for quotation)",
        "1. At present, para 2 of Part I of Schedule B of the Insurance Regulatory "
        "and Development Authority Regulations, 2002 provides as under for "
        'recognition of premium: "2. Premium',
        "quoted_reference"
    ),
    (
        "Section with 'reads as follows'",
        "The relevant provision of the Act reads as follows: "
        '"Every bank shall maintain a minimum capital."',
        "quoted_reference"
    ),
    (
        "Regular obligation (shall maintain) - NOT quoted",
        "The bank shall maintain adequate capital ratios at all times.",
        "obligation"
    ),
    (
        "Regular definition - NOT quoted",
        '"Regulated Entity" means a bank or NBFC regulated by the Reserve Bank.',
        "definition"
    ),
])
def test_rule_engine_clause_type_classification(engine, description, text, expected):
    assert engine.classify_clause_type(text) == expected, description

@pytest.mark.parametrize("description, text, expected", [
    (
        "Fallback: text with 'provides as under'",
        "Regulation 5 provides as under for premium recognition: "
        '"Premium shall be recognized as income."',
        "quoted_reference"
    ),
    (
        "Fallback: regular obligation",
        "All insurers shall submit quarterly returns.",
        "obligation"
    ),
])
def test_fallback_classify_clause_type(description, text, expected):
    assert _classify_clause_type(text) == expected, description

@pytest.mark.parametrize("description, text, expected", [
    ("Smart-quoted block", '\u201cPremium shall be recognized as income over the contract period.\u201d', True),
    ("Regular-quoted block", '"Premium shall be recognized as income."', True),
    ("Numbered clause with smart quotes", '(i) \u201cPremium shall be recognized as income.\u201d', True),
    ("Not a quoted block", "The bank shall maintain adequate capital.", False),
])
def test_is_quoted_block_detection(description, text, expected):
    assert _is_quoted_block(text) == expected, description

def test_propagate_quoted_reference():
    sections = [
        SectionSchema(
            id="1", heading="Para 1",
            text="1. At present, para 2 of Schedule B provides as under for recognition of premium:",
            clause_type="other", level=1,
        ),
        SectionSchema(
            id="i", heading="Sub (i)",
            text="(i) Premium shall be recognized as income over the contract period.",
            clause_type="obligation", level=3,
        ),
        SectionSchema(
            id="ii", heading="Sub (ii)",
            text='(ii) \u201cPremium received in Advance\u201d is the premium where inception is outside.',
            clause_type="other", level=3,
        ),
        SectionSchema(
            id="iii", heading="Sub (iii)",
            text='(iii) \u201cUnallocated premium\u201d includes premium deposit.',
            clause_type="definition", level=3,
        ),
        SectionSchema(
            id="2", heading="Para 2",
            text="2. The Authority has carried out an analysis of the premium.",
            clause_type="other", level=1,
        ),
        SectionSchema(
            id="4", heading="Para 4",
            text="4. FRBs/Reinsurers shall ensure that no premium is accrued on estimate basis.",
            clause_type="obligation", level=1,
        ),
    ]

    _propagate_quoted_reference(sections)

    assert sections[0].clause_type == "quoted_reference", "Parent section 1 (has lead-in) -> quoted_reference"
    assert sections[1].clause_type == "quoted_reference", "Child (i) under section 1 -> quoted_reference"
    assert sections[2].clause_type == "quoted_reference", "Child (ii) under section 1 -> quoted_reference"
    assert sections[3].clause_type == "quoted_reference", "Child (iii) under section 1 -> quoted_reference"
    assert sections[4].clause_type == "other", "Section 2 (next same-level) -> NOT quoted (stays 'other')"
    assert sections[5].clause_type == "obligation", "Section 4 (real obligation) -> NOT quoted (stays 'obligation')"

def test_skip_clause_types_includes_quoted_reference():
    assert "quoted_reference" in SKIP_CLAUSE_TYPES, "quoted_reference in SKIP_CLAUSE_TYPES"
