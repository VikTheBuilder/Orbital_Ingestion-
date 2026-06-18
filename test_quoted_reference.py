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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.rule_engine import RuleEngine
from backend.ingestion.structure_extractor import (
    _classify_clause_type,
    _propagate_quoted_reference,
    _is_quoted_block,
)
from backend.ingestion.schemas import SectionSchema


def test_quoted_reference():
    engine = RuleEngine()

    passed = 0
    failed = 0

    def check(description, actual, expected):
        nonlocal passed, failed
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            failed += 1
            print(f"  [{status}] {description}")
            print(f"         Expected: {expected}")
            print(f"         Actual:   {actual}")
        else:
            passed += 1
            print(f"  [{status}] {description} -> {actual}")

    # ── Test 1: Rule engine classify_clause_type ──
    print("\n=== Test 1: Rule engine clause type classification ===")

    check(
        "Section with 'provides as under' (lead-in for quotation)",
        engine.classify_clause_type(
            "1. At present, para 2 of Part I of Schedule B of the Insurance Regulatory "
            "and Development Authority Regulations, 2002 provides as under for "
            'recognition of premium: "2. Premium'
        ),
        "quoted_reference"
    )

    check(
        "Section with 'reads as follows'",
        engine.classify_clause_type(
            "The relevant provision of the Act reads as follows: "
            '"Every bank shall maintain a minimum capital."'
        ),
        "quoted_reference"
    )

    check(
        "Regular obligation (shall maintain) - NOT quoted",
        engine.classify_clause_type(
            "The bank shall maintain adequate capital ratios at all times."
        ),
        "obligation"
    )

    check(
        "Regular definition - NOT quoted",
        engine.classify_clause_type(
            '"Regulated Entity" means a bank or NBFC regulated by the Reserve Bank.'
        ),
        "definition"
    )

    # ── Test 2: Fallback _classify_clause_type ──
    print("\n=== Test 2: Fallback _classify_clause_type ===")

    check(
        "Fallback: text with 'provides as under'",
        _classify_clause_type(
            "Regulation 5 provides as under for premium recognition: "
            '"Premium shall be recognized as income."'
        ),
        "quoted_reference"
    )

    check(
        "Fallback: regular obligation",
        _classify_clause_type(
            "All insurers shall submit quarterly returns."
        ),
        "obligation"
    )

    # ── Test 3: _is_quoted_block ──
    print("\n=== Test 3: _is_quoted_block detection ===")

    check(
        "Smart-quoted block",
        _is_quoted_block('\u201cPremium shall be recognized as income over the contract period.\u201d'),
        True
    )

    check(
        "Regular-quoted block",
        _is_quoted_block('"Premium shall be recognized as income."'),
        True
    )

    check(
        "Numbered clause with smart quotes",
        _is_quoted_block('(i) \u201cPremium shall be recognized as income.\u201d'),
        True
    )

    check(
        "Not a quoted block",
        _is_quoted_block("The bank shall maintain adequate capital."),
        False
    )

    # ── Test 4: _propagate_quoted_reference ──
    print("\n=== Test 4: Parent-to-child quote propagation ===")

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

    check(
        "Parent section 1 (has lead-in) -> quoted_reference",
        sections[0].clause_type, "quoted_reference"
    )
    check(
        "Child (i) under section 1 -> quoted_reference",
        sections[1].clause_type, "quoted_reference"
    )
    check(
        "Child (ii) under section 1 -> quoted_reference",
        sections[2].clause_type, "quoted_reference"
    )
    check(
        "Child (iii) under section 1 -> quoted_reference",
        sections[3].clause_type, "quoted_reference"
    )
    check(
        "Section 2 (next same-level) -> NOT quoted (stays 'other')",
        sections[4].clause_type, "other"
    )
    check(
        "Section 4 (real obligation) -> NOT quoted (stays 'obligation')",
        sections[5].clause_type, "obligation"
    )

    # ── Test 5: Obligation extractor skips quoted_reference ──
    print("\n=== Test 5: SKIP_CLAUSE_TYPES includes quoted_reference ===")
    from backend.ingestion.obligation_extractor import SKIP_CLAUSE_TYPES
    check(
        "quoted_reference in SKIP_CLAUSE_TYPES",
        "quoted_reference" in SKIP_CLAUSE_TYPES,
        True
    )

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = test_quoted_reference()
    sys.exit(0 if success else 1)
