"""
Verification for both fixes:
  1. Financial year effective date extraction
  2. Discretionary obligation type detection

Shows before/after behavior for each fix.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.utils import parse_financial_year
from backend.core.rule_engine import RuleEngine
from backend.ingestion.structure_extractor import _extract_effective_date


def run_tests():
    passed = 0
    failed = 0
    re_engine = RuleEngine()

    def check(description, actual, expected):
        nonlocal passed, failed
        if actual == expected:
            passed += 1
            print(f"  [PASS] {description} -> {actual}")
        else:
            failed += 1
            print(f"  [FAIL] {description}")
            print(f"         Expected: {expected}")
            print(f"         Actual:   {actual}")

    # ══════════════════════════════════════════════════════════════════════
    # FIX 1: Financial Year Effective Date
    # ══════════════════════════════════════════════════════════════════════

    print("\n=== Fix 1: Financial Year Effective Date ===")

    # -- parse_financial_year utility --
    print("\n--- parse_financial_year() ---")
    check("FY 2022-23", parse_financial_year("FY 2022-23"), "2022-04-01")
    check("FY2022-23 (no space)", parse_financial_year("FY2022-23"), "2022-04-01")
    check("financial year 2022-23", parse_financial_year("financial year 2022-23"), "2022-04-01")
    check("financial year 2024-2025", parse_financial_year("financial year 2024-2025"), "2024-04-01")
    check("2022-23 (bare)", parse_financial_year("2022-23"), "2022-04-01")
    check("random text (no match)", parse_financial_year("random text"), None)
    check("empty string", parse_financial_year(""), None)

    # -- _extract_effective_date from structure_extractor --
    print("\n--- _extract_effective_date() ---")
    irdai_text = "5. This circular is effective from financial year 2022-23 onwards."
    check(
        "IRDAI circular FY clause (BEFORE: None, AFTER: 2022-04-01)",
        _extract_effective_date(irdai_text),
        "2022-04-01",
    )

    check(
        "Effective from FY 2025-26",
        _extract_effective_date("These guidelines are effective from FY 2025-26."),
        "2025-04-01",
    )

    check(
        "With effect from financial year 2023-24",
        _extract_effective_date("Applicable with effect from financial year 2023-24."),
        "2023-04-01",
    )

    # Ensure existing patterns still work
    check(
        "Existing: with effect from April 1, 2024",
        _extract_effective_date("with effect from April 1, 2024"),
        "2024-04-01",
    )

    check(
        "Existing: shall come into force on January 15, 2026",
        _extract_effective_date("shall come into force on January 15, 2026"),
        "2026-01-15",
    )

    # -- Rule engine extract_effective_date --
    print("\n--- RuleEngine.extract_effective_date() ---")
    check(
        "Rule engine: FY 2022-23",
        re_engine.extract_effective_date("This circular is effective from financial year 2022-23 onwards."),
        "2022-04-01",
    )

    check(
        "Rule engine: FY abbreviation",
        re_engine.extract_effective_date("Effective from FY 2025-26."),
        "2025-04-01",
    )

    check(
        "Rule engine: existing date (regression)",
        re_engine.extract_effective_date("with effect from April 1, 2024"),
        "2024-04-01",
    )

    # ══════════════════════════════════════════════════════════════════════
    # FIX 2: Discretionary Obligation Type
    # ══════════════════════════════════════════════════════════════════════

    print("\n=== Fix 2: Discretionary Obligation Type ===")

    # -- classify_obligation_type without trigger_rule --
    print("\n--- classify_obligation_type() (no trigger_rule) ---")

    check(
        "'may be accounted on estimate basis' (BEFORE: mandatory, AFTER: discretionary)",
        re_engine.classify_obligation_type(
            "Premium on reinsurance accepted may be accounted on estimate basis.", None
        ),
        "discretionary",
    )

    check(
        "'The insurer may appoint' (BEFORE: mandatory, AFTER: discretionary)",
        re_engine.classify_obligation_type(
            "The insurer may appoint an external actuary for this purpose.", None
        ),
        "discretionary",
    )

    check(
        "'at its discretion' -- discretionary",
        re_engine.classify_obligation_type(
            "The Authority may, at its discretion, grant extension.", None
        ),
        "discretionary",
    )

    check(
        "'at the discretion of' -- discretionary",
        re_engine.classify_obligation_type(
            "This shall be at the discretion of the Board.", None
        ),
        "discretionary",
    )

    # Ensure 'shall' still → mandatory
    check(
        "'shall ensure' -- mandatory (regression)",
        re_engine.classify_obligation_type(
            "The insurer shall ensure no premium is accrued on estimate basis.", None
        ),
        "mandatory",
    )

    # 'shall ... may' mixed → mandatory (shall takes precedence)
    check(
        "'shall ... may ...' mixed -- mandatory (shall dominates)",
        re_engine.classify_obligation_type(
            "The insurer shall submit reports, and may include supplementary notes.", None
        ),
        "mandatory",
    )

    # Conditional still works
    check(
        "'if the deviation exceeds' -- conditional (regression)",
        re_engine.classify_obligation_type(
            "If the deviation exceeds 10%, the entity must report.", None
        ),
        "conditional",
    )

    # -- Before/After table --
    print("\n\n" + "=" * 74)
    print(" BEFORE / AFTER COMPARISON")
    print("=" * 74)

    test_cases = [
        # (description, text, before, after)
        (
            "FY effective date",
            "This circular is effective from financial year 2022-23 onwards.",
            "effective_date: null",
            f"effective_date: {_extract_effective_date('This circular is effective from financial year 2022-23 onwards.')}",
        ),
        (
            "Discretionary 'may' clause",
            "Premium on reinsurance accepted may be accounted on estimate basis.",
            "obligation_type: mandatory",
            f"obligation_type: {re_engine.classify_obligation_type('Premium on reinsurance accepted may be accounted on estimate basis.', None)}",
        ),
    ]

    print(f"\n  {'Test Case':<30} {'Before':<30} {'After':<30}")
    print(f"  {'-'*30} {'-'*30} {'-'*30}")
    for desc, text, before, after in test_cases:
        print(f"  {desc:<30} {before:<30} {after:<30}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
