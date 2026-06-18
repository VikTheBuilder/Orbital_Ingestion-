"""
Verification script for source-scoped actor detection fix.
Tests that:
  1. IRDAI documents no longer match "Bank" as an actor
  2. RBI documents still correctly match "Bank"
  3. IRDAI-specific actors (Insurer, Reinsurer, FRB, etc.) are detected
  4. Universal actors (Board of Directors, CEO, etc.) work for all sources
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.rule_engine import RuleEngine

def test_source_scoped_actors():
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
    
    print("\n=== Test 1: IRDAI documents should NOT match 'Bank' ===")
    check(
        "Premium shall be recognized as income (IRDAI)",
        engine.find_actor("Premium shall be recognized as income over the contract period", source="IRDAI"),
        "Regulated Entity"
    )
    check(
        "The bank shall submit (IRDAI) - 'bank' scoped out",
        engine.find_actor("the bank shall submit the report", source="IRDAI"),
        "Regulated Entity"
    )
    check(
        "Banks shall comply (IRDAI) - 'banks' scoped out",
        engine.find_actor("banks shall comply with these directions", source="IRDAI"),
        "Regulated Entity"
    )
    
    print("\n=== Test 2: RBI documents should STILL match 'Bank' ===")
    check(
        "The bank shall submit (RBI)",
        engine.find_actor("the bank shall submit the report", source="RBI"),
        "Bank"
    )
    check(
        "Banks shall comply (RBI)",
        engine.find_actor("banks shall comply with these directions", source="RBI"),
        "Bank"
    )
    check(
        "Scheduled commercial banks shall (RBI)",
        engine.find_actor("scheduled commercial banks shall maintain", source="RBI"),
        "Scheduled Commercial Bank"
    )
    
    print("\n=== Test 3: IRDAI-specific actors are detected ===")
    check(
        "the insurer shall ensure (IRDAI)",
        engine.find_actor("the insurer shall ensure compliance", source="IRDAI"),
        "Insurer"
    )
    check(
        "the reinsurer shall (IRDAI)",
        engine.find_actor("the reinsurer shall submit the accounts", source="IRDAI"),
        "Reinsurer"
    )
    check(
        "FRBs/Reinsurers shall ensure (IRDAI)",
        engine.find_actor("FRBs/Reinsurers shall ensure that in annual financial statements", source="IRDAI"),
        "FRBs/Reinsurers"
    )
    check(
        "GIC Re shall (IRDAI)",
        engine.find_actor("GIC Re shall report the premium", source="IRDAI"),
        "GIC Re"
    )
    check(
        "insurance company shall (IRDAI)",
        engine.find_actor("the insurance company shall maintain records", source="IRDAI"),
        "Insurer"
    )
    check(
        "appointed actuary shall (IRDAI)",
        engine.find_actor("the appointed actuary shall certify", source="IRDAI"),
        "Appointed Actuary"
    )
    check(
        "policyholder (IRDAI)",
        engine.find_actor("the policyholder shall submit documents", source="IRDAI"),
        "Policyholder"
    )
    
    print("\n=== Test 4: Universal actors work for ALL sources ===")
    check(
        "Board of Directors (IRDAI)",
        engine.find_actor("the Board of Directors shall approve", source="IRDAI"),
        "Board of Directors"
    )
    check(
        "Board of Directors (RBI)",
        engine.find_actor("the Board of Directors shall approve", source="RBI"),
        "Board of Directors"
    )
    check(
        "CEO (IRDAI)",
        engine.find_actor("the Chief Executive Officer shall ensure", source="IRDAI"),
        "Chief Executive Officer"
    )
    check(
        "Regulated Entity (IRDAI)",
        engine.find_actor("all regulated entities shall comply", source="IRDAI"),
        "Regulated Entity"
    )
    check(
        "Auditor (IRDAI)",
        engine.find_actor("the statutory auditor shall verify", source="IRDAI"),
        "Auditor"
    )
    
    print("\n=== Test 5: Backward compatibility - no source param ===")
    check(
        "The bank shall (no source) - all patterns active",
        engine.find_actor("the bank shall submit the report"),
        "Bank"
    )
    check(
        "Board of Directors (no source)",
        engine.find_actor("the Board of Directors shall approve"),
        "Board of Directors"
    )
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == "__main__":
    success = test_source_scoped_actors()
    sys.exit(0 if success else 1)
