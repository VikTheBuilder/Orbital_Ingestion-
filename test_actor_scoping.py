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
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.rule_engine import RuleEngine

@pytest.fixture
def engine():
    return RuleEngine()

@pytest.mark.parametrize("description, text, source, expected", [
    ("Premium shall be recognized as income (IRDAI)", "Premium shall be recognized as income over the contract period", "IRDAI", "Regulated Entity"),
    ("The bank shall submit (IRDAI) - 'bank' scoped out", "the bank shall submit the report", "IRDAI", "Regulated Entity"),
    ("Banks shall comply (IRDAI) - 'banks' scoped out", "banks shall comply with these directions", "IRDAI", "Regulated Entity"),
])
def test_irdai_should_not_match_bank(engine, description, text, source, expected):
    assert engine.find_actor(text, source=source) == expected, description

@pytest.mark.parametrize("description, text, source, expected", [
    ("The bank shall submit (RBI)", "the bank shall submit the report", "RBI", "Bank"),
    ("Banks shall comply (RBI)", "banks shall comply with these directions", "RBI", "Bank"),
    ("Scheduled commercial banks shall (RBI)", "scheduled commercial banks shall maintain", "RBI", "Scheduled Commercial Bank"),
])
def test_rbi_should_match_bank(engine, description, text, source, expected):
    assert engine.find_actor(text, source=source) == expected, description

@pytest.mark.parametrize("description, text, source, expected", [
    ("the insurer shall ensure (IRDAI)", "the insurer shall ensure compliance", "IRDAI", "Insurer"),
    ("the reinsurer shall (IRDAI)", "the reinsurer shall submit the accounts", "IRDAI", "Reinsurer"),
    ("FRBs/Reinsurers shall ensure (IRDAI)", "FRBs/Reinsurers shall ensure that in annual financial statements", "IRDAI", "FRBs/Reinsurers"),
    ("GIC Re shall (IRDAI)", "GIC Re shall report the premium", "IRDAI", "GIC Re"),
    ("insurance company shall (IRDAI)", "the insurance company shall maintain records", "IRDAI", "Insurer"),
    ("appointed actuary shall (IRDAI)", "the appointed actuary shall certify", "IRDAI", "Appointed Actuary"),
    ("policyholder (IRDAI)", "the policyholder shall submit documents", "IRDAI", "Policyholder"),
])
def test_irdai_specific_actors(engine, description, text, source, expected):
    assert engine.find_actor(text, source=source) == expected, description

@pytest.mark.parametrize("description, text, source, expected", [
    ("Board of Directors (IRDAI)", "the Board of Directors shall approve", "IRDAI", "Board of Directors"),
    ("Board of Directors (RBI)", "the Board of Directors shall approve", "RBI", "Board of Directors"),
    ("CEO (IRDAI)", "the Chief Executive Officer shall ensure", "IRDAI", "Chief Executive Officer"),
    ("Regulated Entity (IRDAI)", "all regulated entities shall comply", "IRDAI", "Regulated Entity"),
    ("Auditor (IRDAI)", "the statutory auditor shall verify", "IRDAI", "Auditor"),
])
def test_universal_actors_all_sources(engine, description, text, source, expected):
    assert engine.find_actor(text, source=source) == expected, description

@pytest.mark.parametrize("description, text, expected", [
    ("The bank shall (no source) - all patterns active", "the bank shall submit the report", "Bank"),
    ("Board of Directors (no source)", "the Board of Directors shall approve", "Board of Directors"),
])
def test_backward_compatibility_no_source(engine, description, text, expected):
    assert engine.find_actor(text) == expected, description
