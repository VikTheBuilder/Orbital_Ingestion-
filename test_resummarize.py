"""
Verification for verbatim re-summarization logic.
Tests that:
  1. _try_resummarize() calls the LLM and accepts good rewrites
  2. _try_resummarize() falls back to the original + tag when LLM fails or
     the rewrite is still too verbatim
  3. The LLM client's resummarize_action() method exists and is callable
  4. The prompt template is correctly defined
"""

import sys
import os
from unittest.mock import patch, MagicMock
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.obligation_extractor import (
    _try_resummarize,
    ACTION_VERBATIM_THRESHOLD,
    ACTION_MAX_LENGTH,
    ACTION_MIN_LENGTH,
)
from backend.core.llm_client import llm, _PROMPT_RESUMMARIZE_ACTION


def test_resummarize():
    passed = 0
    failed = 0

    def check(description, actual, expected):
        nonlocal passed, failed
        if actual == expected:
            passed += 1
            print(f"  [PASS] {description}")
        else:
            failed += 1
            print(f"  [FAIL] {description}")
            print(f"         Expected: {expected}")
            print(f"         Actual:   {actual}")

    def check_contains(description, text, substring):
        nonlocal passed, failed
        if substring in (text or ""):
            passed += 1
            print(f"  [PASS] {description}")
        else:
            failed += 1
            print(f"  [FAIL] {description}")
            print(f"         Expected substring: {substring}")
            print(f"         In text: {(text or '')[:200]}")

    def check_not_contains(description, text, substring):
        nonlocal passed, failed
        if substring not in (text or ""):
            passed += 1
            print(f"  [PASS] {description}")
        else:
            failed += 1
            print(f"  [FAIL] {description}")
            print(f"         Unexpected substring: {substring}")
            print(f"         In text: {(text or '')[:200]}")

    # ── Test 1: Prompt template exists ──
    print("\n=== Test 1: Prompt template ===")
    check_contains(
        "Prompt starts with summariser instruction",
        _PROMPT_RESUMMARIZE_ACTION, "compliance obligation summariser"
    )
    check_contains(
        "Prompt requires imperative verb",
        _PROMPT_RESUMMARIZE_ACTION, "imperative verb"
    )
    check_contains(
        "Prompt sets max 40 words",
        _PROMPT_RESUMMARIZE_ACTION, "Maximum 40 words"
    )

    # ── Test 2: resummarize_action method exists ──
    print("\n=== Test 2: LLM client method ===")
    check("resummarize_action method exists", hasattr(llm, "resummarize_action"), True)
    check("resummarize_action is callable", callable(getattr(llm, "resummarize_action", None)), True)

    # ── Test 3: _try_resummarize with good LLM rewrite ──
    print("\n=== Test 3: Good LLM rewrite accepted ===")

    verbatim_action = (
        "Be made in the annual report for the next financial year; and e) "
        "If the actual figure s are n ot available at the time of clos ure of "
        "books of accounts for the financial year, any deviation beyond ± 10% "
        "be reported to the Authority in the format referred in above para 4(d) "
        "within 15 days from the end of first quarter of the next financial year."
    )
    source_text = verbatim_action  # verbatim = ratio ~1.0

    good_rewrite = "Report deviation beyond ±10% to the Authority within 15 days of end of first quarter of the next financial year."

    with patch.object(llm, "resummarize_action", return_value=good_rewrite):
        action, notes = _try_resummarize(verbatim_action, source_text, None)

    # The rewrite is short and different from source → should be accepted
    ratio = SequenceMatcher(None, good_rewrite.lower(), source_text[:len(good_rewrite)].lower()).ratio()
    check(f"Good rewrite ratio ({ratio:.2f}) <= threshold ({ACTION_VERBATIM_THRESHOLD})",
          ratio <= ACTION_VERBATIM_THRESHOLD, True)
    check("Action was replaced", action, good_rewrite)
    check_contains("Notes say re-summarized", notes, "re-summarized by LLM")
    check_not_contains("Notes do NOT say verbatim needs review", notes, "verbatim")

    # ── Test 4: _try_resummarize falls back when LLM returns empty ──
    print("\n=== Test 4: LLM failure falls back to original + tag ===")

    with patch.object(llm, "resummarize_action", return_value=""):
        action, notes = _try_resummarize(verbatim_action, source_text, None)

    check("Action unchanged on LLM failure", action, verbatim_action)
    check_contains("Notes contain verbatim tag", notes, "verbatim — needs review")

    # ── Test 5: _try_resummarize falls back when rewrite is still verbatim ──
    print("\n=== Test 5: Still-verbatim rewrite rejected ===")

    # Return something that's basically the same as the source
    near_copy = verbatim_action  # exact copy → ratio = 1.0

    with patch.object(llm, "resummarize_action", return_value=near_copy):
        action, notes = _try_resummarize(verbatim_action, source_text, None)

    check("Action unchanged (rewrite too similar)", action, verbatim_action)
    check_contains("Notes contain verbatim tag", notes, "verbatim — needs review")

    # ── Test 6: _try_resummarize falls back when rewrite is too short ──
    print("\n=== Test 6: Too-short rewrite rejected ===")

    with patch.object(llm, "resummarize_action", return_value="Report."):
        action, notes = _try_resummarize(verbatim_action, source_text, None)

    check("Action unchanged (rewrite too short)", action, verbatim_action)
    check_contains("Notes contain verbatim tag", notes, "verbatim — needs review")

    # ── Test 7: _try_resummarize falls back when LLM throws exception ──
    print("\n=== Test 7: LLM exception falls back gracefully ===")

    with patch.object(llm, "resummarize_action", side_effect=RuntimeError("API down")):
        action, notes = _try_resummarize(verbatim_action, source_text, None)

    check("Action unchanged on exception", action, verbatim_action)
    check_contains("Notes contain verbatim tag", notes, "verbatim — needs review")

    # ── Test 8: Existing notes are preserved ──
    print("\n=== Test 8: Existing notes preserved ===")

    existing_notes = "This is a prohibition — the actor must NOT perform the stated action."
    with patch.object(llm, "resummarize_action", return_value=good_rewrite):
        action, notes = _try_resummarize(verbatim_action, source_text, existing_notes)

    check_contains("Original notes preserved", notes, "prohibition")
    check_contains("Re-summarization note appended", notes, "re-summarized by LLM")

    # ── Test 9: Rewrite that's too long is rejected ──
    print("\n=== Test 9: Too-long rewrite rejected ===")

    long_rewrite = "X " * 200 + "end."  # > ACTION_MAX_LENGTH
    with patch.object(llm, "resummarize_action", return_value=long_rewrite):
        action, notes = _try_resummarize(verbatim_action, source_text, None)

    check("Action unchanged (rewrite too long)", action, verbatim_action)
    check_contains("Notes contain verbatim tag", notes, "verbatim — needs review")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = test_resummarize()
    sys.exit(0 if success else 1)
