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
import pytest
from unittest.mock import patch
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.obligation_extractor import (
    _try_resummarize,
    ACTION_VERBATIM_THRESHOLD,
    ACTION_MAX_LENGTH,
)
from backend.core.llm_client import llm, _PROMPT_RESUMMARIZE_ACTION


def test_prompt_template_exists():
    assert "compliance obligation summariser" in _PROMPT_RESUMMARIZE_ACTION, "Prompt starts with summariser instruction"
    assert "imperative verb" in _PROMPT_RESUMMARIZE_ACTION, "Prompt requires imperative verb"
    assert "Maximum 40 words" in _PROMPT_RESUMMARIZE_ACTION, "Prompt sets max 40 words"

def test_resummarize_action_method_exists():
    assert hasattr(llm, "resummarize_action"), "resummarize_action method exists"
    assert callable(getattr(llm, "resummarize_action", None)), "resummarize_action is callable"

def test_try_resummarize_with_good_llm_rewrite():
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

    ratio = SequenceMatcher(None, good_rewrite.lower(), source_text[:len(good_rewrite)].lower()).ratio()
    assert ratio <= ACTION_VERBATIM_THRESHOLD, f"Good rewrite ratio ({ratio:.2f}) <= threshold ({ACTION_VERBATIM_THRESHOLD})"
    assert action == good_rewrite, "Action was replaced"
    assert "re-summarized by LLM" in notes, "Notes say re-summarized"
    assert "verbatim" not in notes, "Notes do NOT say verbatim needs review"

def test_try_resummarize_falls_back_when_llm_returns_empty():
    verbatim_action = "Some verbatim action text here."
    with patch.object(llm, "resummarize_action", return_value=""):
        action, notes = _try_resummarize(verbatim_action, verbatim_action, None)

    assert action == verbatim_action, "Action unchanged on LLM failure"
    assert "verbatim — needs review" in notes, "Notes contain verbatim tag"

def test_try_resummarize_falls_back_when_rewrite_is_still_verbatim():
    verbatim_action = "Some verbatim action text here."
    with patch.object(llm, "resummarize_action", return_value=verbatim_action):
        action, notes = _try_resummarize(verbatim_action, verbatim_action, None)

    assert action == verbatim_action, "Action unchanged (rewrite too similar)"
    assert "verbatim — needs review" in notes, "Notes contain verbatim tag"

def test_try_resummarize_falls_back_when_rewrite_is_too_short():
    verbatim_action = "Some verbatim action text here."
    with patch.object(llm, "resummarize_action", return_value="Report."):
        action, notes = _try_resummarize(verbatim_action, verbatim_action, None)

    assert action == verbatim_action, "Action unchanged (rewrite too short)"
    assert "verbatim — needs review" in notes, "Notes contain verbatim tag"

def test_try_resummarize_falls_back_when_llm_throws_exception():
    verbatim_action = "Some verbatim action text here."
    with patch.object(llm, "resummarize_action", side_effect=RuntimeError("API down")):
        action, notes = _try_resummarize(verbatim_action, verbatim_action, None)

    assert action == verbatim_action, "Action unchanged on exception"
    assert "verbatim — needs review" in notes, "Notes contain verbatim tag"

def test_existing_notes_are_preserved():
    verbatim_action = "Some verbatim action text here."
    good_rewrite = "Report something nicely."
    existing_notes = "This is a prohibition — the actor must NOT perform the stated action."
    with patch.object(llm, "resummarize_action", return_value=good_rewrite):
        action, notes = _try_resummarize(verbatim_action, verbatim_action, existing_notes)

    assert "prohibition" in notes, "Original notes preserved"
    assert "re-summarized by LLM" in notes, "Re-summarization note appended"

def test_rewrite_that_is_too_long_is_rejected():
    verbatim_action = "Some verbatim action text here."
    long_rewrite = "X " * 200 + "end."  # > ACTION_MAX_LENGTH
    with patch.object(llm, "resummarize_action", return_value=long_rewrite):
        action, notes = _try_resummarize(verbatim_action, verbatim_action, None)

    assert action == verbatim_action, "Action unchanged (rewrite too long)"
    assert "verbatim — needs review" in notes, "Notes contain verbatim tag"
