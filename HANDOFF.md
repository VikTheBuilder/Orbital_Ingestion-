# Orbital Ingestion Pipeline - Handoff & Changelog

## Overview
This document serves as a comprehensive handoff guide and changelog for the recent debugging session on the ORBITAL Extraction Pipeline. 

The pipeline was rigorously tested against a real IRDAI circular PDF ("Accounting of Premium, claims and related expenses on estimation basis"), which surfaced several real-world bugs. Each bug was fixed individually, tested in isolation via unit tests, and then the fixes were composed and re-tested end-to-end against the same document. This end-to-end testing approach proved critical, as it caught a real regression in the obligation merging logic (detailed in "Known Issues" below).

---

## Fixes Completed and Verified

The following issues were successfully identified, fixed, and verified via the test suite:

1. **Source-scoped actor detection** 
   - **Problem:** IRDAI documents were incorrectly tagging actors as "Bank", bleeding over from a shared or generic actor list.
   - **Fix/Files Changed:** Scoped actor detection per regulatory source. Updated `rules/actors/irdai_actor_patterns.json`, `backend/core/rule_engine.py` (`find_actor()`), and `backend/ingestion/obligation_extractor.py`.
   - **Verification:** Verified via `test_actor_scoping.py` (20 test cases).

2. **Quoted-reference clause detection**
   - **Problem:** Obligations were being extracted from quoted pre-existing law rather than actual directives issued by the circular itself.
   - **Fix/Files Changed:** Added detection for background quotations. Updated `backend/ingestion/schemas.py`, `rules/clause_segmentation.json`, `backend/core/rule_engine.py` (`classify_clause_type()`), `backend/ingestion/structure_extractor.py`, and `backend/ingestion/obligation_extractor.py` (`SKIP_CLAUSE_TYPES`).
   - **Verification:** Verified via `test_quoted_reference.py` (12 test cases).

3. **Obligation deduplication and sub-clause merging**
   - **Problem:** Single cohesive regulatory clauses (e.g., one clause with sub-points a-e) were being fragmented into multiple near-duplicate obligation records.
   - **Fix/Files Changed:** Added a deduplication and merge pass to combine overlapping clauses into a single obligation with sub-actions. Updated `backend/ingestion/obligation_extractor.py` (`_deduplicate_and_merge()`).
   - **Verification:** Verified via `test_obligation_merge.py` (5 test cases).

4. **Globally unique obligation IDs**
   - **Problem:** Obligation `id` fields were reused across different sections, causing ambiguities during validation matching.
   - **Fix/Files Changed:** Shifted to a globally unique `{section_id}-{sequence_number}` format. Updated `backend/ingestion/obligation_extractor.py` (`_coerce_llm_obligation()`) and `extract_obligations()` (final renumbering pass).
   - **Verification:** Verified via `test_unique_ids.py` (5 test cases).

5. **LLM resummarization of verbatim-flagged actions**
   - **Problem:** Obligations flagged as `[action_quality: verbatim — needs review]` were passing verbatim text through the pipeline without actually attempting to resolve the issue.
   - **Fix/Files Changed:** Added a re-summarization step asking the LLM to rewrite the action into a concise imperative summary. Updated `backend/core/llm_client.py` (`resummarize_action()`) and `backend/ingestion/obligation_extractor.py` (`_try_resummarize()`).
   - **Verification:** Verified via `test_resummarize.py` (9 test cases).

6. **Financial-year effective date parsing + discretionary obligation type detection**
   - **Problem:** Effective dates stated as financial years (e.g., "FY 2022-23") were missed, and clauses with discretionary language ("may be accounted...") were wrongly marked "mandatory".
   - **Fix/Files Changed:** Added pattern matching for financial years and discretionary verbs ("may", "at its discretion"). Updated `backend/core/utils.py` (`parse_financial_year()`), `backend/ingestion/structure_extractor.py` (`_extract_effective_date()`), `rules/effective_date_patterns.json`, and `backend/core/rule_engine.py` (`classify_obligation_type()`).
   - **Verification:** Verified via `test_effdate_discretionary.py` (22 test cases).

7. **LLM JSON parsing resilience for fine-tune pair generation**
   - **Problem:** The local Ollama model (phi3:latest) occasionally returned malformed JSON (e.g., markdown fences, trailing prose, `±` characters) during fine-tune pair generation, causing silent drops.
   - **Fix/Files Changed:** Added robust pre-parse cleanup and a single-retry mechanism with explicit JSON-only instructions. Also exposed the number of dropped pairs to CLI summaries. Updated `backend/core/llm_client.py` (`_parse_json()`, `_chat_and_parse()`) and `backend/ingestion/pipeline.py`.
   - **Verification:** Verified via `test_llm_json_parsing.py` (3 test cases).

8. **Validator/extractor effective_date desync fix**
   - **Problem:** The validation pass reported missing effective dates (and populated `missed_obligations`) even when `doc_structure.effective_date` was already correctly populated.
   - **Fix/Files Changed:** Updated `validate_extraction()` to accept the doc-level effective date, suppressing false positives. Updated CLI warning logic in `backend/ingestion/pipeline.py` and `backend/ingestion/validator.py`.
   - **Verification:** Verified via `test_validator_effective_date.py` (1 test case).

---

## Known Issues — NOT Yet Resolved

> [!WARNING]
> The following issue is an active regression introduced during the debugging session that negatively impacts real-world extractions. 

- **Attempted fix:** Mixed obligation type handling during merge.
  - **Goal:** In `backend/ingestion/obligation_extractor.py` (`_merge_section_group()`), we intended to add a structured `mixed_obligation_types` note when merging clauses with differing `obligation_type` values (e.g., a mandatory core rule containing a discretionary carve-out), and set the top-level type to the most restrictive value.
- **Status: REGRESSED, not fixed.** 
  - When re-tested against the real IRDAI PDF after this change, obligation extraction quality got WORSE, not better.
  - **Previous run (before this fix):** Yielded 1 obligation where the action text correctly led with the core mandatory rule, merged 9 sub-clauses, and the validator reported only 1 missed obligation.
  - **This run (after this fix):** Yielded 1 obligation, but the action text *incorrectly leads with the Q4 discretionary exception* instead of the core rule. It only merged 3 sub-clauses, and the validator reported 6 missed obligations (sub-points a-e individually flagged as "shall obligation not extracted").
  - **Missing Notes:** The structured `mixed_obligation_types` note demonstrated in the synthetic unit test never appeared in the real-document output.
  - **Contradictory Fields:** The `severity_reason` and `severity` fields are now internally contradictory in the output ("discretionary, informational, or long/conditional timeline" reasoning is now paired with `severity: "high"`).
- **Next steps needed:** 
  1. Root-cause why `_merge_section_group()` is now selecting/merging fewer candidate obligations per section than before the change.
  2. Investigate why the `mixed_obligation_types` field isn't appearing in real runs despite passing the synthetic unit test (this indicates the unit test fixture does not accurately represent the real rule-engine output shape).

*(Note: See the `data/structured/` outputs in previous Git commits for direct diff comparisons of the before/after JSON states.)*

---

## Test Suite Status

- **Total Cases:** 76+ pytest cases across `test_actor_scoping.py`, `test_quoted_reference.py`, `test_obligation_merge.py`, `test_unique_ids.py`, `test_resummarize.py`, `test_effdate_discretionary.py`, `test_llm_json_parsing.py`, and `test_validator_effective_date.py`.
- **Refactoring:** All tests were successfully converted from a custom `check()`/counter pattern to native `pytest assert` + `parametrize` structures (eliminating the `PytestReturnNotNoneWarning`).
- **IMPORTANT CAVEAT:** All of these tests currently pass in isolation, but they **do NOT currently catch the Known Issue regression above.** This is a significant test coverage gap. 
  - *Recommendation:* Add an end-to-end integration test that runs the actual IRDAI PDF through the full pipeline and asserts on obligation count / merged sub-clause count, not just unit-level fixture behavior, so future regressions are caught automatically.

---

## How to Test

**Step 1: Environment Setup**
Ensure your `.env` file is configured (copy from `.env.example`). You will need either a local Ollama model (e.g., `phi3:latest`) or a valid Groq API key configured.

**Step 2: Run Unit Tests**
Run the full test suite to confirm basic functionality:
```bash
pytest -v
```

**Step 3: End-to-End Pipeline Run**
Run the pipeline against a real PDF with verbose logging:
```bash
python backend/scripts/run_pipeline.py --pdf path/to/document.pdf --verbose
```

**Step 4: Manual Verification**
Review the generated structured JSON (found in `data/structured/`) and verify the following fields:
- `actor` correctness (ensure IRDAI documents don't default to "Bank").
- `clause_type` for quoted text (should be `quoted_reference`).
- Obligation merge quality (look for fragmented sub-clauses vs a cohesive merged action).
- `effective_date` at the document level.
- `obligation_type` and any warning flags in the `notes` field.

---

## Repo / Git Notes

- Ensure you review the current branch state before beginning work (`git status`).
- The `data/raw`, `data/structured`, and `data/finetune` directories are gitignored by design, as they contain generated outputs and not source code. 
- You can leverage `git diff` against the previous commit to inspect exactly how `_merge_section_group()` was altered before the regression occurred.
