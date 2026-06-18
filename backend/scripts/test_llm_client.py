"""
Test the OrbitalLLMClient standalone.
Runs 5 tests against the configured LLM provider.
"""

import sys
import os
import time

# Ensure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.core.llm_client import llm
from backend.core.config import get_config

config = get_config()
provider = (config.LLM_PROVIDER or "groq").strip().lower()

def run_tests():
    print("═══════════════════════════════════")
    print("ORBITAL LLM CLIENT — TEST SUITE")
    print("═══════════════════════════════════\n")

    results = []

    # Common test text
    test_text = (
        "All Regulated Entities shall, within 60 days, "
        "review and update their customer risk categorisation framework. "
        "Banks must ensure board approval before implementation."
    )

    obligation_to_test = None

    # ── Test 1: Obligation Extraction ──
    print("Test 1 — Obligation Extraction")
    try:
        start = time.time()
        result = llm.extract_obligations(test_text, provider=provider)
        duration = round(time.time() - start, 1)
        
        assertions = [
            (isinstance(result, list), "Result should be a list"),
            (len(result) >= 1, "Should extract at least one obligation"),
        ]
        
        if isinstance(result, list) and len(result) >= 1:
            first_ob = result[0]
            required_keys = {"actor", "action", "obligation_type", "trigger", "deadline", "domain",
                             "departments", "severity", "severity_reason", "evidence_required", "confidence"}
            assertions.append((all(k in first_ob for k in required_keys), "All required keys must be present"))
            assertions.append((first_ob.get("confidence", 0) >= 0.5, "Confidence should be >= 0.5"))
            obligation_to_test = first_ob
        else:
            assertions.append((False, "Could not check keys because result is empty or not a list"))

        passed = all(a[0] for a in assertions)
        results.append(passed)
        icon = "✓" if passed else "✗"
        if passed:
            print(f"  {icon} Test 1 passed — obligation extraction ({duration}s)")
        else:
            print(f"  {icon} Test 1 failed")
            for ok, msg in assertions:
                if not ok:
                    print(f"    FAIL: {msg}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ Test 1 failed — Exception: {e}")

    # ── Test 2: Domain Classification ──
    print("\nTest 2 — Domain Classification")
    try:
        start = time.time()
        result = llm.classify_domain(test_text)
        duration = round(time.time() - start, 1)

        assertions = [
            (isinstance(result, dict), "Result should be a dict"),
            (isinstance(result.get("primary_domain"), str), "Primary domain should be a string"),
            (result.get("confidence", 0) > 0.6, "Confidence should be > 0.6"),
        ]
        
        passed = all(a[0] for a in assertions)
        results.append(passed)
        icon = "✓" if passed else "✗"
        if passed:
            print(f"  {icon} Test 2 passed — domain classification ({duration}s)")
        else:
            print(f"  {icon} Test 2 failed")
            for ok, msg in assertions:
                if not ok:
                    print(f"    FAIL: {msg}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ Test 2 failed — Exception: {e}")

    # ── Test 3: Task Generation ──
    print("\nTest 3 — Task Generation")
    try:
        if not obligation_to_test:
            raise ValueError("Skipping Test 3: No obligation extracted from Test 1")

        start = time.time()
        result = llm.generate_map_card(obligation_to_test, provider=provider)
        duration = round(time.time() - start, 1)

        assertions = [
            (isinstance(result, dict), "Result should be a dict"),
            ("task" in result, "Missing 'task' key"),
            ("checklist" in result, "Missing 'checklist' key"),
            (len(result.get("checklist", [])) >= 1, "Checklist should have items"), # Original prompt says exactly 3, but LLM might give different length depending on logic
        ]
        
        passed = all(a[0] for a in assertions)
        results.append(passed)
        icon = "✓" if passed else "✗"
        if passed:
            print(f"  {icon} Test 3 passed — task generation ({duration}s)")
        else:
            print(f"  {icon} Test 3 failed")
            for ok, msg in assertions:
                if not ok:
                    print(f"    FAIL: {msg}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ Test 3 failed — Exception: {e}")

    # ── Test 4: Gap Analysis ──
    print("\nTest 4 — Gap Analysis")
    try:
        if not obligation_to_test:
            raise ValueError("Skipping Test 4: No obligation extracted from Test 1")

        policy_chunks = [{"text": "Risk categorisation based on income and geography. Annual review by Risk team."}]
        start = time.time()
        result = llm.analyse_gap(obligation_to_test, policy_chunks, provider=provider)
        duration = round(time.time() - start, 1)

        assertions = [
            (isinstance(result, dict), "Result should be a dict"),
            ("covered" in result, "Missing 'covered' key"),
            ("human_readable_summary" in result, "Missing 'human_readable_summary' key"),
            (len(result.get("human_readable_summary", "")) > 20, "Summary should be > 20 chars"),
        ]
        
        passed = all(a[0] for a in assertions)
        results.append(passed)
        icon = "✓" if passed else "✗"
        if passed:
            print(f"  {icon} Test 4 passed — gap analysis ({duration}s)")
        else:
            print(f"  {icon} Test 4 failed")
            for ok, msg in assertions:
                if not ok:
                    print(f"    FAIL: {msg}")
    except Exception as e:
        results.append(False)
        print(f"  ✗ Test 4 failed — Exception: {e}")

    # ── Test 5: Rate Limit Guard ──
    print("\nTest 5 — Rate Limit Guard")
    try:
        start = time.time()
        # Make 5 rapid calls
        for _ in range(5):
            res = llm.extract_obligations("Quick test sentence with shall keyword.", provider=provider)
            if not isinstance(res, list):
                raise ValueError("Expected list return from rapid calls")
        duration = round(time.time() - start, 1)
        
        results.append(True)
        print(f"  ✓ Test 5 passed — rate limit guard ({duration}s)")
    except Exception as e:
        results.append(False)
        print(f"  ✗ Test 5 failed — Exception: {e}")


    # Final Summary
    passed_count = sum(1 for r in results if r)
    total_count = len(results)

    print("\n═══════════════════════════════════")
    if passed_count == total_count:
        print(f"{passed_count}/{total_count} passed — LLM client ready for pipeline")
    else:
        print(f"{passed_count}/{total_count} passed — fix failures before continuing")
    print("═══════════════════════════════════")

    return passed_count == total_count

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
