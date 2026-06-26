"""
Repair an extracted compliance JSON using validation findings.

Usage:
    python -m backend.scripts.repair_json --extracted path/to/extracted.json --validation path/to/validation.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.ingestion.json_repair import repair_extracted_json_from_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair extracted compliance JSON using validator findings.")
    parser.add_argument("--extracted", required=True, help="Path to extracted JSON")
    parser.add_argument("--validation", required=True, help="Path to validation JSON")
    args = parser.parse_args()

    result = repair_extracted_json_from_files(args.extracted, args.validation)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
