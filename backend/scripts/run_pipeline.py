"""
ORBITAL CLI — Run the ingestion pipeline on PDFs.

Usage:
    python -m backend.scripts.run_pipeline --file path/to/doc.pdf --source RBI
    python -m backend.scripts.run_pipeline --folder data/raw/rbi/ --source auto
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.core.logger import get_logger
from backend.ingestion.pipeline import run_pipeline

logger = get_logger(__name__)


def detect_source_from_filename(filename: str) -> str:
    """Auto-detect regulatory source from filename."""
    lower = filename.lower()
    if "rbi" in lower:
        return "RBI"
    elif "sebi" in lower:
        return "SEBI"
    elif "cert" in lower:
        return "CERT-In"
    elif "dpdp" in lower or "meity" in lower:
        return "DPDP"
    elif "fiu" in lower:
        return "FIU-IND"
    elif "npci" in lower:
        return "NPCI"
    elif "irdai" in lower:
        return "IRDAI"
    else:
        return "OTHER"


def process_single_file(pdf_path: str, source: str, verbose: bool = False):
    """Process a single PDF file and print results."""
    result = run_pipeline(pdf_path, source)

    status_icon = "✓" if result.status in ("success", "partial") else "✗"
    severity_summary = ", ".join(
        f"{count} {sev}"
        for sev, count in sorted(result.obligations_by_severity.items())
    )

    validation_summary = (
        f"validation: {result.validation_missed_count} missed, "
        f"{result.validation_incorrect_count} incorrect, "
        f"confidence {result.validation_confidence:.2f}"
    ) if result.validation_confidence > 0 else "validation: skipped"

    print(
        f"  {os.path.basename(pdf_path)} → "
        f"{result.total_obligations} obligations "
        f"({severity_summary}) — "
        f"{result.processing_time_seconds}s {status_icon}"
    )
    print(f"    {validation_summary}")

    if result.warnings:
        for w in result.warnings:
            print(f"    ⚠ {w}")

    if verbose and result.total_obligations > 0:
        print("\n  Obligations:")
        # Re-load obligations from structured JSON
        import json
        try:
            with open(result.structured_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for ob in data.get("obligations", []):
                print(
                    f"    [{ob['severity'].upper():>8}] "
                    f"[{ob['domain']:>10}] "
                    f"{ob['actor']}: {ob['action'][:80]}"
                )
        except Exception:
            pass
        print()

    return result


def process_folder(folder_path: str, source: str, verbose: bool = False):
    """Process all PDFs in a folder recursively."""
    pdf_files = sorted(Path(folder_path).rglob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return

    total_files = len(pdf_files)
    print(f"\nFound {total_files} PDF file(s) in {folder_path}\n")

    all_results = []
    total_obligations = 0
    total_pairs = 0
    total_dropped = 0
    domain_counter = Counter()
    severity_counter = Counter()

    for idx, pdf_file in enumerate(pdf_files, 1):
        file_source = source
        if source == "auto":
            file_source = detect_source_from_filename(pdf_file.name)

        print(f"[{idx}/{total_files}] ", end="")
        result = process_single_file(str(pdf_file), file_source, verbose)
        all_results.append(result)

        total_obligations += result.total_obligations
        total_pairs += result.total_obligations * 2  # 2 pairs per obligation (roughly)
        total_dropped += getattr(result, 'finetune_dropped_count', 0)
        for domain, count in result.obligations_by_domain.items():
            domain_counter[domain] += count
        for severity, count in result.obligations_by_severity.items():
            severity_counter[severity] += count

    # Print final summary
    finetune_path = all_results[-1].finetune_pairs_path if all_results else "N/A"

    print("\n═══════════════════════════════════")
    print("ORBITAL INGESTION COMPLETE")
    print("═══════════════════════════════════")
    print(f"Files processed  : {total_files}")
    print(f"Total obligations: {total_obligations}")
    print("By domain:")
    for domain, count in domain_counter.most_common():
        print(f"  {domain:<15}: {count}")
    print(f"Fine-tune pairs  : {total_pairs} (dropped: {total_dropped})")
    print(f"Saved to         : {finetune_path}")
    print("═══════════════════════════════════")


def main():
    parser = argparse.ArgumentParser(
        description="ORBITAL Ingestion Pipeline — Process regulatory PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single PDF file to process",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to a folder of PDFs to process recursively",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["RBI", "SEBI", "CERT-In", "NPCI", "IRDAI", "DPDP", "FIU-IND", "IBA", "auto"],
        help="Regulatory source (default: auto-detect from filename)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full obligation list after each file",
    )

    args = parser.parse_args()

    if not args.file and not args.folder:
        parser.print_help()
        print("\nError: Please provide either --file or --folder")
        sys.exit(1)

    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)

        source = args.source
        if source == "auto":
            source = detect_source_from_filename(os.path.basename(args.file))

        print(f"\nProcessing: {args.file} (source: {source})\n")
        process_single_file(args.file, source, args.verbose)

    elif args.folder:
        if not os.path.isdir(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            sys.exit(1)

        process_folder(args.folder, args.source, args.verbose)


if __name__ == "__main__":
    main()
