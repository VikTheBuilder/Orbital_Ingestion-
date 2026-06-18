"""
ORBITAL Fine-Tune Pair Generator (Batch/Async CLI)

Run this SEPARATELY from the main ingestion pipeline to generate LLM training
pairs without blocking document processing.

Usage:
    python -m backend.scripts.generate_finetune --source RBI
    python -m backend.scripts.generate_finetune --doc-id RBI-RBI-2026-27-46
    python -m backend.scripts.generate_finetune --all

The script reads already-structured JSONs from data/structured/, loads them
into DocumentStructureSchema, and calls generate_finetune_pairs() which makes
LLM calls at a controlled rate.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.core.config import get_config
from backend.core.logger import get_logger
from backend.ingestion.pipeline import generate_finetune_pairs
from backend.ingestion.schemas import DocumentStructureSchema

logger = get_logger(__name__)


def _load_doc_structure(json_path: str) -> DocumentStructureSchema | None:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DocumentStructureSchema(**data)
    except Exception as e:
        logger.error("Failed to load structured JSON", path=json_path, error=str(e))
        return None


def main():
    config = get_config()
    parser = argparse.ArgumentParser(
        description="ORBITAL Fine-Tune Pair Generator — run separately from the main pipeline"
    )
    parser.add_argument("--source", type=str, help="Only process a specific source (e.g. RBI)")
    parser.add_argument("--doc-id", type=str, help="Process a single document by doc_id")
    parser.add_argument("--all", action="store_true", help="Process all structured JSONs")
    args = parser.parse_args()

    if not args.source and not args.doc_id and not args.all:
        parser.print_help()
        sys.exit(1)

    structured_root = Path(config.STRUCTURED_DATA_PATH)
    finetune_path = os.path.join(config.FINETUNE_DATA_PATH, "raw_pairs.jsonl")
    total_pairs = 0
    total_dropped = 0

    if args.doc_id:
        # Find the JSON anywhere under structured/
        matches = list(structured_root.rglob(f"{args.doc_id}.json"))
        if not matches:
            print(f"Document not found: {args.doc_id}")
            sys.exit(1)
        json_files = matches
    elif args.source:
        source_dir = structured_root / args.source
        json_files = list(source_dir.glob("*.json")) if source_dir.exists() else []
    else:
        json_files = list(structured_root.rglob("*.json"))

    print(f"\nGenerating fine-tune pairs for {len(json_files)} document(s)...\n")

    for json_path in json_files:
        doc = _load_doc_structure(str(json_path))
        if not doc:
            continue
        pairs, dropped = generate_finetune_pairs(doc_structure=doc, finetune_path=finetune_path)
        print(f"  {json_path.stem}: {pairs} pairs written, {dropped} dropped")
        total_pairs += pairs
        total_dropped += dropped

    print(f"\nTotal fine-tune pairs written: {total_pairs} (dropped: {total_dropped})")
    print(f"Saved to: {finetune_path}")


if __name__ == "__main__":
    main()
