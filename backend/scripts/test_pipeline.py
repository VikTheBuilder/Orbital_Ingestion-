"""
ORBITAL Test Pipeline
Runs 7 integration tests using a synthetically generated PDF.
No external PDFs required.
"""

import json
import os
import sys
import tempfile
import traceback

# Ensure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.core.config import get_config
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# Synthetic test PDF content
# ──────────────────────────────────────────────────────────────

TEST_PDF_TEXT = """RBI/2025-26/155
Reserve Bank of India
MASTER DIRECTION ON KYC — TEST CIRCULAR
Date: 01.01.2026

1. Introduction
This circular is issued under Section 35A of the Banking Regulation Act 1949.

2. Customer Risk Categorisation
All Regulated Entities shall, within 60 days of the date of this circular, review and update their customer risk categorisation framework.
Banks must ensure the updated framework is approved by the Board of Directors before implementation.

3. Enhanced Due Diligence
All RE shall conduct Enhanced Due Diligence for customers categorised as High Risk. Transaction pattern analysis is mandatory for all High Risk customers with immediate effect.

Annex I — Risk Categorisation Criteria
Low Risk: Salaried individuals with known sources.
High Risk: PEPs, customers from high-risk countries."""


def create_test_pdf(output_path: str) -> str:
    """Create a synthetic test PDF using pypdf."""
    from pypdf import PdfWriter
    from pypdf._page import PageObject
    from io import BytesIO
    import reportlab

    # Try using reportlab for proper text rendering if available
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # Write test content line by line
        y = height - 50
        for line in TEST_PDF_TEXT.strip().split("\n"):
            if y < 50:
                c.showPage()
                y = height - 50
            c.setFont("Helvetica", 10)
            c.drawString(50, y, line)
            y -= 14

        c.save()
        buffer.seek(0)

        with open(output_path, "wb") as f:
            f.write(buffer.read())

        return output_path

    except ImportError:
        # Fallback: use pypdf PdfWriter with basic page
        writer = PdfWriter()

        # Create a minimal PDF with text using pypdf's lower-level API
        # Since pypdf's PdfWriter doesn't easily create pages with text,
        # we'll create a PDF using the raw PDF content stream
        from pypdf.generic import (
            ArrayObject,
            DecodedStreamObject,
            DictionaryObject,
            NameObject,
            NumberObject,
            TextStringObject,
            create_string_object,
        )

        # Build a simple PDF page with text content
        lines = TEST_PDF_TEXT.strip().split("\n")
        content_lines = []
        y = 750
        content_lines.append("BT")
        content_lines.append("/F1 10 Tf")
        for line in lines:
            # Escape special PDF characters
            escaped = (
                line.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
            )
            content_lines.append(f"1 0 0 1 50 {y} Tm")
            content_lines.append(f"({escaped}) Tj")
            y -= 14
            if y < 50:
                y = 750

        content_lines.append("ET")
        content_stream = "\n".join(content_lines)

        # Create font dictionary
        font_dict = DictionaryObject()
        font_dict.update(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )

        resources = DictionaryObject()
        font_res = DictionaryObject()
        font_res[NameObject("/F1")] = font_dict
        resources[NameObject("/Font")] = font_res

        # Create page
        page = PageObject.create_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = resources

        # Add content stream
        stream = DecodedStreamObject()
        stream.set_data(content_stream.encode("latin-1"))
        page[NameObject("/Contents")] = stream

        writer.add_page(page)
        with open(output_path, "wb") as f:
            writer.write(f)

        return output_path


def run_tests():
    """Run all 7 integration tests."""
    print("\n═══════════════════════════════════")
    print("ORBITAL INGESTION — TEST SUITE")
    print("═══════════════════════════════════\n")

    config = get_config()
    results = []
    test_dir = tempfile.mkdtemp(prefix="orbital_test_")
    test_pdf_path = os.path.join(test_dir, "rbi_kyc_test.pdf")

    # Clean up any previous test artifacts
    structured_test_dir = os.path.join(config.STRUCTURED_DATA_PATH, "RBI")
    finetune_path = os.path.join(config.FINETUNE_DATA_PATH, "raw_pairs.jsonl")

    try:
        # Create synthetic test PDF
        print("Creating synthetic test PDF... ", end="")
        try:
            create_test_pdf(test_pdf_path)
            print("✓\n")
        except Exception as e:
            print(f"✗ — {e}")
            print(f"\nFull traceback:\n{traceback.format_exc()}")
            print("\nTrying alternative PDF creation method...")
            _create_minimal_pdf(test_pdf_path)
            print("✓ (minimal method)\n")

        # ── Test 1: Format Detection ──
        print("Test 1 — Format Detection")
        try:
            from backend.ingestion.format_detector import detect_format

            format_result = detect_format(test_pdf_path)
            assertions = [
                (format_result["is_digital"] == True, "is_digital should be True"),
                (format_result["page_count"] >= 1, f"page_count={format_result['page_count']} should be >= 1"),
                (format_result["language"] == "en", f"language='{format_result['language']}' should be 'en'"),
            ]
            passed = all(a[0] for a in assertions)
            results.append(passed)
            icon = "✓" if passed else "✗"
            print(f"  {icon} is_digital={format_result['is_digital']}, "
                  f"page_count={format_result['page_count']}, "
                  f"language={format_result['language']}")
            for ok, msg in assertions:
                if not ok:
                    print(f"    FAIL: {msg}")
        except Exception as e:
            results.append(False)
            print(f"  ✗ Exception: {e}")
            traceback.print_exc()

        # ── Test 2: Text Extraction ──
        print("\nTest 2 — Text Extraction")
        try:
            from backend.ingestion.ocr import extract_text

            text_result = extract_text(test_pdf_path, format_result)
            assertions = [
                (text_result["total_chars"] > 100, f"total_chars={text_result['total_chars']} should be > 100"),
                (len(text_result["pages"]) >= 1, f"pages count={len(text_result['pages'])} should be >= 1"),
                ("RBI" in text_result["full_text"], "full_text should contain 'RBI'"),
            ]
            passed = all(a[0] for a in assertions)
            results.append(passed)
            icon = "✓" if passed else "✗"
            print(f"  {icon} total_chars={text_result['total_chars']}, "
                  f"pages={len(text_result['pages'])}, "
                  f"method={text_result['extraction_method']}")
            for ok, msg in assertions:
                if not ok:
                    print(f"    FAIL: {msg}")
        except Exception as e:
            results.append(False)
            print(f"  ✗ Exception: {e}")
            traceback.print_exc()

        # ── Test 3: Structure Extraction ──
        print("\nTest 3 — Structure Extraction")
        try:
            from backend.ingestion.structure_extractor import extract_structure

            doc_structure = extract_structure(
                full_text=text_result["full_text"],
                pages=text_result["pages"],
                source="RBI",
                pdf_path=test_pdf_path,
            )
            assertions = [
                (doc_structure.circular_number is not None,
                 f"circular_number={doc_structure.circular_number} should not be None"),
                (doc_structure.date is not None,
                 f"date={doc_structure.date} should not be None"),
                (len(doc_structure.sections) >= 2,
                 f"sections={len(doc_structure.sections)} should be >= 2"),
                (len(doc_structure.annexures) >= 1 or any("annex" in s.heading.lower() for s in doc_structure.sections),
                 "should have annexures or annex section"),
            ]
            passed = all(a[0] for a in assertions)
            results.append(passed)
            icon = "✓" if passed else "✗"
            print(f"  {icon} circular_number={doc_structure.circular_number}, "
                  f"date={doc_structure.date}, "
                  f"sections={len(doc_structure.sections)}, "
                  f"annexures={len(doc_structure.annexures)}")
            for ok, msg in assertions:
                if not ok:
                    print(f"    FAIL: {msg}")
        except Exception as e:
            results.append(False)
            print(f"  ✗ Exception: {e}")
            traceback.print_exc()

        # ── Test 4: Obligation Extraction ──
        print("\nTest 4 — Obligation Extraction")
        try:
            from backend.ingestion.obligation_extractor import extract_obligations

            obligations = extract_obligations(doc_structure)
            assertions = [
                (len(obligations) >= 2,
                 f"obligations count={len(obligations)} should be >= 2"),
                (all(o.confidence >= 0.65 for o in obligations),
                 "all obligations should have confidence >= 0.65"),
                (any(o.domain == "KYC" for o in obligations),
                 "should have at least one KYC obligation"),
                (any(o.severity in ["high", "critical"] for o in obligations),
                 "should have at least one high/critical obligation"),
            ]
            passed = all(a[0] for a in assertions)
            results.append(passed)
            icon = "✓" if passed else "✗"
            domains = set(o.domain for o in obligations)
            severities = set(o.severity for o in obligations)
            print(f"  {icon} obligations={len(obligations)}, "
                  f"domains={domains}, "
                  f"severities={severities}")
            for ok, msg in assertions:
                if not ok:
                    print(f"    FAIL: {msg}")

            # Print obligation details
            for ob in obligations:
                print(f"    [{ob.severity:>8}] [{ob.domain:>10}] "
                      f"{ob.actor}: {ob.action[:60]}... "
                      f"(conf={ob.confidence})")
        except Exception as e:
            results.append(False)
            print(f"  ✗ Exception: {e}")
            traceback.print_exc()

        # ── Test 5: Chunking ──
        print("\nTest 5 — Chunking")
        try:
            from backend.ingestion.chunker import chunk_document

            chunks = chunk_document(doc_structure, config)
            assertions = [
                (len(chunks) >= 1,
                 f"chunks count={len(chunks)} should be >= 1"),
                (all(len(c.text) >= 100 for c in chunks),
                 "all chunks should have text >= 100 chars"),
                (all(c.doc_id == doc_structure.doc_id for c in chunks),
                 "all chunks should have matching doc_id"),
            ]
            passed = all(a[0] for a in assertions)
            results.append(passed)
            icon = "✓" if passed else "✗"
            print(f"  {icon} chunks={len(chunks)}")
            for c in chunks:
                print(f"    {c.chunk_id}: {c.section_heading[:40]} "
                      f"({len(c.text)} chars, {c.chunk_type})")
            for ok, msg in assertions:
                if not ok:
                    print(f"    FAIL: {msg}")
        except Exception as e:
            results.append(False)
            print(f"  ✗ Exception: {e}")
            traceback.print_exc()

        # ── Test 6: Full Pipeline ──
        print("\nTest 6 — Full Pipeline")
        try:
            from backend.ingestion.pipeline import run_pipeline

            # Clean up any previous test output to ensure fresh run
            _cleanup_test_output(config, doc_structure.doc_id if 'doc_structure' in dir() else None)

            pipeline_result = run_pipeline(test_pdf_path, "RBI")
            assertions = [
                (pipeline_result.status in ["success", "partial"],
                 f"status='{pipeline_result.status}' should be success/partial"),
                (pipeline_result.total_obligations >= 2,
                 f"total_obligations={pipeline_result.total_obligations} should be >= 2"),
                (pipeline_result.total_chunks >= 1,
                 f"total_chunks={pipeline_result.total_chunks} should be >= 1"),
                (os.path.exists(pipeline_result.structured_json_path),
                 f"structured JSON should exist at {pipeline_result.structured_json_path}"),
                (os.path.exists(pipeline_result.finetune_pairs_path),
                 f"finetune JSONL should exist at {pipeline_result.finetune_pairs_path}"),
            ]
            passed = all(a[0] for a in assertions)
            results.append(passed)
            icon = "✓" if passed else "✗"
            print(f"  {icon} status={pipeline_result.status}, "
                  f"obligations={pipeline_result.total_obligations}, "
                  f"chunks={pipeline_result.total_chunks}, "
                  f"time={pipeline_result.processing_time_seconds}s")
            if pipeline_result.warnings:
                for w in pipeline_result.warnings:
                    print(f"    ⚠ {w}")
            for ok, msg in assertions:
                if not ok:
                    print(f"    FAIL: {msg}")
        except Exception as e:
            results.append(False)
            print(f"  ✗ Exception: {e}")
            traceback.print_exc()

        # ── Test 7: Idempotency ──
        print("\nTest 7 — Idempotency")
        try:
            from backend.ingestion.pipeline import run_pipeline

            # Count lines in finetune file before second run
            lines_before = 0
            if os.path.exists(finetune_path):
                with open(finetune_path, "r", encoding="utf-8") as f:
                    lines_before = sum(1 for _ in f)

            result2 = run_pipeline(test_pdf_path, "RBI")

            # Count lines after second run
            lines_after = 0
            if os.path.exists(finetune_path):
                with open(finetune_path, "r", encoding="utf-8") as f:
                    lines_after = sum(1 for _ in f)

            assertions = [
                (pipeline_result.doc_id == result2.doc_id,
                 f"doc_id mismatch: {pipeline_result.doc_id} != {result2.doc_id}"),
                (lines_after == lines_before,
                 f"finetune lines increased: {lines_before} → {lines_after} (should be same)"),
            ]
            passed = all(a[0] for a in assertions)
            results.append(passed)
            icon = "✓" if passed else "✗"
            print(f"  {icon} doc_id match={pipeline_result.doc_id == result2.doc_id}, "
                  f"finetune lines: {lines_before} → {lines_after}")
            for ok, msg in assertions:
                if not ok:
                    print(f"    FAIL: {msg}")
        except Exception as e:
            results.append(False)
            print(f"  ✗ Exception: {e}")
            traceback.print_exc()

    finally:
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(test_dir, ignore_errors=True)
        except Exception:
            pass

    # ── Final Summary ──
    passed_count = sum(1 for r in results if r)
    total_count = len(results)

    print(f"\n═══════════════════════════════════")
    if passed_count == total_count:
        print(f"{passed_count}/{total_count} tests passed — ingestion layer ready")
    else:
        print(f"{passed_count}/{total_count} tests passed")
        failed_tests = [i + 1 for i, r in enumerate(results) if not r]
        print(f"Failed tests: {failed_tests}")
    print(f"═══════════════════════════════════\n")

    return passed_count == total_count


def _cleanup_test_output(config, doc_id: str = None):
    """Remove test output files to ensure clean test runs."""
    import shutil

    finetune_path = os.path.join(config.FINETUNE_DATA_PATH, "raw_pairs.jsonl")
    if os.path.exists(finetune_path):
        os.remove(finetune_path)

    if doc_id:
        structured_path = os.path.join(config.STRUCTURED_DATA_PATH, "RBI", f"{doc_id}.json")
        if os.path.exists(structured_path):
            os.remove(structured_path)


def _create_minimal_pdf(output_path: str):
    """Create a minimal PDF with embedded text using pure pypdf."""
    # Build a raw PDF file manually as a fallback
    lines = TEST_PDF_TEXT.strip().split("\n")
    y = 750
    text_ops = []
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        text_ops.append(f"BT /F1 10 Tf 1 0 0 1 50 {y} Tm ({escaped}) Tj ET")
        y -= 14
        if y < 50:
            y = 750

    stream_content = "\n".join(text_ops)
    stream_length = len(stream_content)

    pdf_content = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj

4 0 obj
<< /Length {stream_length} >>
stream
{stream_content}
endstream
endobj

5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
{str(307 + stream_length).zfill(10)} 00000 n 

trailer
<< /Size 6 /Root 1 0 R >>
startxref
{367 + stream_length}
%%EOF"""

    with open(output_path, "wb") as f:
        f.write(pdf_content.encode("latin-1"))


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
