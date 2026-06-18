"""
ORBITAL Ingestion Pipeline
Orchestrates the full PDF → JSON extraction pipeline.

Architecture:
  PDF → Format Detection → OCR → Structure Extraction (Rule Engine V1)
      → Clause Classification → Obligation Extraction (Rule Engine V1 + selective LLM)
      → Rule Engine V2 Re-score → Confidence Gating → Validation
      → Chunking → Save Structured JSON

Fine-tuning pair generation is DECOUPLED — call generate_finetune_pairs()
separately (async job, CLI, or post-processing batch).
"""

import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

from backend.core.config import get_config
from backend.core.logger import get_logger
from backend.core.llm_client import llm
from backend.ingestion.chunker import chunk_document
from backend.ingestion.format_detector import detect_format
from backend.ingestion.ocr import extract_text
from backend.ingestion.obligation_extractor import extract_obligations
from backend.ingestion.schemas import PipelineResultSchema, ValidationResultSchema
from backend.ingestion.structure_extractor import extract_structure
from backend.ingestion.validator import validate_extraction

logger = get_logger(__name__)

VALID_SOURCES = {"RBI", "SEBI", "CERT-In", "NPCI", "IRDAI", "DPDP", "FIU-IND", "IBA", "OTHER"}


def _detect_source_from_filename(filename: str) -> str:
    """Detect a regulatory source from the PDF filename."""
    lower = (filename or "").lower()
    if "rbi" in lower:
        return "RBI"
    if "sebi" in lower:
        return "SEBI"
    if "cert" in lower:
        return "CERT-In"
    if "dpdp" in lower or "meity" in lower:
        return "DPDP"
    if "fiu" in lower:
        return "FIU-IND"
    if "npci" in lower:
        return "NPCI"
    if "irdai" in lower or "insurance" in lower:
        return "IRDAI"
    if "iba" in lower:
        return "IBA"
    return "OTHER"


def _detect_source_from_text(text: str) -> str:
    """Detect a regulatory source from OCR/text content."""
    lower = (text or "")[:500].lower()
    if "rbi/" in lower or "reserve bank of india" in lower:
        return "RBI"
    if "sebi/" in lower or "securities and exchange board" in lower:
        return "SEBI"
    if "cert-in" in lower or "cert-in.org" in lower:
        return "CERT-In"
    if "dpdp" in lower or "digital personal data" in lower:
        return "DPDP"
    if "fiu-ind" in lower or "financial intelligence unit" in lower:
        return "FIU-IND"
    if "npci" in lower or "national payments corporation" in lower:
        return "NPCI"
    if "irdai" in lower or "insurance regulatory and development authority" in lower:
        return "IRDAI"
    if "iba" in lower or "indian banks association" in lower:
        return "IBA"
    return "OTHER"


def run_pipeline(pdf_path: str, source: str) -> PipelineResultSchema:
    """
    Run the full ingestion pipeline on a single PDF document.

    Args:
        pdf_path: Path to the PDF file.
        source   : Regulatory source (RBI, SEBI, CERT-In, ..., or "auto").

    Returns:
        PipelineResultSchema summarising the extraction results.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
    """
    config = get_config()
    warnings_list: list[str] = []
    start_time = time.time()

    try:
        # ── Step 1: Validate PDF path ──────────────────────────────────────
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # ── Step 2: Resolve source ─────────────────────────────────────────
        if source == "auto":
            source = _detect_source_from_filename(os.path.basename(pdf_path))

        if source not in VALID_SOURCES:
            logger.warning("Unknown source, defaulting to OTHER", provided_source=source)
            source = "OTHER"

        logger.info("Pipeline started", filename=os.path.basename(pdf_path), source=source)

        # ── Step 3: Format detection ───────────────────────────────────────
        format_info = detect_format(pdf_path)
        logger.info(
            "Format detection result",
            is_scanned=format_info["is_scanned"],
            language=format_info["language"],
            page_count=format_info["page_count"],
        )

        # ── Step 4: Text extraction / OCR ──────────────────────────────────
        text_result = extract_text(pdf_path, format_info)
        logger.info(
            "Text extraction result",
            total_chars=text_result["total_chars"],
            extraction_method=text_result["extraction_method"],
        )
        if text_result["total_chars"] < 100:
            warnings_list.append("Low text extraction — document may be heavily scanned")

        # ── Step 5: Source detection from extracted text ───────────────────
        if source == "OTHER":
            detected = _detect_source_from_text(text_result["full_text"])
            if detected != "OTHER":
                source = detected
                logger.info("Source detected from extracted text", source=source)

        # ── Step 6: Structure extraction + Rule Engine V1 (metadata) ──────
        doc_structure = extract_structure(
            full_text=text_result["full_text"],
            pages=text_result["pages"],
            source=source,
            pdf_path=pdf_path,
        )
        logger.info(
            "Structure extraction result",
            sections=len(doc_structure.sections),
            tables=len(doc_structure.tables),
        )
        if len(doc_structure.sections) == 0:
            warnings_list.append("No sections detected — check document format")

        # ── Step 7: Idempotency check ──────────────────────────────────────
        structured_dir = os.path.join(config.STRUCTURED_DATA_PATH, source)
        structured_path = os.path.join(structured_dir, f"{doc_structure.doc_id}.json")
        finetune_path = os.path.join(config.FINETUNE_DATA_PATH, "raw_pairs.jsonl")

        if os.path.isfile(structured_path):
            logger.warning(
                "Document already processed — skipping duplicate",
                doc_id=doc_structure.doc_id,
                path=structured_path,
            )
            try:
                with open(structured_path, "r", encoding="utf-8") as f:
                    prev_data = json.load(f)
                processing_time = time.time() - start_time
                prev_validation = prev_data.get("validation") or {}
                return PipelineResultSchema(
                    doc_id=doc_structure.doc_id,
                    source=source,
                    title=prev_data.get("title", doc_structure.title),
                    total_pages=prev_data.get("total_pages", 0),
                    total_sections=len(prev_data.get("sections", [])),
                    total_obligations=len(prev_data.get("obligations", [])),
                    obligations_by_domain=_count_field(prev_data.get("obligations", []), "domain"),
                    obligations_by_severity=_count_field(prev_data.get("obligations", []), "severity"),
                    total_chunks=0,
                    processing_time_seconds=round(processing_time, 2),
                    structured_json_path=structured_path,
                    finetune_pairs_path=finetune_path,
                    status="success",
                    warnings=["Document already processed — returned cached result"],
                    validation_missed_count=len(prev_validation.get("missed_obligations", [])),
                    validation_incorrect_count=len(prev_validation.get("incorrect_extractions", [])),
                    validation_confidence=float(prev_validation.get("overall_confidence", 0.0)),
                )
            except Exception:
                pass  # Re-process if cached read fails

        # ── Step 8: Obligation extraction (Rule Engine V1 + selective LLM) ─
        obligations = extract_obligations(doc_structure)
        doc_structure.obligations = obligations

        domain_counts = dict(Counter(o.domain for o in obligations))
        severity_counts = dict(Counter(o.severity for o in obligations))
        logger.info(
            "Obligation extraction result",
            total=len(obligations),
            by_domain=domain_counts,
            by_severity=severity_counts,
        )

        # ── Step 9: Validation ─────────────────────────────────────────────
        # Only run LLM validation when we actually extracted something and the
        # document has enough text to be worth reviewing.
        if len(obligations) > 0 and text_result["total_chars"] > 500:
            validation_result: ValidationResultSchema = validate_extraction(
                raw_text=text_result["full_text"],
                obligations=obligations,
                doc_effective_date=doc_structure.effective_date,
            )
        else:
            validation_result = ValidationResultSchema(
                overall_confidence=0.0,
                validation_notes="Validation skipped — no obligations or insufficient text.",
            )

        doc_structure.validation = validation_result.model_dump(mode="json")
        logger.info(
            "Extraction validation complete",
            missed=len(validation_result.missed_obligations),
            incorrect=len(validation_result.incorrect_extractions),
            overall_confidence=validation_result.overall_confidence,
        )
        if validation_result.missed_obligations:
            warnings_list.append(
                f"Validation found {len(validation_result.missed_obligations)} potentially missed obligation(s)"
            )
        if validation_result.missing_effective_date and not doc_structure.effective_date:
            warnings_list.append(
                f"Effective date not captured: {validation_result.missing_effective_date}"
            )

        # ── Step 10: Chunking ──────────────────────────────────────────────
        chunks = chunk_document(doc_structure, config)
        logger.info("Chunking result", total_chunks=len(chunks))

        # ── Step 11: Save structured JSON ──────────────────────────────────
        os.makedirs(structured_dir, exist_ok=True)
        doc_dict = doc_structure.model_dump(mode="json")
        with open(structured_path, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False)
        logger.info("Structured JSON saved", path=structured_path)

        # ── Step 12: Fine-tuning pair generation (OPTIONAL, inline) ────────
        # NOTE: For production use, call generate_finetune_pairs() from a
        # background worker / separate CLI job to avoid blocking the pipeline.
        pairs_written = 0
        pairs_dropped = 0
        if config.GENERATE_FINETUNE_PAIRS:
            pairs_written, pairs_dropped = generate_finetune_pairs(
                doc_structure=doc_structure,
                finetune_path=finetune_path,
            )
            logger.info("Fine-tune pairs saved", pairs=pairs_written, dropped=pairs_dropped, path=finetune_path)

        # ── Step 13: Return result ─────────────────────────────────────────
        processing_time = round(time.time() - start_time, 2)
        status = "success"
        if warnings_list:
            status = "partial"
        if len(obligations) == 0:
            status = "partial"
            warnings_list.append("No obligations extracted")

        return PipelineResultSchema(
            doc_id=doc_structure.doc_id,
            source=source,
            title=doc_structure.title,
            total_pages=doc_structure.total_pages,
            total_sections=len(doc_structure.sections),
            total_obligations=len(obligations),
            obligations_by_domain=domain_counts,
            obligations_by_severity=severity_counts,
            total_chunks=len(chunks),
            processing_time_seconds=processing_time,
            structured_json_path=structured_path,
            finetune_pairs_path=finetune_path,
            finetune_dropped_count=pairs_dropped,
            status=status,
            warnings=warnings_list,
            validation_missed_count=len(validation_result.missed_obligations),
            validation_incorrect_count=len(validation_result.incorrect_extractions),
            validation_confidence=validation_result.overall_confidence,
        )

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(
            "Pipeline failed with uncaught exception",
            error=str(e),
            traceback=traceback.format_exc(),
        )
        processing_time = round(time.time() - start_time, 2)
        return PipelineResultSchema(
            doc_id="error",
            source=source,
            title="",
            total_pages=0,
            total_sections=0,
            total_obligations=0,
            obligations_by_domain={},
            obligations_by_severity={},
            total_chunks=0,
            processing_time_seconds=processing_time,
            structured_json_path="",
            finetune_pairs_path="",
            status="failed",
            warnings=[f"Pipeline failed: {str(e)}"],
        )


def generate_finetune_pairs(doc_structure, finetune_path: str) -> int:
    """
    Generate fine-tuning pairs from a processed DocumentStructureSchema.

    This function is intentionally SEPARATE from run_pipeline() so it can be:
      - Called from a background worker after the main pipeline completes
      - Batched overnight across many documents
      - Skipped entirely in high-throughput ingestion mode

    Returns a tuple: (pairs_written, dropped_pairs)
    """
    config = get_config()
    os.makedirs(os.path.dirname(finetune_path) or ".", exist_ok=True)
    pairs_written = 0
    dropped_pairs = 0

    try:
        with open(finetune_path, "a", encoding="utf-8") as f:
            for obligation in doc_structure.obligations:
                # Find the section text for this obligation
                section_text = ""
                section_heading = obligation.clause_number
                for section in doc_structure.sections:
                    if section.id == obligation.section_id:
                        section_text = section.text
                        section_heading = section.heading
                        break

                # Only generate pairs for obligations with sufficient confidence
                # to avoid training on low-quality extractions
                if obligation.confidence < config.MIN_OBLIGATION_CONFIDENCE:
                    continue

                # Type 1: obligation_extraction
                fail_before = llm.parse_failures
                ext_output = llm.extract_obligations(section_text)
                if llm.parse_failures > fail_before:
                    dropped_pairs += 1
                elif ext_output:
                    pair1 = {
                        "type": "obligation_extraction",
                        "instruction": "Extract compliance obligations from the following Indian regulatory text.",
                        "input": f"{section_heading}\n\n{section_text}",
                        "output": json.dumps(ext_output, ensure_ascii=False),
                        "source": "llm_enhanced",
                        "doc_id": doc_structure.doc_id,
                        "domain": obligation.domain,
                        "quality_score": obligation.confidence,
                    }
                    f.write(json.dumps(pair1, ensure_ascii=False) + "\n")
                    pairs_written += 1

                # Type 2: domain_classification
                fail_before = llm.parse_failures
                dom_output = llm.classify_domain(obligation.action)
                if llm.parse_failures > fail_before:
                    dropped_pairs += 1
                elif dom_output:
                    pair2 = {
                        "type": "domain_classification",
                        "instruction": "Classify this regulatory text into its compliance domain.",
                        "input": obligation.action,
                        "output": json.dumps(dom_output, ensure_ascii=False),
                        "source": "llm_enhanced",
                        "doc_id": doc_structure.doc_id,
                        "domain": obligation.domain,
                        "quality_score": round(obligation.confidence * 0.9, 4),
                    }
                    f.write(json.dumps(pair2, ensure_ascii=False) + "\n")
                    pairs_written += 1

                # Type 3: task_generation
                fail_before = llm.parse_failures
                task_output = llm.generate_map_card(obligation.model_dump(mode="json"))
                if llm.parse_failures > fail_before:
                    dropped_pairs += 1
                elif task_output:
                    pair3 = {
                        "type": "task_generation",
                        "instruction": "Convert regulatory obligations into MAP cards.",
                        "input": json.dumps(obligation.model_dump(mode="json"), ensure_ascii=False),
                        "output": json.dumps(task_output, ensure_ascii=False),
                        "source": "llm_enhanced",
                        "doc_id": doc_structure.doc_id,
                        "domain": obligation.domain,
                        "quality_score": round(obligation.confidence * 0.8, 4),
                    }
                    f.write(json.dumps(pair3, ensure_ascii=False) + "\n")
                    pairs_written += 1

                # Type 4: severity_assessment
                sev_prompt = (
                    f"Assess severity of: {obligation.action}\n"
                    f'Expected output: {{"severity":"X","justification":"Y"}}'
                )
                sev_output = llm.call("chat_qa", sev_prompt)
                pair4 = {
                    "type": "severity_assessment",
                    "instruction": "Assess severity of the obligation.",
                    "input": obligation.action,
                    "output": sev_output,
                    "source": "llm_enhanced",
                    "doc_id": doc_structure.doc_id,
                    "domain": obligation.domain,
                    "quality_score": round(obligation.confidence * 0.85, 4),
                }
                f.write(json.dumps(pair4, ensure_ascii=False) + "\n")
                pairs_written += 1

    except Exception as e:
        logger.error("Fine-tune pair generation failed", error=str(e))

    return pairs_written, dropped_pairs


def _count_field(items: list, field: str) -> dict:
    """Count occurrences of a field value in a list of dicts."""
    counts: dict[str, int] = {}
    for item in items:
        val = item.get(field, "Unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts
