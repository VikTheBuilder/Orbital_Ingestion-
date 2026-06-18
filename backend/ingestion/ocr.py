"""
ORBITAL OCR & Text Extraction
Extracts text from PDFs using pypdf (digital) and surya-ocr (scanned).
"""

from backend.core.logger import get_logger

logger = get_logger(__name__)


def extract_text(pdf_path: str, format_info: dict) -> dict:
    """
    Extract text from a PDF using the appropriate method based on format_info.

    Args:
        pdf_path: Path to the PDF file.
        format_info: Dict returned by format_detector.detect_format().

    Returns:
        Dict with keys: pages, full_text, language, extraction_method,
        total_chars, total_pages.
    """
    result = {
        "pages": [],
        "full_text": "",
        "language": format_info.get("language", "en"),
        "extraction_method": "pypdf",
        "total_chars": 0,
        "total_pages": 0,
    }

    try:
        from pypdf import PdfReader

        logger.info(
            "Text extraction started",
            pdf_path=pdf_path,
            is_digital=format_info.get("is_digital"),
            is_scanned=format_info.get("is_scanned"),
            is_mixed=format_info.get("is_mixed"),
        )

        reader = PdfReader(pdf_path)
        pages = []

        if format_info.get("is_digital", True):
            # Pure digital — use pypdf for all pages
            result["extraction_method"] = "pypdf"
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append({
                    "page_number": i + 1,
                    "text": text,
                    "method": "pypdf",
                    "char_count": len(text),
                })

        elif format_info.get("is_scanned", False):
            # Pure scanned — try surya, fall back to pypdf
            method = "pypdf_fallback"
            try:
                from surya.recognition import RecognitionPredictor
                from surya.detection import DetectionPredictor
                method = "surya"
                logger.info("Using surya OCR for scanned document")
            except ImportError:
                logger.warning(
                    "surya-ocr not installed — falling back to pypdf. "
                    "Scanned pages will return empty or minimal text. "
                    "Install surya-ocr for proper OCR support."
                )

            if method == "surya":
                pages = _extract_with_surya(pdf_path, reader)
            else:
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if len(text.strip()) < 10:
                        logger.warning(
                            "Scanned page returned minimal text via pypdf",
                            page_number=i + 1,
                            char_count=len(text),
                        )
                    pages.append({
                        "page_number": i + 1,
                        "text": text,
                        "method": "pypdf_fallback",
                        "char_count": len(text),
                    })

            result["extraction_method"] = method

        elif format_info.get("is_mixed", False):
            # Mixed — pypdf for digital pages, surya for scanned pages
            scanned_set = set(format_info.get("scanned_pages", []))
            surya_available = False

            try:
                from surya.recognition import RecognitionPredictor
                from surya.detection import DetectionPredictor
                surya_available = True
                logger.info("Using mixed extraction: pypdf + surya")
            except ImportError:
                logger.warning(
                    "surya-ocr not installed — scanned pages will use pypdf fallback"
                )

            for i, page in enumerate(reader.pages):
                page_num = i + 1
                if page_num in scanned_set and surya_available:
                    # Use surya for this page
                    surya_pages = _extract_with_surya(pdf_path, reader, pages_to_extract=[i])
                    if surya_pages:
                        pages.append(surya_pages[0])
                    else:
                        text = page.extract_text() or ""
                        pages.append({
                            "page_number": page_num,
                            "text": text,
                            "method": "pypdf_fallback",
                            "char_count": len(text),
                        })
                else:
                    text = page.extract_text() or ""
                    pages.append({
                        "page_number": page_num,
                        "text": text,
                        "method": "pypdf",
                        "char_count": len(text),
                    })

            result["extraction_method"] = "mixed"

        else:
            # Default fallback — treat as digital
            result["extraction_method"] = "pypdf"
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append({
                    "page_number": i + 1,
                    "text": text,
                    "method": "pypdf",
                    "char_count": len(text),
                })

        result["pages"] = pages
        result["full_text"] = "\n\n".join(p["text"] for p in pages)
        result["total_chars"] = sum(p["char_count"] for p in pages)
        result["total_pages"] = len(pages)

        logger.info(
            "Text extraction complete",
            extraction_method=result["extraction_method"],
            total_chars=result["total_chars"],
            total_pages=result["total_pages"],
        )

    except Exception as e:
        logger.error("Text extraction failed", error=str(e), pdf_path=pdf_path)

    return result


def _extract_with_surya(pdf_path: str, reader, pages_to_extract: list = None) -> list:
    """
    Extract text from scanned pages using surya-ocr.

    Args:
        pdf_path: Path to the PDF file.
        reader: pypdf PdfReader instance.
        pages_to_extract: Optional list of 0-indexed page numbers to extract.
                          If None, extracts all pages.

    Returns:
        List of page dicts with text extracted via surya.
    """
    pages = []
    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        from PIL import Image
        import io

        rec_predictor = RecognitionPredictor()
        det_predictor = DetectionPredictor()

        indices = pages_to_extract if pages_to_extract is not None else range(len(reader.pages))

        for i in indices:
            page = reader.pages[i]
            # Try to extract images from the page for OCR
            text_parts = []

            # Attempt to get page as image
            # pypdf doesn't directly render pages to images, so we extract embedded images
            if hasattr(page, "images") and page.images:
                for img in page.images:
                    try:
                        image = Image.open(io.BytesIO(img.data))
                        # Use surya for recognition
                        predictions = rec_predictor([image])
                        for pred in predictions:
                            if hasattr(pred, "text_lines"):
                                for line in pred.text_lines:
                                    text_parts.append(line.text)
                    except Exception:
                        continue

            text = "\n".join(text_parts)
            pages.append({
                "page_number": i + 1,
                "text": text,
                "method": "surya",
                "char_count": len(text),
            })

    except Exception as e:
        logger.error("Surya OCR extraction failed", error=str(e))

    return pages
