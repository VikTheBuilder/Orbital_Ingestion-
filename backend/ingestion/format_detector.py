"""
ORBITAL Format Detector
Detects whether a PDF is digital, scanned, or mixed, and identifies language.
"""

from backend.core.logger import get_logger

logger = get_logger(__name__)


def detect_format(pdf_path: str) -> dict:
    """
    Analyse a PDF to determine its format characteristics.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with keys: is_scanned, is_digital, is_mixed, language,
        has_tables, has_annexures, page_count, scanned_pages, digital_pages.
    """
    # Safe defaults
    result = {
        "is_scanned": False,
        "is_digital": True,
        "is_mixed": False,
        "language": "en",
        "has_tables": False,
        "has_annexures": False,
        "page_count": 0,
        "scanned_pages": [],
        "digital_pages": [],
    }

    try:
        from pypdf import PdfReader

        logger.info("Format detection started", pdf_path=pdf_path)

        reader = PdfReader(pdf_path)
        page_count = len(reader.pages)
        result["page_count"] = page_count

        all_text = ""
        scanned_pages = []
        digital_pages = []

        for i, page in enumerate(reader.pages):
            page_num = i + 1
            text = page.extract_text() or ""
            char_count = len(text.strip())

            if char_count < 50:
                scanned_pages.append(page_num)
            else:
                digital_pages.append(page_num)

            all_text += text + "\n"

        result["scanned_pages"] = scanned_pages
        result["digital_pages"] = digital_pages

        # Determine document type
        if len(scanned_pages) == 0:
            result["is_digital"] = True
            result["is_scanned"] = False
            result["is_mixed"] = False
        elif len(digital_pages) == 0:
            result["is_digital"] = False
            result["is_scanned"] = True
            result["is_mixed"] = False
        else:
            result["is_digital"] = False
            result["is_scanned"] = False
            result["is_mixed"] = True

        # Language detection via Devanagari character ratio
        total_chars = len(all_text)
        if total_chars > 0:
            devanagari_count = sum(
                1 for ch in all_text if "\u0900" <= ch <= "\u097F"
            )
            ratio = devanagari_count / total_chars

            if ratio > 0.20:
                result["language"] = "hi"
            elif ratio > 0.05:
                result["language"] = "mixed"
            else:
                result["language"] = "en"

        # Table detection: 3+ tab or pipe characters in a row on any page
        for page in reader.pages:
            text = page.extract_text() or ""
            for line in text.split("\n"):
                if line.count("\t") >= 3 or line.count("|") >= 3:
                    result["has_tables"] = True
                    break
            if result["has_tables"]:
                break

        # Annexure detection
        lower_text = all_text.lower()
        if any(kw in lower_text for kw in ["annex", "annexure", "schedule"]):
            result["has_annexures"] = True

        logger.info(
            "Format detection complete",
            is_digital=result["is_digital"],
            is_scanned=result["is_scanned"],
            is_mixed=result["is_mixed"],
            language=result["language"],
            page_count=result["page_count"],
            scanned_pages_count=len(scanned_pages),
            digital_pages_count=len(digital_pages),
        )

    except Exception as e:
        logger.error("Format detection failed", error=str(e), pdf_path=pdf_path)

    return result
