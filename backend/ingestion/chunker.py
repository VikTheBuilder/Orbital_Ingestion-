"""
ORBITAL Document Chunker
Splits structured document sections into sized chunks for embedding/retrieval.
"""

import re
from collections import Counter
from typing import List

from backend.core.logger import get_logger
from backend.ingestion.schemas import ChunkSchema, DocumentStructureSchema

logger = get_logger(__name__)


def chunk_document(
    doc_structure: DocumentStructureSchema,
    config,
) -> List[ChunkSchema]:
    """
    Split document sections into chunks with overlap.

    Args:
        doc_structure: Parsed document structure with sections and obligations.
        config: OrbitalConfig instance with CHUNK_SIZE and CHUNK_OVERLAP.

    Returns:
        List of ChunkSchema objects.
    """
    try:
        chunk_size = config.CHUNK_SIZE
        chunk_overlap = config.CHUNK_OVERLAP
        all_chunks: List[ChunkSchema] = []

        logger.info(
            "Chunking started",
            doc_id=doc_structure.doc_id,
            sections=len(doc_structure.sections),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for section in doc_structure.sections:
            section_text = section.text
            if not section_text or len(section_text.strip()) < 100:
                continue

            # Determine domain info from section obligations
            if section.obligations:
                domain_counts = Counter(o.domain for o in section.obligations)
                domain = domain_counts.most_common(1)[0][0]
                domain_tags = list(set(o.domain for o in section.obligations))
                has_obligation = True
                obligation_ids = [str(o.id) for o in section.obligations]
            else:
                domain = "General"
                domain_tags = []
                has_obligation = False
                obligation_ids = []

            chunk_type = "obligation" if has_obligation else "context"

            # Split section into chunks
            if len(section_text) <= chunk_size:
                # Single chunk for short sections
                raw_chunks = [section_text]
            else:
                raw_chunks = _split_with_overlap(
                    section_text, chunk_size, chunk_overlap
                )

            for raw_chunk in raw_chunks:
                if len(raw_chunk.strip()) < 100:
                    continue
                all_chunks.append(ChunkSchema(
                    chunk_id="",  # placeholder, assigned below
                    doc_id=doc_structure.doc_id,
                    source=doc_structure.source,
                    title=doc_structure.title,
                    section_heading=section.heading,
                    text=raw_chunk.strip(),
                    page_number=section.page_number,
                    chunk_index=0,  # assigned below
                    total_chunks=0,  # assigned below
                    domain=domain,
                    domain_tags=domain_tags,
                    has_obligation=has_obligation,
                    obligation_ids=obligation_ids,
                    date=doc_structure.date,
                    chunk_type=chunk_type,
                    language=doc_structure.language,
                ))

        # Assign chunk indices and IDs
        total = len(all_chunks)
        for idx, chunk in enumerate(all_chunks):
            chunk.chunk_index = idx
            chunk.total_chunks = total
            chunk.chunk_id = f"{doc_structure.doc_id}-chunk-{idx:04d}"

        logger.info("Chunking complete", total_chunks=total)
        return all_chunks

    except Exception as e:
        logger.error("Chunking failed", error=str(e))
        return []


def _split_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks of approximately chunk_size characters,
    splitting on paragraph and then sentence boundaries, with overlap.
    """
    # First try to split on paragraph boundaries
    paragraphs = re.split(r"\n\n+", text)

    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If a single paragraph exceeds chunk_size, split on sentences
        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                    current_chunk = (
                        current_chunk + " " + sentence
                        if current_chunk
                        else sentence
                    )
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
        elif len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (
                current_chunk + "\n\n" + para if current_chunk else para
            )
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    # Apply overlap: prepend last N chars of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-overlap:] if len(prev) >= overlap else prev
            overlapped.append(overlap_text + " " + chunks[i])
        return overlapped

    return chunks
