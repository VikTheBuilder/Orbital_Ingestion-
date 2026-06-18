# ORBITAL

ORBITAL is a regulatory PDF ingestion pipeline for Indian compliance documents. It reads raw PDFs, extracts text, identifies structure, derives obligations, validates the results, and writes the output into structured JSON and fine-tuning datasets.

The project is built to support two LLM providers:

- Groq for hosted inference
- Ollama for local inference, with `phi3:latest` as the default local model

## Overview

The pipeline is designed for regulatory documents such as RBI circulars and similar compliance notices. It is optimized for documents that may be:

- digitally generated PDFs
- scanned PDFs that need OCR fallback
- long, sectioned circulars with annexures and tables
- documents that contain one or more compliance obligations per section

At a high level, ORBITAL:

1. Detects the PDF format
2. Extracts text from the file
3. Infers document structure
4. Extracts obligations from each section
5. Validates the extraction against the original text
6. Saves structured JSON and training pairs

## Key features

- Regulatory source detection for RBI, SEBI, CERT-In, NPCI, IRDAI, DPDP, FIU-IND, and IBA
- Digital/scanned PDF detection
- OCR fallback for image-heavy documents
- Structure extraction for sections, annexures, tables, and headings
- Obligation extraction with metadata such as actor, action, deadline, domain, severity, and evidence requirements
- Validation pass to catch missed or incorrect extraction
- Generation of fine-tuning examples for future model improvement
- Provider selection between Groq and Ollama through a single config switch

## Project layout

- `backend/core/` — configuration, logging, and the LLM client
- `backend/ingestion/` — chunking, OCR, structure extraction, obligation extraction, validation, and pipeline orchestration
- `backend/scripts/` — CLI entrypoints and test scripts
- `data/raw/` — source PDFs to be processed
- `data/extracted/` — extracted intermediate text artifacts
- `data/structured/` — final structured JSON outputs grouped by source
- `data/finetune/` — generated training pairs in JSONL format
- `rules/` — project reference material and supporting documentation

## Requirements

- Python 3.10+ recommended
- `pip` for dependency installation
- Ollama installed locally if you want to use the local model path
- A Groq API key if you want to use Groq

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Environment setup

Copy `.env.example` to `.env` and edit the values you need.

### Provider selection

Choose one provider globally:

```env
LLM_PROVIDER=groq
```

or:

```env
LLM_PROVIDER=ollama
```

### Groq configuration

Use Groq when you want hosted inference and already have an API key.

Important variables:

- `GROQ_API_KEY`
- `GROQ_MODEL_EXTRACTION`
- `GROQ_MODEL_GAP_ANALYSIS`
- `GROQ_MODEL_EVIDENCE`
- `GROQ_MODEL_CHAT`
- `GROQ_MODEL_CLASSIFICATION`
- `GROQ_MAX_TOKENS`
- `GROQ_TEMPERATURE`
- `GROQ_TIMEOUT_SECONDS`

### Ollama configuration

Use Ollama when you want local inference.

Important variables:

- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=phi3:latest`
- `OLLAMA_MAX_TOKENS`
- `OLLAMA_TEMPERATURE`
- `OLLAMA_TIMEOUT_SECONDS`

If needed, pull the model first:

```powershell
ollama pull phi3:latest
```

## Starting Ollama

If Ollama is not already running, start it with:

```powershell
ollama serve
```

Then verify the model is available:

```powershell
ollama run phi3:latest
```

The application sends requests to the local Ollama HTTP API at `http://localhost:11434`.

## How the pipeline works

The main ingestion flow is implemented in `backend/ingestion/pipeline.py` and follows this order:

### 1. Input validation

The pipeline checks that the PDF path exists and resolves the regulatory source.

### 2. Format detection

`backend/ingestion/format_detector.py` determines whether the PDF is digital or scanned and estimates language and page count.

### 3. Text extraction

`backend/ingestion/ocr.py` extracts text from the PDF. OCR is used when the document is scanned or if direct extraction is insufficient.

### 4. Structure extraction

`backend/ingestion/structure_extractor.py` segments the text into sections, tables, and annexures and generates a document structure object.

### 5. Obligation extraction

`backend/ingestion/obligation_extractor.py` turns structured sections into obligation records.

### 6. Validation

`backend/ingestion/validator.py` checks the extracted obligations against the raw text for missed items, incorrect fields, and missing dates.

### 7. Chunking and persistence

`backend/ingestion/chunker.py` creates downstream chunks and the pipeline writes:

- structured JSON to `data/structured/`
- fine-tuning pairs to `data/finetune/raw_pairs.jsonl`

## Running the pipeline

Process one PDF:

```powershell
python -m backend.scripts.run_pipeline --file path\to\document.pdf --source RBI
```

Process a folder recursively:

```powershell
python -m backend.scripts.run_pipeline --folder data/raw/rbi/ --source auto
```

### Supported sources

- `RBI`
- `SEBI`
- `CERT-In`
- `NPCI`
- `IRDAI`
- `DPDP`
- `FIU-IND`
- `IBA`
- `auto`

When `auto` is used, the source is inferred from the filename.

## CLI options

The main runner is `backend/scripts/run_pipeline.py`.

Available options:

- `--file` — process a single PDF
- `--folder` — process all PDFs recursively inside a folder
- `--source` — override or auto-detect the regulatory source
- `--verbose` — print individual obligations after each file

Example:

```powershell
python -m backend.scripts.run_pipeline --folder data/raw/rbi/ --source auto --verbose
```

## Output files

### Structured JSON

Each processed document is saved to:

```text
data/structured/<SOURCE>/<DOC_ID>.json
```

These JSON files contain the document structure, extracted obligations, and validation output.

### Fine-tuning data

Fine-tuning examples are appended to:

```text
data/finetune/raw_pairs.jsonl
```

The file may contain multiple training examples per document, including extraction, classification, task generation, and severity-related examples.

### Duplicate handling

If a structured output already exists for a document ID, the pipeline reuses the cached result instead of reprocessing it.

## Testing

Run the standalone LLM client test:

```powershell
python -m backend.scripts.test_llm_client
```

Run the synthetic end-to-end pipeline test:

```powershell
python -m backend.scripts.test_pipeline
```

These tests help verify extraction, classification, validation, and pipeline behavior without requiring a real document set.

## Important files

- `backend/core/config.py` — environment-backed configuration
- `backend/core/llm_client.py` — Groq/Ollama abstraction
- `backend/ingestion/pipeline.py` — orchestration logic
- `backend/ingestion/obligation_extractor.py` — obligation extraction logic
- `backend/ingestion/validator.py` — validation pass
- `backend/scripts/run_pipeline.py` — CLI runner
- `backend/scripts/test_llm_client.py` — client smoke test
- `backend/scripts/test_pipeline.py` — synthetic integration test

## Working with data

The `data/` directory is where the pipeline writes outputs. These files can grow quickly, so only keep the documents and artifacts you actually need.

Recommended practice:

- store source PDFs in `data/raw/`
- keep structured outputs in `data/structured/`
- treat `data/extracted/` and `data/finetune/` as generated artifacts

## Troubleshooting

### Ollama is not responding

- confirm `ollama serve` is running
- confirm `phi3:latest` exists locally
- verify the base URL is `http://localhost:11434`

### Groq requests fail

- confirm `GROQ_API_KEY` is set in `.env`
- verify the selected Groq models are valid
- check network access and any rate limits

### OCR or PDF extraction looks wrong

- confirm the PDF is readable and not severely corrupted
- check whether the file is scanned and needs OCR
- review `backend/ingestion/format_detector.py` and `backend/ingestion/ocr.py`

### Results do not update after changing `.env`

- restart the Python process
- the config is cached during runtime, so environment changes are not always picked up immediately

## Notes

- The pipeline is intentionally resilient and will try to keep moving even if one LLM call fails.
- `phi3:latest` is the default local model, but you can change it in `.env`.
- The repo currently does not include a license file.

## Suggested next steps

- Add a sample `.env` with safer placeholder values
- Add `.gitkeep` files if you want to preserve empty data directories in git
- Add a short architecture diagram if you plan to onboard other contributors
