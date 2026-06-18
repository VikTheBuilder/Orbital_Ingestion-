# ORBITAL JSON Rule Engine

Production-grade rule packs for deterministic extraction of compliance obligations from RBI circulars.

## Architecture

```
PDF → OCR → Clause Segmentation → JSON Rule Engine → LLM Enrichment → Validation → Final JSON
```

The rule engine handles all deterministic work. The LLM only resolves ambiguity, summarises, and classifies novel patterns.

## Folder Map

| Folder | Purpose |
|--------|---------|
| `common/` | Shared trigger verbs, obligation verbs, negation patterns |
| `actors/` | Actor detection — who the obligation applies to |
| `obligations/` | Obligation type classification rules |
| `deadlines/` | Deadline and duration extraction |
| `domains/` | Domain keyword and phrase dictionaries |
| `departments/` | Department mapping rules |
| `severity/` | Severity scoring rules |
| `metadata/` | Circular number, date, effective date extraction |
| `clause_detection/` | Regex for clause segmentation |
| `cross_references/` | Reference to other circulars, acts, directions |
| `validations/` | Post-extraction QA rules |
| `confidence/` | Confidence scoring factors |
| `sector_rules/` | Sector-specific override packs |
| `rule_discovery/` | Self-improving rule discovery specification |

## Rule File Schema

Every JSON rule file follows this envelope:

```json
{
  "version": "1.0.0",
  "last_updated": "2026-06-17",
  "source_authority": "RBI",
  "rules": [ ... ]
}
```
