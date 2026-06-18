"""
ORBITAL LLM Client
Groq API wrapper for all LLM-powered extraction, classification, and validation steps.
"""

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from backend.core.config import get_config
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── Prompt Templates ────────────────────────────────────────────────────────

_PROMPT_EXTRACT_OBLIGATIONS_PREFIX = """\
You are a compliance obligation extractor for Indian banking regulations.
You will receive sections from a structured regulatory document. For each section, extract every compliance obligation present.

One section may contain MULTIPLE obligations — extract each separately.

Obligations can be:
- Mandatory: triggered by "shall", "must", "required to", "is obligated"
- Discretionary: triggered by "may", "can", "at its discretion"
- Conditional: triggered by "in case of", "where", "if", "subject to"
- Time-bound: contains a deadline, date, or duration
- Cross-referenced: refers to another circular for requirements

Return ONLY valid JSON array, no explanation, no markdown.

Return this exact shape per obligation:
[{
  "id": "string - generate sequential ID",
  "section_id": "string - matches section id from structure",
  "clause_number": "string - exact clause number e.g. 121A, 3.2.1",
  "actor": "string - Bank / Board of Directors / Chief Compliance Officer / Regulated Entity / Customer / RBI",
  "action": "string - precise action in one clear sentence. Do NOT copy raw text. Summarise cleanly.",
  "obligation_type": "mandatory / discretionary / conditional / time_bound",
  "trigger": "string - what triggers this e.g. immediately / always / on customer request",
  "deadline": {
    "text": "string - deadline as written e.g. within 30 days",
    "absolute_date": "ISO date or null",
    "duration": "string or null - e.g. 30 days",
    "urgency": "immediate / short_term / medium_term / long_term / ongoing / triggered"
  },
  "domain": "KYC_AML / Cybersecurity / DataPrivacy / FinancialInclusion / BusinessContinuity / FraudManagement / CapitalAdequacy / Payments / CustomerService / Governance / ITInfrastructure / ReportingAudit / HR_Training / FEMA / Other",
  "departments": ["Compliance / AML_KYC / Cybersecurity / IT / DigitalBanking / Operations / BranchNetwork / RiskManagement / Legal / Finance / Treasury / HR / CustomerService / FraudManagement / InternalAudit / RetailBanking"],
  "severity": "critical / high / medium / low",
  "severity_reason": "string - one sentence",
  "evidence_required": ["list of proof items"],
  "penalty_if_missed": "string or null",
  "fine_exposure_inr": "number or null",
  "cross_references": ["list of circulars referenced"],
  "confidence": 0.0,
  "notes": "string or null"
}]

Severity: critical=immediate/penalty, high=within 30 days, medium=within 90 days, low=discretionary/long timeline.

SECTIONS TO PROCESS:
"""

_PROMPT_RESUMMARIZE_ACTION = """\
You are a compliance obligation summariser.
Rewrite the following raw regulatory clause text as a single concise imperative action sentence.

Rules:
- Start with an imperative verb (e.g. "Ensure", "Submit", "Report", "Maintain").
- Preserve ALL specific requirements: deadlines, percentages, formats, authorities.
- Remove legislative boilerplate, cross-references to other clauses, and filler words.
- Maximum 40 words.
- Return ONLY the rewritten sentence, no explanation, no quotes, no markdown.

RAW TEXT:
"""

_PROMPT_CLASSIFY_DOMAIN_PREFIX = """\
You are a regulatory compliance domain classifier for Indian banking regulations.
Classify the following text into its primary compliance domain.

Return ONLY valid JSON, no explanation, in this exact shape:
{"primary_domain": "KYC_AML / Cybersecurity / DataPrivacy / FinancialInclusion / BusinessContinuity / FraudManagement / CapitalAdequacy / Payments / CustomerService / Governance / ITInfrastructure / ReportingAudit / HR_Training / FEMA / Other", "secondary_domains": [], "confidence": 0.0, "reasoning": "one sentence"}

TEXT:
"""

_PROMPT_GENERATE_MAP_CARD_PREFIX = """\
You are a compliance task generator for Indian banking regulations.
Convert the obligation below into an actionable MAP card.

Return ONLY valid JSON, no explanation, in this shape:
{"task": "title", "description": "2-3 sentences", "checklist": ["step1", "step2", "step3"], "owner_department": "dept", "due_in_days": null, "priority": "critical/high/medium/low", "evidence_template": "what proof to generate"}

OBLIGATION:
"""

_PROMPT_ANALYSE_GAP_PREFIX = """\
You are a regulatory gap analysis agent for Indian banking compliance.
Compare the obligation against the policy chunks and identify gaps.

Return ONLY valid JSON, no explanation, in this shape:
{"covered": false, "coverage_percentage": 0, "gaps": [], "recommendations": [], "human_readable_summary": "2-3 sentences"}

OBLIGATION:
"""

_PROMPT_ANALYSE_GAP_MIDDLE = "\n\nPOLICY CHUNKS:\n"

_PROMPT_VALIDATE_EXTRACTION_PREFIX = """\
You are a compliance validation agent reviewing obligation extraction from Indian banking regulations.
You will receive the original raw document text and the obligations extracted so far.

Find what was MISSED or WRONG. Check for:
- "shall" obligations that were not extracted
- Deadlines mentioned but not captured
- Effective dates not extracted
- Departments mapped incorrectly
- Actions too vague or verbatim raw text
- Duplicate obligations that should be merged
- Discretionary "may" clauses marked mandatory

Return ONLY valid JSON, no explanation, in this exact shape:
{"missed_obligations": [{"clause_number": "string", "raw_text": "exact sentence", "reason_missed": "why missed"}], "incorrect_extractions": [{"obligation_id": "string", "field": "which field", "current_value": "string", "correct_value": "string", "reason": "string"}], "missing_effective_date": "ISO date or null", "overall_confidence": 0.0, "validation_notes": "summary of document type and parsing challenges"}

ORIGINAL DOCUMENT TEXT:
"""

_PROMPT_VALIDATE_EXTRACTION_MIDDLE = "\n\nEXTRACTED OBLIGATIONS:\n"


# ── Rate Limiter ─────────────────────────────────────────────────────────────

class _RateLimiter:
    """Token-bucket rate limiter to stay within Groq's free-tier limits."""

    def __init__(self, calls_per_minute: int = 28):
        self._calls_per_minute = calls_per_minute
        self._min_interval = 60.0 / calls_per_minute
        self._last_call: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


# ── LLM Client ───────────────────────────────────────────────────────────────

class OrbitalLLMClient:
    """
    Thin wrapper around Groq and Ollama chat completion APIs.
    All methods return parsed Python objects (list or dict).
    Falls back to empty structures on any error so the pipeline never crashes.
    """

    def __init__(self):
        self._config = get_config()
        self._rate_limiter = _RateLimiter(calls_per_minute=28)
        self._groq_client = None  # lazy-loaded
        self._default_provider = (self._config.LLM_PROVIDER or "groq").strip().lower()
        self.parse_failures = 0

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_groq_client(self):
        """Lazy-load the Groq client to avoid import errors at startup."""
        if self._groq_client is None:
            try:
                from groq import Groq  # type: ignore
                self._groq_client = Groq(api_key=self._config.GROQ_API_KEY)
            except ImportError:
                logger.warning("groq package not installed — LLM features disabled")
                self._groq_client = None
        return self._groq_client

    def _get_provider(self, provider: str | None) -> str:
        selected = (provider or self._default_provider or "groq").strip().lower()
        if selected not in {"groq", "ollama"}:
            logger.warning("Unknown LLM provider, defaulting to Groq", provider=selected)
            return "groq"
        return selected

    def _chat_groq(self, model: str, prompt: str, max_tokens: int | None = None) -> str:
        client = self._get_groq_client()
        if client is None:
            return ""

        self._rate_limiter.wait()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or self._config.GROQ_MAX_TOKENS,
                temperature=self._config.GROQ_TEMPERATURE,
                timeout=self._config.GROQ_TIMEOUT_SECONDS,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("Groq API call failed", model=model, error=str(exc))
            return ""

    def _chat_ollama(self, model: str, prompt: str, max_tokens: int | None = None) -> str:
        self._rate_limiter.wait()

        url = self._config.OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self._config.OLLAMA_TEMPERATURE,
                "num_predict": max_tokens or self._config.OLLAMA_MAX_TOKENS,
            },
        }

        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self._config.OLLAMA_TIMEOUT_SECONDS) as response:
                data = json.loads(response.read().decode("utf-8"))
                message = data.get("message") or {}
                return message.get("content", "") or ""
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Ollama API call failed", model=model, error=str(exc))
            return ""

    def _chat(self, model: str, prompt: str, max_tokens: int | None = None, provider: str | None = None) -> str:
        """
        Make a single chat completion call.
        Returns the raw content string, or "" on failure.
        """
        selected_provider = self._get_provider(provider)
        if selected_provider == "ollama":
            return self._chat_ollama(model, prompt, max_tokens=max_tokens)
        return self._chat_groq(model, prompt, max_tokens=max_tokens)

    def _resolve_model(self, mode: str, provider: str | None = None) -> str:
        selected_provider = self._get_provider(provider)
        if selected_provider == "ollama":
            return self._config.OLLAMA_MODEL

        model_map = {
            "chat_qa": self._config.GROQ_MODEL_CHAT,
            "extraction": self._config.GROQ_MODEL_EXTRACTION,
            "gap_analysis": self._config.GROQ_MODEL_GAP_ANALYSIS,
            "evidence": self._config.GROQ_MODEL_EVIDENCE,
            "classification": self._config.GROQ_MODEL_CLASSIFICATION,
        }
        return model_map.get(mode, self._config.GROQ_MODEL_CHAT)

    def _chat_and_parse(self, model: str, prompt: str, provider: str | None = None, max_tokens: int | None = None) -> Any:
        """Chat wrapper that parses JSON and automatically retries once on parse failure."""
        content = self._chat(model, prompt, max_tokens=max_tokens, provider=provider)
        result = self._parse_json(content)
        if result is not None:
            return result
            
        logger.warning("JSON parse failed, retrying once", provider=provider)
        retry_prompt = prompt + "\n\nCRITICAL: Your previous response contained invalid JSON. You MUST return ONLY valid JSON, no markdown fences, no explanation."
        content_retry = self._chat(model, retry_prompt, max_tokens=max_tokens, provider=provider)
        result_retry = self._parse_json(content_retry)
        if result_retry is not None:
            return result_retry
            
        self.parse_failures += 1
        logger.error("Permanent JSON parse failure after retry", snippet=content_retry[:120] if content_retry else "Empty")
        return None

    @staticmethod
    def _parse_json(content: str) -> Any:
        """
        Robustly extract JSON from an LLM response.
        Handles markdown code fences and stray text before/after JSON.
        """
        if not content:
            return None
            
        # Strip markdown fences safely
        content = re.sub(r"```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"```", "", content)
        
        # Strip leading/trailing non-JSON prose
        match = re.search(r"([\[\{].*[\]\}])", content, re.DOTALL)
        if match:
            content = match.group(1)
            
        try:
            return json.loads(content, strict=False)
        except json.JSONDecodeError:
            # Last-resort: strip trailing comma before closing bracket/brace
            cleaned = re.sub(r",\s*([}\]])", r"\1", content)
            try:
                return json.loads(cleaned, strict=False)
            except json.JSONDecodeError:
                return None

    # ── Public API ────────────────────────────────────────────────────────────

    def call(self, mode: str, prompt: str, provider: str | None = None) -> str:
        """
        Generic LLM call using the chat model.
        Returns the raw string response.
        Used for one-off prompts (severity assessment, etc.).
        """
        model = self._resolve_model(mode, provider=provider)
        return self._chat(model, prompt, provider=provider)

    def extract_obligations(self, section_text: str, provider: str | None = None) -> list:
        """
        Prompt 2 — Extract structured obligations from a section of regulatory text.
        Returns a list of obligation dicts, or [] on failure.
        """
        if not section_text or not section_text.strip():
            return []

        prompt = _PROMPT_EXTRACT_OBLIGATIONS_PREFIX + section_text
        result = self._chat_and_parse(self._resolve_model("extraction", provider=provider), prompt, provider=provider)

        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # Some models wrap the array in {"obligations": [...]}
            for key in ("obligations", "items", "data", "results"):
                if isinstance(result.get(key), list):
                    return result[key]
        if result is not None:
            logger.warning("extract_obligations: unexpected response shape", snippet=str(result)[:120])
        return []

    def resummarize_action(self, raw_action: str, provider: str | None = None) -> str:
        """Rewrite a verbatim regulatory action as a concise imperative summary.

        Returns the rewritten string, or "" on failure / empty response.
        Uses a low max_tokens cap to keep responses tight.
        """
        if not raw_action or not raw_action.strip():
            return ""

        prompt = _PROMPT_RESUMMARIZE_ACTION + raw_action
        content = self._chat(
            self._resolve_model("extraction", provider=provider),
            prompt,
            max_tokens=120,
            provider=provider,
        )
        # Strip quotes, markdown, and whitespace the model might add
        cleaned = content.strip().strip('"').strip("'").strip("`").strip()
        if not cleaned or len(cleaned) < 10:
            return ""
        # Ensure it ends with a period
        if not cleaned.endswith("."):
            cleaned += "."
        return cleaned

    def classify_domain(self, text: str, provider: str | None = None) -> dict:
        """
        Classify text into a primary compliance domain.
        Returns a dict with primary_domain, confidence, etc.
        """
        if not text or not text.strip():
            return {"primary_domain": "Other", "confidence": 0.0, "secondary_domains": [], "reasoning": ""}

        prompt = _PROMPT_CLASSIFY_DOMAIN_PREFIX + text
        result = self._chat_and_parse(self._resolve_model("classification", provider=provider), prompt, provider=provider)

        if isinstance(result, dict):
            return result
        return {"primary_domain": "Other", "secondary_domains": [], "confidence": 0.0, "reasoning": "parse_failed"}

    def generate_map_card(self, obligation: dict, provider: str | None = None) -> dict:
        """
        Generate a MAP compliance task card from an obligation dict.
        Returns a dict with task, checklist, priority, etc.
        """
        if not isinstance(obligation, dict):
            return {}

        prompt = _PROMPT_GENERATE_MAP_CARD_PREFIX + json.dumps(obligation, ensure_ascii=False, indent=2)
        result = self._chat_and_parse(self._resolve_model("extraction", provider=provider), prompt, provider=provider)

        if isinstance(result, dict):
            return result
        return {}

    def analyse_gap(self, obligation: dict, policy_chunks: list, provider: str | None = None) -> dict:
        """
        Prompt-based gap analysis comparing an obligation to policy chunks.
        Returns a dict with covered, gaps, recommendations, etc.
        """
        if not isinstance(obligation, dict) or not isinstance(policy_chunks, list):
            return {"covered": False, "coverage_percentage": 0, "gaps": [], "recommendations": [], "human_readable_summary": ""}

        prompt = (
            _PROMPT_ANALYSE_GAP_PREFIX
            + json.dumps(obligation, ensure_ascii=False, indent=2)
            + _PROMPT_ANALYSE_GAP_MIDDLE
            + json.dumps(policy_chunks, ensure_ascii=False, indent=2)
        )
        result = self._chat_and_parse(self._resolve_model("gap_analysis", provider=provider), prompt, provider=provider)

        if isinstance(result, dict):
            return result
        return {"covered": False, "coverage_percentage": 0, "gaps": [], "recommendations": [], "human_readable_summary": ""}

    def validate_extraction(self, content: str, provider: str | None = None) -> dict:
        """
        Prompt 3 — Validate extracted obligations against original document text.

        `content` is a JSON string with keys:
            "raw_text"         — original document text
            "obligations_json" — list of extracted obligation dicts

        Returns a ValidationResultSchema-compatible dict, or {} on failure.
        """
        if not content or not content.strip():
            return {}

        try:
            payload = json.loads(content)
            raw_text = payload.get("raw_text", "")
            obligations_json = payload.get("obligations_json", [])
        except (json.JSONDecodeError, AttributeError):
            raw_text = content
            obligations_json = []

        # Truncate raw_text to stay within token limits (~12 000 chars ≈ ~3 000 tokens)
        MAX_RAW_CHARS = 12000
        truncated_raw = raw_text[:MAX_RAW_CHARS]
        if len(raw_text) > MAX_RAW_CHARS:
            truncated_raw += "\n[...text truncated for token limit...]"

        # Compact obligations JSON — keep only the most relevant fields to save tokens
        compact_obligations = []
        for ob in obligations_json:
            if not isinstance(ob, dict):
                continue
            compact_obligations.append({
                "id": ob.get("id", ""),
                "clause_number": ob.get("clause_number", ""),
                "actor": ob.get("actor", ""),
                "action": ob.get("action", ""),
                "obligation_type": ob.get("obligation_type", ""),
                "trigger": ob.get("trigger", ""),
                "deadline": ob.get("deadline", {}),
                "domain": ob.get("domain", ""),
                "departments": ob.get("departments", []),
                "severity": ob.get("severity", ""),
                "confidence": ob.get("confidence", 0.0),
            })

        prompt = (
            _PROMPT_VALIDATE_EXTRACTION_PREFIX
            + truncated_raw
            + _PROMPT_VALIDATE_EXTRACTION_MIDDLE
            + json.dumps(compact_obligations, ensure_ascii=False, indent=2)
        )

        result = self._chat_and_parse(self._resolve_model("gap_analysis", provider=provider), prompt, max_tokens=2048, provider=provider)

        if isinstance(result, dict):
            return result

        if result is not None:
            logger.warning("validate_extraction: unexpected response shape", snippet=str(result)[:120])
        return {}


# ── Singleton ────────────────────────────────────────────────────────────────

llm = OrbitalLLMClient()
