"""
ORBITAL Rule Engine
Loads all JSON rule packs from the rules/ directory once at startup
and exposes deterministic extraction methods used by the pipeline.

Architecture position:
  PDF → OCR → Clause Segmentation → [RuleEngine V1] → LLM Enrichment (selective)
    → [RuleEngine V2 repass] → Confidence Engine → Validation → JSON
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from backend.core.logger import get_logger
from backend.core.utils import parse_date_to_iso, parse_financial_year, unique_ordered

logger = get_logger(__name__)

# ── Resolve rules/ root relative to this file ──────────────────────────────
_RULES_ROOT = Path(__file__).resolve().parent.parent.parent / "rules"


def _load(relative_path: str) -> dict:
    """Load a single JSON rule file. Returns empty dict on failure."""
    full = _RULES_ROOT / relative_path
    try:
        return json.loads(full.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Rule file load failed", path=str(full), error=str(exc))
        return {}


# ── RuleEngine ──────────────────────────────────────────────────────────────

class RuleEngine:
    """
    Singleton rule engine.  All public methods are pure and stateless once
    initialised — safe to call from multiple threads.
    """

    def __init__(self):
        logger.info("RuleEngine initialising", rules_root=str(_RULES_ROOT))
        self._load_all()
        logger.info(
            "RuleEngine ready",
            trigger_verbs=len(self._trigger_rules),
            actor_patterns=len(self._actor_rules),
            deadline_patterns=len(self._deadline_rules),
            domain_keywords=len(self._domain_rules),
            sector_packs=len(self._sector_packs),
        )

    # ── Loaders ────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        self._trigger_rules   = _load("common/trigger_verbs.json").get("rules", [])
        self._negation_rules  = _load("common/negation_patterns.json").get("rules", [])
        self._actor_rules     = _load("actors/actor_patterns.json").get("rules", [])
        # Merge source-specific actor packs (IRDAI, etc.)
        for actor_file in (_RULES_ROOT / "actors").glob("*_actor_patterns.json"):
            if actor_file.name == "actor_patterns.json":
                continue
            extra = json.loads(actor_file.read_text(encoding="utf-8"))
            self._actor_rules.extend(extra.get("rules", []))
            logger.info("Loaded source-specific actor pack", file=actor_file.name,
                        rules=len(extra.get("rules", [])))
        self._obligation_types = _load("obligations/obligation_types.json").get("types", [])
        deadline_pack         = _load("deadlines/deadline_patterns.json")
        self._deadline_rules  = deadline_pack.get("rules", [])
        # Cache urgency thresholds once — never re-read in the hot path
        self._urgency_thresholds: dict = deadline_pack.get("urgency_thresholds", {
            "immediate_days": 0, "short_term_days": 30,
            "medium_term_days": 90, "long_term_days": 365,
        })
        self._effdate_rules   = _load("deadlines/effective_date_patterns.json").get("rules", [])
        self._domain_rules    = _load("domains/domain_keywords.json").get("rules", [])
        self._dept_map        = _load("departments/department_mapping.json")
        self._severity_rules  = _load("severity/severity_rules.json").get("rules", [])
        self._confidence_factors = _load("confidence/confidence_scoring.json")
        self._clause_rules    = _load("clause_detection/clause_segmentation.json")
        self._xref_rules      = _load("cross_references/cross_reference_patterns.json").get("rules", [])
        self._meta_rules      = _load("metadata/circular_number_patterns.json")
        self._validation_rules = _load("validations/validation_rules.json").get("rules", [])

        # Sector rule packs — load all, indexed by pack name
        self._sector_packs: dict[str, dict] = {}
        for pack_file in (_RULES_ROOT / "sector_rules").rglob("*.json"):
            pack = json.loads(pack_file.read_text(encoding="utf-8"))
            self._sector_packs[pack_file.stem] = pack

        # Pre-compile all regex patterns for performance
        self._compiled_triggers  = self._compile(self._trigger_rules)
        self._compiled_negations = self._compile(self._negation_rules)
        self._compiled_actors    = self._compile(self._actor_rules)
        self._compiled_deadlines = self._compile(self._deadline_rules)
        self._compiled_xrefs     = self._compile(self._xref_rules)
        self._compiled_meta_circulars = self._compile(
            self._meta_rules.get("circular_number_patterns", [])
        )
        self._compiled_meta_dates = self._compile(
            self._meta_rules.get("date_patterns", [])
        )
        self._compiled_effdates = self._compile(self._effdate_rules)

    @staticmethod
    def _compile(rules: list) -> list[tuple[dict, re.Pattern]]:
        """Return (rule_dict, compiled_pattern) pairs, skipping bad regex."""
        result = []
        for rule in rules:
            pattern = rule.get("pattern", "")
            if not pattern:
                continue
            try:
                result.append((rule, re.compile(pattern, re.IGNORECASE)))
            except re.error as exc:
                logger.warning("Bad regex in rule", rule_id=rule.get("id"), error=str(exc))
        return result

    # ── Trigger Detection ──────────────────────────────────────────────────

    def find_trigger(self, text: str) -> Optional[dict]:
        """
        Return the highest-weight matching trigger rule dict, or None.
        Negation rules are checked first — if the sentence is a prohibition,
        the trigger is returned with obligation_type='prohibition'.

        The returned dict includes 'urgency_override' if the matched trigger
        rule carries one (e.g. TV-037 "shall take immediate steps").
        """
        # Normalise newlines so multi-line OCR sentences match correctly
        text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
        lower = text.lower()

        # Check negations first (they take precedence)
        for rule, pat in self._compiled_negations:
            if pat.search(lower):
                return {
                    "id": rule["id"],
                    "pattern": rule["pattern"],
                    "obligation_type": rule.get("effect", "prohibition"),
                    "weight": rule.get("weight", 1.0),
                    "matched_text": pat.search(lower).group(0),
                    "urgency_override": rule.get("urgency_override"),
                }

        # Then check positive triggers
        best: Optional[dict] = None
        for rule, pat in self._compiled_triggers:
            m = pat.search(lower)
            if m and (best is None or rule.get("weight", 0) > best.get("weight", 0)):
                best = {
                    **rule,
                    "matched_text": m.group(0),
                    "urgency_override": rule.get("urgency_override"),
                }
        return best

    def is_negation(self, text: str) -> bool:
        """Return True if the sentence contains a prohibition or negation."""
        lower = text.lower()
        return any(pat.search(lower) for _, pat in self._compiled_negations)

    # ── Actor Detection ────────────────────────────────────────────────────

    def find_actor(self, text: str, source: str = None) -> str:
        """Return canonical actor name from the highest-weight matching rule.

        Args:
            text:   The clause text to search for actor patterns.
            source: Regulatory source (e.g. 'RBI', 'IRDAI', 'SEBI').  When
                    provided, patterns that carry a ``source_scope`` list are
                    skipped unless *source* appears in that list.  Patterns
                    without ``source_scope`` are always considered (universal).
        """
        lower = text.lower()
        best_actor = "Regulated Entity"
        best_weight = -1.0
        for rule, pat in self._compiled_actors:
            # ── Source-scope filtering ──────────────────────────────────
            scope = rule.get("source_scope")
            if scope and source and source not in scope:
                continue
            if pat.search(lower):
                w = rule.get("weight", 0.5)
                if w > best_weight:
                    best_weight = w
                    best_actor = rule["canonical_actor"]
        return best_actor

    # ── Obligation Type Classification ─────────────────────────────────────

    def classify_obligation_type(self, text: str, trigger_rule: Optional[dict]) -> str:
        """
        Map a clause to mandatory / discretionary / conditional / time_bound / prohibition.
        Uses trigger_rule from find_trigger() for efficiency.
        """
        if trigger_rule:
            otype = trigger_rule.get("obligation_type", "mandatory")
            # Map prohibition back to a valid schema type
            if otype == "prohibition":
                return "mandatory"  # stored as mandatory; action field describes the prohibition
            if otype in {"mandatory", "discretionary", "conditional", "time_bound",
                         "approval_required", "recommendation", "reporting", "governance"}:
                if otype in {"approval_required", "reporting", "governance", "recommendation"}:
                    return "mandatory"
                return otype

        lower = text.lower()
        # Conditional check
        if any(phrase in lower for phrase in ["in case of", "where", "if", "subject to", "provided that"]):
            return "conditional"
        # Discretionary check — "may", "at its discretion", "can" without mandatory verbs
        if re.search(r"\bmay\b", lower) and not re.search(r"\b(?:shall|must|required to|is obligated)\b", lower):
            return "discretionary"
        if "at its discretion" in lower or "at the discretion" in lower:
            return "discretionary"
        return "mandatory"

    # ── Obligation Sub-Type ────────────────────────────────────────────────

    def find_obligation_sub_type(self, text: str) -> Optional[str]:
        """
        Match text against obligation_types.json trigger_phrases.
        Returns the sub_type string of the best match, or None.
        """
        lower = text.lower()
        for otype in self._obligation_types:
            for phrase in otype.get("trigger_phrases", []):
                if phrase in lower:
                    return otype["sub_type"]
        return None

    def get_sub_type_departments(self, sub_type: str) -> list[str]:
        """Return departments for an obligation sub_type from department_mapping.json."""
        return self._dept_map.get("sub_type_overrides", {}).get(sub_type, [])

    def get_sub_type_severity_floor(self, sub_type: str) -> Optional[str]:
        """Return the severity_floor for a sub_type from obligation_types.json."""
        for otype in self._obligation_types:
            if otype["sub_type"] == sub_type:
                return otype.get("severity_floor")
        return None

    # ── Deadline Extraction ────────────────────────────────────────────────

    def extract_deadline(self, text: str, urgency_override: Optional[str] = None) -> dict:
        """
        Return a deadline dict compatible with DeadlineSchema.
        Tries all rules in order, returns the first match.

        urgency_override: if the trigger rule carries an urgency_override (e.g.
        "immediate" from TV-037/TV-040), it takes precedence over the computed
        urgency from the deadline pattern.
        """
        # Normalise newlines/extra whitespace so multi-line OCR text matches correctly
        text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()

        for rule, pat in self._compiled_deadlines:
            m = pat.search(text)
            if not m:
                continue
            raw = m.group(0).strip()
            urgency = rule.get("urgency", "ongoing")

            # Computed urgency — work out days from captured groups
            if urgency == "computed":
                urgency = self._compute_urgency_from_match(m, rule)

            # Trigger-level urgency_override takes precedence
            if urgency_override and urgency not in ("immediate",):
                urgency = urgency_override

            duration = rule.get("duration")
            if duration and "{1}" in str(duration):
                try:
                    duration = duration.replace("{1}", m.group(1)).replace("{2}", m.group(2))
                except IndexError:
                    duration = None

            absolute_date = rule.get("absolute_date")
            if absolute_date and "{1}" in str(absolute_date):
                try:
                    absolute_date = parse_date_to_iso(m.group(1))
                except IndexError:
                    absolute_date = None

            return {
                "text": raw,
                "absolute_date": absolute_date,
                "duration": duration,
                "urgency": urgency,
                "frequency": rule.get("frequency"),
            }

        # No deadline pattern matched.
        # If we have an urgency_override from the trigger (e.g. "immediate"),
        # still surface it even without an explicit deadline phrase.
        if urgency_override:
            return {
                "text": urgency_override.replace("_", " "),
                "absolute_date": None,
                "duration": None,
                "urgency": urgency_override,
                "frequency": None,
            }

        # Fallback — check for conditional trigger
        if re.search(r"\bin case of\b|\bwhere\b|\bif\b|\bsubject to\b", text, re.IGNORECASE):
            return {"text": "triggered by condition", "absolute_date": None,
                    "duration": None, "urgency": "triggered", "frequency": None}

        return {"text": "ongoing", "absolute_date": None, "duration": None,
                "urgency": "ongoing", "frequency": None}

    def _compute_urgency_from_match(self, m: re.Match, rule: dict) -> str:
        """Convert a numeric duration match to an urgency level using cached thresholds."""
        thresholds = self._urgency_thresholds
        try:
            number = int(m.group(1))
            unit = m.group(2).lower() if m.lastindex >= 2 else "days"
        except (IndexError, ValueError):
            return "medium_term"

        days = number
        if "week" in unit:
            days *= 7
        elif "month" in unit:
            days *= 30
        elif "year" in unit:
            days *= 365

        if days <= thresholds.get("immediate_days", 0):
            return "immediate"
        if days <= thresholds.get("short_term_days", 30):
            return "short_term"
        if days <= thresholds.get("medium_term_days", 90):
            return "medium_term"
        return "long_term"

    # ── Effective Date Extraction ──────────────────────────────────────────

    def extract_effective_date(self, text: str) -> Optional[str]:
        """Return ISO date string for when the document/direction comes into force."""
        for rule, pat in self._compiled_effdates:
            m = pat.search(text)
            if not m:
                continue
            group_idx = rule.get("date_group")
            if group_idx is None:
                return rule.get("resolved_date")
            try:
                raw = m.group(group_idx)
                # Financial year rules resolve via parse_financial_year
                if rule.get("is_financial_year"):
                    result = parse_financial_year(raw)
                    if result:
                        return result
                    continue
                return parse_date_to_iso(raw)
            except IndexError:
                continue
        return None

    # ── Domain Classification ──────────────────────────────────────────────

    def classify_domain(self, text: str) -> tuple[str, int]:
        """
        Return (domain_name, keyword_match_count).
        Uses keywords + phrases from domain_keywords.json.
        Sector packs can add additional keywords via domain_keyword_additions.
        """
        lower = text.lower()
        best_domain = "Other"
        best_score = 0

        for rule in self._domain_rules:
            domain = rule["domain"]
            if domain == "Other":
                continue
            score = 0
            for kw in rule.get("keywords", []):
                if kw in lower:
                    score += 1
            for phrase in rule.get("phrases", []):
                if phrase in lower:
                    score += 2  # phrases worth more
            if score > best_score:
                best_score = score
                best_domain = domain

        # Sector pack keyword additions
        for pack in self._sector_packs.values():
            additions = pack.get("domain_keyword_additions", {})
            for domain, keywords in additions.items():
                score = sum(1 for kw in keywords if kw in lower)
                if score > best_score:
                    best_score = score
                    best_domain = domain

        return best_domain, best_score

    # ── Department Mapping ─────────────────────────────────────────────────

    def get_departments(self, domain: str, sub_type: Optional[str] = None,
                        actor: Optional[str] = None, text: Optional[str] = None) -> list[str]:
        """
        Return a deduplicated, ordered list of responsible departments.
        Priority: sub_type_overrides > domain_to_departments > keyword_signals > actor_to_departments
        """
        result: list[str] = []

        # 1. Sub-type overrides (most specific)
        if sub_type:
            result.extend(self._dept_map.get("sub_type_overrides", {}).get(sub_type, []))

        # 2. Domain base mapping
        result.extend(self._dept_map.get("domain_to_departments", {}).get(domain, ["Compliance"]))

        # 3. Keyword signals from clause text
        if text:
            lower = text.lower()
            for sig in self._dept_map.get("keyword_signals", []):
                if any(kw in lower for kw in sig.get("keywords", [])):
                    result.extend(sig.get("add_departments", []))

        # 4. Actor-based departments
        if actor:
            result.extend(self._dept_map.get("actor_to_departments", {}).get(actor, []))

        return _unique_ordered(result)

    # ── Sector Rule Matching ───────────────────────────────────────────────

    def find_sector_obligation(self, text: str, actor: str) -> Optional[dict]:
        """
        Check sector rule packs for a matching obligation_addition.
        Returns the first matching pack entry, or None.
        The match is phrase-based — any trigger_phrase must appear in text.
        """
        lower = text.lower()
        for pack in self._sector_packs.values():
            applies_to = pack.get("applies_to", [])
            # Check if current actor is relevant for this pack
            actor_match = not applies_to or any(
                a.lower() in actor.lower() or actor.lower() in a.lower()
                for a in applies_to
            )
            if not actor_match:
                # Still check — actor may be generic "Regulated Entity"
                # Only skip if pack is very specific and actor is clearly wrong
                if actor not in ("Regulated Entity", "Bank") and "Regulated Entity" not in applies_to:
                    continue

            for ob_add in pack.get("obligation_additions", []):
                for phrase in ob_add.get("trigger_phrases", []):
                    if phrase in lower:
                        return ob_add
        return None

    # ── Severity Scoring ───────────────────────────────────────────────────

    def classify_severity(
        self,
        text: str,
        urgency: str,
        obligation_type: str,
        sub_type: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> str:
        """
        Return severity level using severity_rules.json.
        Rules are evaluated in order — first matching rule wins.
        Falls back to urgency-based heuristic if no rule matches.
        """
        lower = text.lower()

        for rule in self._severity_rules:
            conditions = rule.get("conditions", {})

            # urgency match
            if "urgency" in conditions and urgency not in conditions["urgency"]:
                continue
            # obligation_type match
            if "obligation_type" in conditions and obligation_type not in conditions["obligation_type"]:
                continue
            # sub_type match
            if "sub_type" in conditions and sub_type not in conditions["sub_type"]:
                continue
            # domain match
            if "domain" in conditions and domain not in conditions["domain"]:
                continue
            # keyword match — ALL specified keywords must appear
            if "keywords" in conditions:
                if not all(kw in lower for kw in conditions["keywords"]):
                    continue

            return rule["severity"]

        # No rule matched — urgency fallback
        if urgency == "immediate":
            return "critical"
        if urgency == "short_term":
            return "high"
        if obligation_type == "mandatory" or urgency == "medium_term":
            return "medium"
        return "low"

    def severity_reason(self, severity: str, urgency: str, sub_type: Optional[str] = None) -> str:
        """Generate a human-readable severity justification."""
        if severity == "critical":
            if sub_type == "incident_reporting":
                return "Cyber/fraud incident reporting is always critical — delay creates systemic risk."
            if urgency == "immediate":
                return "The clause requires immediate action, creating critical regulatory exposure if missed."
            return "The clause references a penalty, systemic risk, or immediate regulatory action."
        if severity == "high":
            if urgency == "short_term":
                return "The obligation has a short-term deadline (within 30 days) requiring prompt action."
            if sub_type == "customer_protection":
                return "Customer protection obligations carry high regulatory risk if not met."
            return "The obligation has significant operational or regulatory impact."
        if severity == "medium":
            return "The obligation requires a process or control update within 90 days."
        return "The obligation is discretionary, informational, or has a long/conditional timeline."

    # ── Confidence Scoring ─────────────────────────────────────────────────

    def compute_confidence(
        self,
        text: str,
        trigger_rule: Optional[dict],
        urgency: str,
        has_cross_ref: bool,
        actor: str,
        domain_match_count: int,
        llm_matched: bool = False,
        penalty_found: bool = False,
        fine_found: bool = False,
        is_duplicate: bool = False,
        action_verbatim_ratio: float = 0.0,
        clause_type: str = "obligation",
    ) -> float:
        """
        Compute obligation confidence using confidence_scoring.json factors.
        Returns a float clamped between min_score and max_score.

        New parameters vs original:
          action_verbatim_ratio: SequenceMatcher ratio between action and raw clause.
                                 If > 0.80 → apply -0.15 penalty.
          clause_type           : If clause_type is not 'obligation' or 'penalty'
                                  and no strong trigger → apply -0.08 penalty.
        """
        cfg = self._confidence_factors
        score = cfg.get("base_score", 0.50)

        tw = (trigger_rule or {}).get("matched_text", "")
        has_shall = "shall" in tw
        has_strong = any(s in tw for s in ["shall ensure", "shall submit", "shall report", "shall maintain"])
        has_may = "may" in tw and not has_shall

        # CF-016: actor text not found in source clause (hallucination risk)
        actor_in_text = bool(actor) and actor.lower() in text.lower()
        actor_is_generic = actor in ("Regulated Entity", "Bank", "")

        for factor in cfg.get("factors", []):
            fid = factor["id"]
            delta = factor["delta"]

            if fid == "CF-001" and has_shall and not has_strong:
                score += delta
            elif fid == "CF-002" and has_strong:
                score += delta
            elif fid == "CF-003" and urgency != "ongoing":
                score += delta
            elif fid == "CF-004" and urgency not in ("ongoing", "triggered"):
                score += delta
            elif fid == "CF-005" and actor not in ("Regulated Entity", "Bank"):
                score += delta
            elif fid == "CF-006" and has_cross_ref:
                score += delta
            elif fid == "CF-007" and domain_match_count > 2:
                score += delta
            elif fid == "CF-008" and len(text) > 80:
                score += delta
            elif fid == "CF-009" and llm_matched:
                score += delta
            elif fid == "CF-018" and penalty_found:
                score += delta
            elif fid == "CF-019" and fine_found:
                score += delta
            elif fid == "CF-013" and has_may:
                score += delta  # negative
            elif fid == "CF-015" and is_duplicate:
                score += delta  # negative
            # CF-011: missing actor — actor is empty string
            elif fid == "CF-011" and not actor:
                score += delta  # negative
            # CF-016: specific actor named but not present in clause text
            elif fid == "CF-016" and not actor_is_generic and not actor_in_text:
                score += delta  # negative

        # Extra structural penalties (not in JSON — applied here directly)
        # Verbatim action: only penalise genuinely long blobs, not short clean
        # passive sentences.  Threshold sourced from text_normalizer.
        try:
            from backend.ingestion.text_normalizer import VERBATIM_MIN_LENGTH as _VML
        except ImportError:
            _VML = 150
        if action_verbatim_ratio > 0.80 and len(text) >= _VML:
            score -= 0.15
        # Obligation extracted from a non-obligation clause type without strong trigger
        if clause_type not in ("obligation", "penalty") and not (has_shall or has_strong):
            score -= 0.08

        min_s = cfg.get("min_score", 0.30)
        max_s = cfg.get("max_score", 0.98)
        return round(max(min_s, min(max_s, score)), 4)

    # ── Cross-Reference Extraction ─────────────────────────────────────────

    def extract_cross_references(self, text: str) -> list[str]:
        """Return list of cross-reference strings found in text."""
        results = []
        for rule, pat in self._compiled_xrefs:
            for m in pat.finditer(text):
                val = re.sub(r"\s+", " ", m.group(0)).strip()
                if val and val not in results:
                    results.append(val)
        return results[:8]  # cap to avoid noise

    # ── Circular Number / Metadata Extraction ──────────────────────────────

    def extract_circular_number(self, text: str) -> Optional[str]:
        """Return the first matching circular number from metadata rules."""
        for rule, pat in self._compiled_meta_circulars:
            if rule.get("field") == "circular_number":
                m = pat.search(text)
                if m:
                    return m.group(0)
        return None

    def extract_reference_number(self, text: str) -> Optional[str]:
        """Return the first matching reference/department number."""
        for rule, pat in self._compiled_meta_circulars:
            if rule.get("field") == "reference_number":
                m = pat.search(text)
                if m:
                    return m.group(0)
        return None

    def extract_header_date(self, text: str) -> Optional[str]:
        """Extract the first date from the document header."""
        for rule, pat in self._compiled_meta_dates:
            m = pat.search(text)
            if m:
                return parse_date_to_iso(m.group(0))
        return None

    def detect_amends(self, text: str) -> Optional[str]:
        """Return a reference to the circular being amended/superseded."""
        for rule in self._meta_rules.get("amends_patterns", []):
            try:
                pat = re.compile(rule["pattern"], re.IGNORECASE)
                m = pat.search(text)
                if m:
                    ref_group = rule.get("reference_group", 1)
                    return re.sub(r"\s+", " ", m.group(ref_group)).strip()
            except (re.error, IndexError):
                continue
        return None

    def classify_document_type(self, text: str) -> Optional[str]:
        """Return 'master_direction', 'master_circular', or None."""
        for rule, pat in self._compiled_meta_circulars:
            if rule.get("field") == "document_type":
                if pat.search(text):
                    return rule.get("resolved_value")
        return None

    # ── Clause Segmentation Helpers ────────────────────────────────────────

    def get_clause_patterns(self) -> list[dict]:
        """Return ordered list of clause segmentation rule dicts."""
        return self._clause_rules.get("rules", [])

    def classify_clause_type(self, text: str) -> str:
        """
        Return clause_type string using clause_detection/clause_segmentation.json
        classifiers.  Uses weight-based best-match so higher-weighted classifiers
        (e.g. quoted_reference at 1.1) beat lower ones (e.g. obligation at 1.0)
        when both trigger words appear.  Falls back to 'other'.
        """
        lower = text.lower()
        best_type = "other"
        best_weight = -1.0
        for ct in self._clause_rules.get("clause_type_classifiers", []):
            w = ct.get("weight", 0.5)
            for word in ct.get("trigger_words", []):
                if word in lower and w > best_weight:
                    best_weight = w
                    best_type = ct["clause_type"]
                    break  # no need to check remaining trigger_words for this classifier
        return best_type

    # ── Validation Rule Lookup ─────────────────────────────────────────────

    def get_validation_rules(self) -> list[dict]:
        """Return all validation rules for use in validator.py."""
        return self._validation_rules

    def get_confidence_penalty(self, rule_id: str) -> float:
        """Return the confidence_penalty for a given validation rule id."""
        for rule in self._validation_rules:
            if rule["id"] == rule_id:
                return rule.get("confidence_penalty", 0.0)
        return 0.0

    # ── Utilities ──────────────────────────────────────────────────────────

    # Date parsing  → backend.core.utils.parse_date_to_iso
    # Deduplication → backend.core.utils.unique_ordered


def _unique_ordered(items: list[str]) -> list[str]:
    """Thin shim kept for any external callers; delegates to utils."""
    return unique_ordered(items)


# ── Singleton ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_rule_engine() -> RuleEngine:
    """Return the cached singleton RuleEngine instance."""
    return RuleEngine()
