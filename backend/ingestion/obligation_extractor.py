"""
ORBITAL Obligation Extractor
(Rewritten to use LLM primary extraction with document context, per user request)
"""
import json
import logging
import os
import re
import uuid
import hashlib
import time
from typing import List, Optional, Dict, Any

from backend.ingestion.schemas import DocumentStructureSchema, SectionSchema, ObligationSchema, DeadlineSchema
from backend.core.llm_client import llm
from backend.core.config import get_config
from backend.core.logger import get_logger

logger = get_logger(__name__)

DOMAIN_TO_DEPT = {
    "KYC": "Compliance",
    "AML": "Compliance",
    "InfoSec": "IT Security",
    "Credit": "Credit Risk",
    "Payments": "Payments",
    "Fraud": "Fraud Management",
    "HR": "Human Resources",
    "Governance": "Board & Governance",
    "Basel": "Finance & Accounts",
    "FOREX": "Treasury",
    "General": "Compliance",
    "CapitalAdequacy": "Finance & Accounts",
    "FinancialInclusion": "Financial Inclusion",
    "ConsumerProtection": "Customer Service",
    "Operations": "Operations",
}

DOMAIN_TO_EVIDENCE = {
    "KYC": "Updated KYC records",
    "AML": "STR filings and monitoring reports",
    "InfoSec": "Security log",
    "Credit": "Credit file",
    "Payments": "Payment record",
    "Fraud": "Fraud report",
    "HR": "HR records",
    "Governance": "Board note",
    "Basel": "Basel report",
    "FOREX": "FEMA filing",
    "General": "Implementation report",
    "CapitalAdequacy": "Capital computation sheet",
    "FinancialInclusion": "Branch service continuity note",
    "ConsumerProtection": "Customer service advisory",
    "Operations": "Operations log",
}

SCHEMA_DOMAINS = {
    "KYC_AML", "Cybersecurity", "DataPrivacy", "FinancialInclusion",
    "BusinessContinuity", "FraudManagement", "CapitalAdequacy", "Payments",
    "CustomerService", "Governance", "ITInfrastructure", "ReportingAudit",
    "HR_Training", "FEMA", "Other"
}

def map_domain(domain: str) -> str:
    mapping = {
        "KYC": "KYC_AML",
        "AML": "KYC_AML",
        "InfoSec": "Cybersecurity",
        "Fraud": "FraudManagement",
        "HR": "HR_Training",
        "Basel": "CapitalAdequacy",
        "FOREX": "FEMA",
        "ConsumerProtection": "CustomerService",
    }
    mapped = mapping.get(domain, domain)
    if mapped in SCHEMA_DOMAINS:
        return mapped
    return "Other"

def extract_obligations(doc_structure: DocumentStructureSchema) -> List[ObligationSchema]:
    # ─── PHASE 1 — Document-level context ─────
    full_text_summary = build_document_summary(doc_structure)
    doc_context = get_document_context(full_text_summary)
    
    # ─── PHASE 2 — Section-level extraction ───
    all_obligations = []
    
    for section in doc_structure.sections:
        if should_skip_section(section):
            continue
        
        llm_obligations = extract_with_llm(
            section_text=section.text,
            section_heading=section.heading,
            doc_context=doc_context,
            doc_id=doc_structure.doc_id,
            source=doc_structure.source,
            section_id=section.id
        )
        
        regex_obligations = extract_with_regex(
            section_text=section.text,
            section_heading=section.heading,
            doc_context=doc_context,
            doc_id=doc_structure.doc_id,
            source=doc_structure.source,
            section_id=section.id
        )
        
        merged = merge_obligations(llm_obligations, regex_obligations)
        all_obligations.extend(merged)
    
    deduped = deduplicate(all_obligations)
    sorted_obs = sort_by_severity(deduped)
    
    # Assign unique ids and assign back to sections
    for seq, ob in enumerate(sorted_obs, 1):
        ob.id = f"{ob.section_id}-OB{seq}"
    
    for section in doc_structure.sections:
        section.obligations = [ob for ob in sorted_obs if ob.section_id == section.id]
        
    return sorted_obs


def build_document_summary(doc_structure: DocumentStructureSchema) -> str:
    all_text = ""
    for section in doc_structure.sections[:5]:
        all_text += section.text + "\n\n"
    return all_text[:2000]


def get_document_context(full_text_summary: str) -> dict:
    system_prompt = """You are an expert at reading Indian regulatory circulars from RBI, SEBI, CERT-In, DPDP, FIU-IND.
Return ONLY valid JSON. No explanation. No markdown."""

    user_prompt = f"""Read the following regulatory document header and identify key context. Return JSON with exactly these fields:

{{
  "primary_actor": "who the circular is addressed TO — e.g. All Authorised Dealer Category I Banks",
  "primary_domain": "one of: KYC|AML|InfoSec|Credit|Payments|Fraud|HR|Governance|Basel|FOREX|FinancialInclusion|CapitalAdequacy|Operations|ConsumerProtection|General",
  "document_type": "circular|amendment|direction|master_direction|guideline",
  "subject": "the subject line of the circular in max 15 words",
  "effective_date": "date string or null",
  "issued_to": "the exact addressee line"
}}

DOCUMENT HEADER:
{full_text_summary}
"""
    try:
        response_text = llm.call(mode="extraction", prompt=system_prompt + "\n\n" + user_prompt)
        from backend.core.llm_client import OrbitalLLMClient
        parsed = OrbitalLLMClient._parse_json(response_text)
        if isinstance(parsed, dict) and "primary_actor" in parsed:
            return parsed
    except Exception as e:
        logger.error(f"Context extraction failed: {e}")
        
    domain = "General"
    if "capital adequacy" in full_text_summary.lower(): domain = "CapitalAdequacy"
    elif "payment" in full_text_summary.lower(): domain = "Payments"
    
    return {
        "primary_actor": "Regulated Entity",
        "primary_domain": domain,
        "document_type": "circular",
        "subject": "",
        "effective_date": None,
        "issued_to": "Regulated Entity"
    }


def clean_action(action: str, raw_text: str) -> str:
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, action[:100], raw_text[:100]).ratio()
    
    if similarity > 0.8:
        SUBJECT_PATTERNS = [
            r'^The financial statements\s+',
            r'^Such profits\s+',
            r'^Losses in the current year\s+',
            r'^All RE\s+',
            r'^A bank\s+',
            r'^The designated [A-Za-z\s]+bank\s+',
            r'^Banks\s+',
            r'^NBFCs\s+',
        ]
        
        cleaned = action
        for pattern in SUBJECT_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        if len(cleaned) > 120:
            cut = cleaned.find(' and ', 80)
            if cut > 0:
                cleaned = cleaned[:cut].strip()
        
        return cleaned if len(cleaned) > 10 else action
    
    return action


def extract_with_llm(section_text, section_heading, doc_context, doc_id, source, section_id):
    system_prompt = """You are a compliance obligation extractor for Indian banks. You read regulatory circulars and extract binding obligations.

Extract ONLY sentences where the bank or entity MUST do something. Look for: shall, must, required, prohibited, mandated.

Also extract "may" clauses ONLY when they are compliance-relevant — meaning the bank must have a process in place to exercise that option (e.g. "may open small accounts" means the bank must have the process ready, even if discretionary).

Return ONLY valid JSON array. Empty array [] if no obligations. No markdown. No explanation.

Each obligation:
{
  "actor": "exact entity that must act",
  "action": "clean verb phrase — what they must do, max 30 words, starts with a verb",
  "trigger_word": "shall|must|required|prohibited|may",
  "obligation_type": "mandatory|discretionary|prohibited",
  "deadline": "deadline string or null",
  "domain": "from context below",
  "department": "from context below",
  "severity": "critical|high|medium|low",
  "evidence_required": "what proof is needed",
  "raw_text": "the exact source sentence",
  "confidence": 0.0 to 1.0
}"""

    user_prompt = f"""DOCUMENT CONTEXT:
- Circular about: {doc_context.get("subject", "")}
- Issued to: {doc_context.get("issued_to", "")}
- Primary actor: {doc_context.get("primary_actor", "")}
- Domain: {doc_context.get("primary_domain", "")}
- Document type: {doc_context.get("document_type", "")}

Use the above context to correctly identify:
- actor: if sentence says "A bank" or "banks" → use the issued_to value from context above
- domain: default to primary_domain from context unless sentence clearly belongs to different domain
- department: map from domain

SECTION HEADING: {section_heading}

SECTION TEXT:
{section_text}

Extract all compliance obligations from this section."""

    try:
        response_text = llm.call(mode="extraction", prompt=system_prompt + "\n\n" + user_prompt)
        from backend.core.llm_client import OrbitalLLMClient
        parsed = OrbitalLLMClient._parse_json(response_text)
        if not isinstance(parsed, list):
            parsed = [parsed] if isinstance(parsed, dict) else []
            
        for ob in parsed:
            ob["action"] = clean_action(ob.get("action", ""), ob.get("raw_text", ""))
            ob["section_id"] = section_id
            
        return parsed
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return []


def split_into_obligation_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    embedded_pattern = re.compile(
        r'[^.!?]*\b(?:shall|must|required to|prohibited|may)\b[^.!?]*[.!?]',
        re.IGNORECASE
    )
    embedded = embedded_pattern.findall(text)
    for match in embedded:
        m_stripped = match.strip()
        if m_stripped not in sentences and len(m_stripped) > 20:
            sentences.append(m_stripped)
            
    return sentences

def has_trigger_word(text: str) -> bool:
    TRIGGER_WORDS = ["shall", "must", "required", "prohibited", "mandated", "obligated", "may", "should"]
    return any(t in text.lower() for t in TRIGGER_WORDS)

def get_trigger(text: str) -> str:
    lower = text.lower()
    for t in ["shall", "must", "required", "prohibited", "mandated", "obligated", "may", "should"]:
        if t in lower: return t
    return "always"

def classify_domain_from_sentence(text: str) -> str:
    text_lower = text.lower()
    if "capital adequacy" in text_lower or "owned fund" in text_lower or "quarterly profits" in text_lower or "statutory auditors" in text_lower or "free reserves" in text_lower:
        return "CapitalAdequacy"
    if "financial inclusion" in text_lower or "calamity" in text_lower or "satellite office" in text_lower:
        return "FinancialInclusion"
    if "customer compensation" in text_lower or "grievance" in text_lower or "ombudsman" in text_lower:
        return "ConsumerProtection"
    if "atm" in text_lower or "iccw" in text_lower or "national financial switch" in text_lower:
        return "Operations"
    return "General"

def extract_actor(text: str) -> str:
    ACTOR_PATTERNS = [
        (r'\b(?:all\s+)?(?:scheduled\s+)?commercial\s+banks?\b', "Commercial Banks"),
        (r'\b(?:all\s+)?(?:authorised?\s+)?dealer\s+(?:category\s+[iI]+\s+)?banks?\b', "Authorised Dealer Banks"),
        (r'\bNBFC[s]?\b', "NBFCs"),
        (r'\b(?:regulated?\s+)?entit(?:y|ies)\b', "Regulated Entity"),
    ]
    for pattern, actor in ACTOR_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return actor
    return "Regulated Entity"

def _extract_action_regex(sentence: str, trigger: str) -> str:
    idx = sentence.lower().find(trigger)
    if idx >= 0:
        return sentence[idx + len(trigger):].strip().capitalize()
    return sentence

def extract_deadline(text: str) -> Optional[str]:
    m = re.search(r'(within \d+ days|with immediate effect|at the earliest)', text, re.IGNORECASE)
    return m.group(1) if m else "ongoing"

def calculate_severity(text: str) -> str:
    if "prohibited" in text.lower() or "at the earliest" in text.lower():
        return "critical"
    if "shall" in text.lower():
        return "high"
    return "medium"

def extract_with_regex(section_text, section_heading, doc_context, doc_id, source, section_id):
    sentences = split_into_obligation_sentences(section_text)
    obligations = []
    for sentence in sentences:
        if not has_trigger_word(sentence):
            continue
            
        domain = classify_domain_from_sentence(sentence)
        if domain == "General":
            domain = doc_context.get("primary_domain", "General")
            
        actor = extract_actor(sentence)
        if actor == "Regulated Entity":
            actor = doc_context.get("primary_actor", "Regulated Entity")
            
        trigger = get_trigger(sentence)
        deadline_str = extract_deadline(sentence) or "ongoing"
        
        ob = ObligationSchema(
            id=str(uuid.uuid4()),
            section_id=section_id,
            clause_number=section_id,
            actor=actor,
            action=_extract_action_regex(sentence, trigger),
            obligation_type="mandatory" if trigger != "may" else "discretionary",
            trigger=trigger,
            deadline=DeadlineSchema(text=deadline_str, urgency="ongoing" if deadline_str=="ongoing" else "short_term"),
            domain=map_domain(domain),
            departments=[DOMAIN_TO_DEPT.get(domain, "Compliance")],
            severity=calculate_severity(sentence),
            severity_reason="Regex extracted",
            evidence_required=[DOMAIN_TO_EVIDENCE.get(domain, "Implementation report")],
            confidence=0.65,
            notes=None,
        )
        ob.__dict__["raw_text"] = sentence
        obligations.append(ob)
    return obligations


def should_skip_section(section) -> bool:
    SKIP_HEADINGS = [
        "yours faithfully", "chief general manager", "general manager",
        "deputy governor", "www.rbi.org.in", "telephone", "tel no",
        "reserve bank of india"
    ]
    SKIP_PATTERNS = [
        r"^\s*www\.", r"^\s*टेलीफोन", r"^\s*फैक्स", r"हिंदी", r"बेटी बचाओ", r"^\s*\d+\s*$"
    ]
    heading_lower = section.heading.lower().strip()
    if heading_lower in SKIP_HEADINGS:
        return True
        
    if len(section.text.strip()) < 50:
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, section.text, re.IGNORECASE):
                return True
                
    TRIGGER_WORDS = ["shall", "must", "required", "prohibited", "mandated", "obligated", "may", "should"]
    has_trigger = any(t in section.text.lower() for t in TRIGGER_WORDS)
    if not has_trigger and len(section.text) < 100:
        return True
        
    return False


def dict_to_obligation_schema(llm_ob: dict) -> ObligationSchema:
    domain_raw = llm_ob.get("domain", "General")
    domain = map_domain(domain_raw)
    
    deadline_str = str(llm_ob.get("deadline", "ongoing"))
    
    sev = llm_ob.get("severity", "medium").lower()
    if sev not in {"critical", "high", "medium", "low"}: sev = "medium"
    
    otype = llm_ob.get("obligation_type", "mandatory").lower()
    if otype not in {"mandatory", "discretionary", "conditional", "time_bound"}: otype = "mandatory"
    
    ev = llm_ob.get("evidence_required", [])
    if ev is None:
        ev = []
    elif isinstance(ev, str):
        ev = [ev]
    
    ob = ObligationSchema(
        id=str(uuid.uuid4()),
        section_id=llm_ob.get("section_id", ""),
        clause_number=llm_ob.get("section_id", ""),
        actor=llm_ob.get("actor", "Regulated Entity"),
        action=llm_ob.get("action", ""),
        obligation_type=otype,
        trigger=llm_ob.get("trigger_word", "always"),
        deadline=DeadlineSchema(text=deadline_str, urgency="ongoing" if deadline_str=="ongoing" else "short_term"),
        domain=domain,
        departments=[DOMAIN_TO_DEPT.get(domain_raw, "Compliance")],
        severity=sev,
        severity_reason="LLM Extracted",
        evidence_required=ev,
        confidence=float(llm_ob.get("confidence", 0.8)),
    )
    ob.__dict__["raw_text"] = llm_ob.get("raw_text", "")
    return ob


def find_matching_regex_ob(ob: ObligationSchema, regex_obs: List[ObligationSchema]):
    from difflib import SequenceMatcher
    best_match = None
    best_ratio = 0
    raw_text = getattr(ob, "raw_text", ob.action)
    for rob in regex_obs:
        rob_raw = getattr(rob, "raw_text", rob.action)
        ratio = SequenceMatcher(None, raw_text[:100], rob_raw[:100]).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = rob
    if best_ratio > 0.6:
        return best_match
    return None

def merge_obligations(llm_obs: List[dict], regex_obs: List[ObligationSchema]) -> List[ObligationSchema]:
    final = []
    for llm_ob in llm_obs:
        if not llm_ob.get("action"): continue
        ob = dict_to_obligation_schema(llm_ob)
        match = find_matching_regex_ob(ob, regex_obs)
        if match:
            ob.__dict__["raw_text"] = getattr(match, "raw_text", match.action)
            ob.confidence = max(ob.confidence, match.confidence)
        if ob.confidence >= 0.5:
            final.append(ob)
            
    for regex_ob in regex_obs:
        found_in_llm = False
        rob_raw = getattr(regex_ob, "raw_text", regex_ob.action)
        for ob in final:
            ob_raw = getattr(ob, "raw_text", ob.action)
            from difflib import SequenceMatcher
            if SequenceMatcher(None, rob_raw[:100], ob_raw[:100]).ratio() > 0.6:
                found_in_llm = True
                break
        if not found_in_llm:
            regex_ob.confidence = min(regex_ob.confidence, 0.65)
            final.append(regex_ob)
            logger.info("Regex caught obligation LLM missed")
            
    return final

def deduplicate(obligations: List[ObligationSchema]) -> List[ObligationSchema]:
    from difflib import SequenceMatcher
    keep = [True] * len(obligations)
    for i in range(len(obligations)):
        if not keep[i]: continue
        for j in range(i+1, len(obligations)):
            if not keep[j]: continue
            if SequenceMatcher(None, obligations[i].action.lower()[:100], obligations[j].action.lower()[:100]).ratio() > 0.75:
                if obligations[i].confidence >= obligations[j].confidence:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [obs for idx, obs in enumerate(obligations) if keep[idx]]

def sort_by_severity(obligations: List[ObligationSchema]) -> List[ObligationSchema]:
    rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    obligations.sort(key=lambda o: (rank.get(o.severity, 4), -o.confidence))
    return obligations
