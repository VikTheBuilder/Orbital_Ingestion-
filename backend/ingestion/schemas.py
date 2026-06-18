"""
ORBITAL Pydantic v2 Schemas
Data models for the ingestion pipeline.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DeadlineSchema(BaseModel):
    """Structured deadline details for an obligation."""

    text: str
    absolute_date: Optional[str] = None
    duration: Optional[str] = None
    urgency: Literal[
        "immediate",
        "short_term",
        "medium_term",
        "long_term",
        "ongoing",
        "triggered",
    ] = "ongoing"


class ValidationFindingSchema(BaseModel):
    """A missed obligation detected during extraction validation."""

    clause_number: str
    raw_text: str
    reason_missed: str


class IncorrectExtractionSchema(BaseModel):
    """A field-level correction for an extracted obligation."""

    obligation_id: str
    field: str
    current_value: str
    correct_value: str
    reason: str


class ObligationSchema(BaseModel):
    """A single regulatory compliance obligation extracted from a document."""

    id: str
    section_id: str
    clause_number: str
    actor: str
    action: str
    obligation_type: Literal["mandatory", "discretionary", "conditional", "time_bound"]
    trigger: str
    deadline: DeadlineSchema
    domain: Literal[
        "KYC_AML",
        "Cybersecurity",
        "DataPrivacy",
        "FinancialInclusion",
        "BusinessContinuity",
        "FraudManagement",
        "CapitalAdequacy",
        "Payments",
        "CustomerService",
        "Governance",
        "ITInfrastructure",
        "ReportingAudit",
        "HR_Training",
        "FEMA",
        "Other",
    ]
    departments: List[
        Literal[
            "Compliance",
            "AML_KYC",
            "Cybersecurity",
            "IT",
            "DigitalBanking",
            "Operations",
            "BranchNetwork",
            "RiskManagement",
            "Legal",
            "Finance",
            "Treasury",
            "HR",
            "CustomerService",
            "FraudManagement",
            "InternalAudit",
            "RetailBanking",
        ]
    ] = []
    severity: Literal["low", "medium", "high", "critical"]
    severity_reason: str
    evidence_required: List[str] = []
    penalty_if_missed: Optional[str] = None
    fine_exposure_inr: Optional[float] = None
    cross_references: List[str] = []
    confidence: float
    notes: Optional[str] = None


class SectionSchema(BaseModel):
    """A structural section of a regulatory document."""

    id: str
    heading: str
    text: str
    page_number: Optional[int] = None
    clause_type: Literal[
        "definition",
        "obligation",
        "permission",
        "penalty",
        "cross_reference",
        "effective_date",
        "quoted_reference",
        "other",
    ] = "other"
    level: int = 1
    obligations: List[ObligationSchema] = []


class DocumentStructureSchema(BaseModel):
    """Complete parsed structure of a regulatory document."""

    doc_id: str
    source: Literal["RBI", "SEBI", "CERT-In", "NPCI", "IRDAI", "DPDP", "FIU-IND", "IBA", "OTHER"]
    title: str
    circular_number: Optional[str] = None
    reference_number: Optional[str] = None
    date: Optional[str] = None
    effective_date: Optional[str] = None
    issued_by: str = ""
    amends: Optional[str] = None
    language: str = "en"
    total_pages: int = 0
    sections: List[SectionSchema] = []
    tables: List[dict] = []
    annexures: List[str] = []
    cross_references: List[str] = []
    obligations: List[ObligationSchema] = []
    validation: Optional[dict] = None  # ValidationResultSchema serialised dict


class ChunkSchema(BaseModel):
    """A text chunk prepared for embedding and retrieval."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    section_heading: str
    text: str
    page_number: Optional[int] = None
    chunk_index: int
    total_chunks: int
    domain: str = "General"
    domain_tags: List[str] = []
    has_obligation: bool = False
    obligation_ids: List[str] = []
    date: Optional[str] = None
    chunk_type: str = "context"
    language: str = "en"


class PipelineResultSchema(BaseModel):
    """Summary result of running the full ingestion pipeline on a document."""

    doc_id: str
    source: str
    title: str
    total_pages: int
    total_sections: int
    total_obligations: int
    obligations_by_domain: dict
    obligations_by_severity: dict
    total_chunks: int
    processing_time_seconds: float
    structured_json_path: str
    finetune_pairs_path: str
    finetune_dropped_count: int = 0
    status: Literal["success", "partial", "failed"]
    warnings: List[str] = []
    # Validation summary (populated when prompt 3 runs)
    validation_missed_count: int = 0
    validation_incorrect_count: int = 0
    validation_confidence: float = 0.0


class ValidationResultSchema(BaseModel):
    """Review output for obligation extraction quality checks."""

    missed_obligations: List[ValidationFindingSchema] = []
    incorrect_extractions: List[IncorrectExtractionSchema] = []
    missing_effective_date: Optional[str] = None
    overall_confidence: float = 0.0
    validation_notes: str = ""
