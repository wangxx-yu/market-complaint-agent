from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.enums import AcceptSuggestion, DecisionSource, ReasonType


class ComplaintAnalyzeRequest(BaseModel):
    registration_id: str | None = None
    complaint_type: str | None = None
    channel: str | None = None
    enterprise_name: str | None = None
    enterprise_address: str | None = None
    incident_location: str | None = None
    appeal_text: str | None = None
    problem_text: str = Field(..., min_length=1)


class ClassificationResult(BaseModel):
    is_market: bool
    accept_suggestion: AcceptSuggestion
    reason_type: ReasonType = ReasonType.UNKNOWN
    confidence: float = Field(ge=0, le=1)
    decision_source: DecisionSource
    evidence_fields: list[str] = Field(default_factory=list)


class RejectReasonSuggestion(BaseModel):
    reason_type: ReasonType
    confidence: float = Field(ge=0, le=1)
    decision_source: DecisionSource
    evidence_fields: list[str] = Field(default_factory=list)
    needs_review: bool = True
    note: str | None = None


class DispatchResult(BaseModel):
    office_code: str
    office_name: str
    confidence: float = Field(ge=0, le=1)
    decision_source: DecisionSource
    matched_rule: str | None = None
    needs_review: bool = False


class RetrievalHit(BaseModel):
    knowledge_id: str | None = None
    title: str
    content: str
    score: float = Field(ge=0, le=1)
    source: str
    source_url: str | None = None
    law_status: str | None = None
    suggested_department: str | None = None
    explanation: str | None = None


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    reason_type: ReasonType = ReasonType.UNKNOWN
    top_k: int = Field(default=5, ge=1, le=10)


class RagSearchResponse(BaseModel):
    query: str
    mode: str
    hits: list[RetrievalHit] = Field(default_factory=list)
    reply_suggestion: str | None = None


class LawDocument(BaseModel):
    doc_id: str
    title: str
    article_count: int
    source: str | None = None
    source_url: str | None = None
    law_status: str | None = None


class LawDocumentResponse(BaseModel):
    documents: list[LawDocument] = Field(default_factory=list)


class LawFullTextResponse(BaseModel):
    doc_id: str
    title: str
    source: str | None = None
    source_url: str | None = None
    law_status: str | None = None
    articles: list[RetrievalHit] = Field(default_factory=list)


class ReplyDraft(BaseModel):
    text: str
    decision_source: DecisionSource
    template_id: str | None = None
    validation_passed: bool = True
    fallback_reason: str | None = None


class AgentStep(BaseModel):
    name: str
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    duration_ms: float
    error: str | None = None
    degraded: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AnalyzeResponse(BaseModel):
    trace_id: str
    classification: ClassificationResult
    reject_reason_suggestion: RejectReasonSuggestion | None = None
    dispatch: DispatchResult | None = None
    retrieval_hits: list[RetrievalHit] = Field(default_factory=list)
    reply_draft: ReplyDraft
    review_required: bool
    review_reasons: list[str] = Field(default_factory=list)
    agent_steps: list[AgentStep] = Field(default_factory=list)


class ReviewConfirmRequest(BaseModel):
    is_market: bool | None = None
    reason_type: ReasonType | None = None
    reject_detail: str | None = None
    office_code: str | None = None
    office_name: str | None = None
    reply_text: str | None = None
    reviewer: str | None = None
    notes: str | None = None
    correct_reason_type: ReasonType | None = None
    correct_office: str | None = None
    reviewer_note: str | None = None
    can_use_for_training: bool = True

    @field_validator("reason_type", mode="before")
    @classmethod
    def normalize_legacy_reason_type(cls, value: object) -> object:
        legacy_map = {
            "ARTICLE15_1_OUT_OF_SCOPE": ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,
            "ARTICLE15_2_NO_SUBJECT": ReasonType.ARTICLE16_5_MISSING_OR_FALSE_MATERIALS,
            "ARTICLE15_3_NOT_CONSUMER_PURCHASE": ReasonType.ARTICLE16_3_NOT_CONSUMER_DISPUTE,
            "ARTICLE15_4_EXPIRED": ReasonType.ARTICLE16_4_EXPIRED,
            "ARTICLE15_5_ALREADY_ACCEPTED": ReasonType.ARTICLE16_2_ALREADY_ACCEPTED_OR_PROCESSED,
        }
        if isinstance(value, str):
            return legacy_map.get(value, value)
        return value


class ReviewConfirmResponse(BaseModel):
    trace_id: str
    saved: bool
    saved_at: datetime


class HistoricalComplaint(BaseModel):
    registration_id: str
    complaint_type: str | None = None
    channel: str | None = None
    enterprise_name: str | None = None
    enterprise_address: str | None = None
    incident_location: str | None = None
    appeal_text: str | None = None
    problem_text: str
    problem_category: str | None = None
    accept_status: str | None = None
    handling_org: str | None = None
    handling_department: str | None = None
    feedback: str | None = None
