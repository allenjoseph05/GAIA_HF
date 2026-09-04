"""Strict contracts shared by research orchestration and framework adapters."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StrictStr,
    TypeAdapter,
    field_serializer,
    model_validator,
)

from gaia_max.domain import Evidence, SemanticAnswer
from gaia_max.domain.models import EvidenceStatus
from gaia_max.retrieval.evidence import CandidateEvidenceBundle

_SEMANTIC_ANSWER = TypeAdapter(SemanticAnswer)
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ResearchStatus(StrEnum):
    APPROVED = "approved"
    EXHAUSTED = "exhausted"
    BLOCKED = "blocked"


class ResearchObjective(BaseModel):
    """Trusted, code-created objective passed to the bounded research worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: StrictStr = Field(min_length=1, max_length=200)
    question: StrictStr = Field(min_length=1, max_length=20_000)
    required_output: StrictStr = Field(min_length=1, max_length=2000)
    unresolved_claims: tuple[StrictStr, ...] = ()
    attempt: int = Field(default=1, ge=1, le=20)


class ResearchClaimDraft(BaseModel):
    """Agent-proposed claim whose citations must resolve in a code-owned catalog."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: StrictStr = Field(pattern=r"^claim_[a-z0-9_]{1,80}$")
    statement: StrictStr = Field(min_length=1, max_length=2000)
    required: bool = True
    citation_ids: tuple[StrictStr, ...] = Field(default=(), max_length=20)


class ResearchDraft(BaseModel):
    """Only accepted output shape from an open-ended research framework."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    semantic_answer: SemanticAnswer
    claims: tuple[ResearchClaimDraft, ...] = Field(min_length=1, max_length=50)
    unresolved_claim_ids: tuple[StrictStr, ...] = Field(default=(), max_length=50)

    @model_validator(mode="after")
    def validate_claim_graph(self) -> ResearchDraft:
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("research claim IDs must be unique")
        unknown = set(self.unresolved_claim_ids) - set(claim_ids)
        if unknown:
            raise ValueError("unresolved claim IDs must reference declared claims")
        for claim in self.claims:
            if (
                claim.required
                and not claim.citation_ids
                and claim.claim_id not in self.unresolved_claim_ids
            ):
                raise ValueError("required claims need citations or an unresolved marker")
        return self


class IndexableDocument(BaseModel):
    """Model-visible text already passed through the source-safety boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: StrictStr = Field(min_length=1, max_length=200)
    text: StrictStr = Field(min_length=1)
    model_text_artifact_id: str = Field(pattern=SHA256_PATTERN)
    source_url: HttpUrl | None = None
    page_number: int | None = Field(default=None, ge=1)
    source_date: str | None = Field(default=None, max_length=100)
    retrieved_at: AwareDatetime
    is_primary_source: bool = False
    safety_disposition: str = Field(pattern=r"^(clean|sanitized)$")

    @field_serializer("source_url")
    def serialize_url(self, value: HttpUrl | None) -> str | None:
        return str(value) if value is not None else None


class RetrievedPassage(BaseModel):
    """Auditable LlamaIndex hit preserving exact source provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: str = Field(pattern=SHA256_PATTERN)
    node_id: StrictStr = Field(min_length=1, max_length=300)
    document_id: StrictStr = Field(min_length=1, max_length=200)
    model_text_artifact_id: str = Field(pattern=SHA256_PATTERN)
    source_url: HttpUrl | None = None
    page_number: int | None = Field(default=None, ge=1)
    source_date: str | None = Field(default=None, max_length=100)
    retrieved_at: AwareDatetime
    is_primary_source: bool
    excerpt: StrictStr = Field(min_length=1, max_length=8000)
    locator: StrictStr = Field(min_length=1, max_length=1000)
    lexical_score: float | None = None
    vector_score: float | None = None
    fused_score: float = Field(ge=0)
    rank: int = Field(ge=1)

    @field_serializer("source_url")
    def serialize_url(self, value: HttpUrl | None) -> str | None:
        return str(value) if value is not None else None

    def to_evidence(self) -> Evidence:
        return Evidence(
            evidence_id=self.passage_id,
            claim="retrieved source passage",
            source_url=self.source_url,
            artifact_id=(None if self.source_url is not None else self.model_text_artifact_id),
            source_type=(
                "primary_document_passage"
                if self.is_primary_source
                else "document_passage"
            ),
            source_date=self.source_date,
            retrieved_at=self.retrieved_at,
            is_primary_source=self.is_primary_source,
            excerpt=self.excerpt,
            locator=self.locator,
            extraction_method="llamaindex_hybrid_retrieval",
            status=(EvidenceStatus.PRIMARY if self.is_primary_source else EvidenceStatus.INDIRECT),
        )


class ResearchRunRecord(BaseModel):
    """Inspectable run result retained outside graph answer/submission state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: StrictStr = Field(min_length=1, max_length=200)
    status: ResearchStatus
    attempts: int = Field(ge=1, le=20)
    draft: ResearchDraft | None = None
    evidence_bundle: CandidateEvidenceBundle | None = None
    reason_codes: tuple[StrictStr, ...] = ()
    completed_at: datetime

    @model_validator(mode="after")
    def approved_has_evidence(self) -> ResearchRunRecord:
        if self.status is ResearchStatus.APPROVED and (
            self.draft is None or self.evidence_bundle is None or self.reason_codes
        ):
            raise ValueError("approved research requires a draft, evidence, and no failures")
        return self


def validate_semantic_answer(value: object) -> SemanticAnswer:
    return _SEMANTIC_ANSWER.validate_python(value)


__all__ = [
    "IndexableDocument",
    "ResearchClaimDraft",
    "ResearchDraft",
    "ResearchObjective",
    "ResearchRunRecord",
    "ResearchStatus",
    "RetrievedPassage",
    "validate_semantic_answer",
]
