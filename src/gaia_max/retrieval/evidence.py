"""Claim-level evidence links and approval readiness for research candidates."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from gaia_max.domain import Evidence
from gaia_max.domain.models import EvidenceStatus


class EvidenceSupport(StrEnum):
    """How one evidence item relates to one explicit claim."""

    DIRECT = "direct"
    INDIRECT = "indirect"
    CONTRADICTS = "contradicts"


class ResearchClaim(BaseModel):
    """One independently auditable proposition used by a candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: StrictStr = Field(min_length=1, max_length=200)
    statement: StrictStr = Field(min_length=1, max_length=2000)
    required: bool = True


class ClaimEvidenceLink(BaseModel):
    """Typed edge from a claim to exact supporting or conflicting evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: StrictStr = Field(min_length=1, max_length=200)
    evidence_id: StrictStr = Field(min_length=1, max_length=200)
    support: EvidenceSupport
    locator: StrictStr | None = Field(default=None, min_length=1, max_length=1000)
    excerpt: StrictStr | None = Field(default=None, min_length=1, max_length=4000)

    @model_validator(mode="after")
    def direct_links_need_location(self) -> ClaimEvidenceLink:
        if self.support is EvidenceSupport.DIRECT and not (self.locator or self.excerpt):
            raise ValueError("direct evidence link requires an excerpt or locator")
        return self


class CandidateEvidenceBundle(BaseModel):
    """Answer-free provenance graph for one deterministic or research candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: StrictStr = Field(min_length=1, max_length=200)
    deterministic: bool = False
    claims: tuple[ResearchClaim, ...] = Field(min_length=1)
    evidence: tuple[Evidence, ...] = ()
    links: tuple[ClaimEvidenceLink, ...] = ()

    @model_validator(mode="after")
    def validate_graph(self) -> CandidateEvidenceBundle:
        claim_ids = [claim.claim_id for claim in self.claims]
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("candidate evidence claims must have unique IDs")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("candidate evidence items must have unique IDs")
        for link in self.links:
            if link.claim_id not in claim_ids or link.evidence_id not in evidence_ids:
                raise ValueError("claim-evidence link references an unknown node")
        return self


class EvidenceReadiness(BaseModel):
    """Deterministic reasons a candidate can or cannot advance to verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ready: bool
    directly_supported_claims: tuple[str, ...] = ()
    missing_direct_support: tuple[str, ...] = ()
    conflicting_claims: tuple[str, ...] = ()


def assess_evidence_readiness(bundle: CandidateEvidenceBundle) -> EvidenceReadiness:
    """Require direct authoritative evidence for every nondeterministic required claim."""

    evidence_by_id = {item.evidence_id: item for item in bundle.evidence}
    direct: list[str] = []
    missing: list[str] = []
    conflicted: list[str] = []
    for claim in bundle.claims:
        if not claim.required:
            continue
        links = [link for link in bundle.links if link.claim_id == claim.claim_id]
        if any(link.support is EvidenceSupport.CONTRADICTS for link in links):
            conflicted.append(claim.claim_id)
        valid_direct = any(
            link.support is EvidenceSupport.DIRECT
            and _direct_authority(evidence_by_id[link.evidence_id])
            for link in links
        )
        if valid_direct:
            direct.append(claim.claim_id)
        elif not bundle.deterministic:
            missing.append(claim.claim_id)
    return EvidenceReadiness(
        ready=not missing and not conflicted,
        directly_supported_claims=tuple(direct),
        missing_direct_support=tuple(missing),
        conflicting_claims=tuple(conflicted),
    )


def _direct_authority(evidence: Evidence) -> bool:
    source_type = evidence.source_type.casefold().replace("-", "_")
    if "snippet" in source_type or source_type in {"search", "discovery"}:
        return False
    return evidence.status in {
        EvidenceStatus.PRIMARY,
        EvidenceStatus.INDEPENDENTLY_VERIFIED,
    }


__all__ = [
    "CandidateEvidenceBundle",
    "ClaimEvidenceLink",
    "EvidenceReadiness",
    "EvidenceSupport",
    "ResearchClaim",
    "assess_evidence_readiness",
]
