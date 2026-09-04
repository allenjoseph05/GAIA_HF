"""Evidence-first structural, source, temporal, independent, and risk verification."""

from __future__ import annotations

from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaia_max.domain import (
    RiskAssessment,
    RiskLevel,
    TemporalConstraint,
    Verdict,
    VerificationResult,
)
from gaia_max.domain.models import EvidenceStatus
from gaia_max.retrieval.archive import TemporalEvidence, validate_temporal_evidence
from gaia_max.retrieval.evidence import (
    CandidateEvidenceBundle,
    ClaimEvidenceLink,
    EvidenceReadiness,
    EvidenceSupport,
    assess_evidence_readiness,
)


class CriticVerdict(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


class IndependentCritique(BaseModel):
    """Structured critic output; approval requires explicit claim coverage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    critic: str = Field(min_length=1)
    verdict: CriticVerdict
    reviewed_claim_ids: tuple[str, ...]
    missing_evidence: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def approval_has_no_gaps(self) -> IndependentCritique:
        if self.verdict is CriticVerdict.APPROVE and (
            not self.reviewed_claim_ids or self.missing_evidence or self.conflicts
        ):
            raise ValueError("critic approval requires reviewed claims and no evidence gaps")
        return self


class CandidateVerificationContext(BaseModel):
    """Answer-free inputs to the verification ladder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle: CandidateEvidenceBundle
    passed_structural_checks: tuple[str, ...] = ()
    structural_failures: tuple[str, ...] = ()
    deterministic_authority_passed: bool = False
    temporal_constraint: TemporalConstraint | None = None
    temporal_evidence: tuple[TemporalEvidence, ...] = ()
    require_independent_retrieval: bool = False
    critique: IndependentCritique | None = None
    conflict_resolutions: dict[str, str] = Field(default_factory=dict)


class VerificationOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verification: VerificationResult
    risk: RiskAssessment
    evidence_readiness: EvidenceReadiness


def verify_candidate(context: CandidateVerificationContext) -> VerificationOutcome:
    """Apply hard gates in authority order; no model vote can override a failure."""

    readiness = assess_evidence_readiness(context.bundle)
    passed = list(context.passed_structural_checks)
    negatives: list[str] = []
    hard: list[str] = list(context.structural_failures)
    missing: list[str] = list(readiness.missing_direct_support)
    conflicts = _unresolved_conflicts(context)

    if context.bundle.deterministic:
        if context.deterministic_authority_passed:
            passed.append("deterministic_authority_passed")
        else:
            hard.append("deterministic_authority_missing")
    elif readiness.directly_supported_claims:
        passed.append("required_claims_directly_supported")

    if missing:
        hard.append("missing_direct_claim_evidence")
    if conflicts:
        hard.append("unresolved_evidence_conflict")

    temporal_codes = _verify_temporal(context)
    for code, valid in temporal_codes:
        if valid:
            passed.append(code)
        else:
            hard.append(code)

    independence_missing = _independence_gaps(context)
    if independence_missing:
        negatives.append("independent_retrieval_incomplete")
        if context.require_independent_retrieval:
            hard.append("required_independent_retrieval_missing")
            missing.extend(independence_missing)
    elif context.require_independent_retrieval:
        passed.append("independent_retrieval_confirmed")

    critique = context.critique
    if critique is not None:
        required_claims = {
            claim.claim_id for claim in context.bundle.claims if claim.required
        }
        unreviewed = required_claims - set(critique.reviewed_claim_ids)
        if unreviewed:
            negatives.append("critic_claim_coverage_incomplete")
        if critique.verdict is CriticVerdict.REJECT:
            hard.append("independent_critic_rejected")
        elif critique.verdict is CriticVerdict.UNCERTAIN:
            negatives.append("independent_critic_uncertain")
        elif unreviewed:
            hard.append("independent_critic_unreviewed_claims")
        else:
            passed.append("independent_critic_approved")
        missing.extend(critique.missing_evidence)
        conflicts.extend(critique.conflicts)

    hard = list(dict.fromkeys(hard))
    negatives = list(dict.fromkeys(negatives))
    missing = list(dict.fromkeys(missing))
    conflicts = list(dict.fromkeys(conflicts))
    risk = assess_verification_risk(passed, negatives, hard)
    verdict = Verdict.REJECT if hard else (Verdict.UNCERTAIN if negatives else Verdict.APPROVE)
    verification = VerificationResult(
        verifier="evidence-first-ladder",
        tier=3,
        verdict=verdict,
        passed_checks=passed,
        reason_codes=[*hard, *negatives],
        critique=(
            "hard evidence gates passed"
            if verdict is Verdict.APPROVE
            else "candidate needs repair"
        ),
        missing_evidence=missing,
        evidence_conflicts=conflicts,
        requested_next_action=(
            None
            if verdict is Verdict.APPROVE
            else "repair only the listed evidence or verification gaps"
        ),
    )
    return VerificationOutcome(
        verification=verification,
        risk=risk,
        evidence_readiness=readiness,
    )


def assess_verification_risk(
    positive_features: list[str],
    negative_features: list[str],
    hard_failures: list[str],
) -> RiskAssessment:
    """Derive risk from evidence features while retaining non-overridable failures."""

    if hard_failures:
        level = RiskLevel.BLOCKED
    elif len(negative_features) >= 2:
        level = RiskLevel.HIGH
    elif negative_features:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW
    return RiskAssessment(
        level=level,
        positive_features=list(dict.fromkeys(positive_features)),
        negative_features=list(dict.fromkeys(negative_features)),
        hard_failures=list(dict.fromkeys(hard_failures)),
    )


def _unresolved_conflicts(context: CandidateVerificationContext) -> list[str]:
    conflicts = list(assess_evidence_readiness(context.bundle).conflicting_claims)
    links_by_claim: dict[str, list[ClaimEvidenceLink]] = {}
    evidence_by_id = {item.evidence_id: item for item in context.bundle.evidence}
    for link in context.bundle.links:
        links_by_claim.setdefault(link.claim_id, []).append(link)
    unresolved: list[str] = []
    for claim_id in conflicts:
        selected_id = context.conflict_resolutions.get(claim_id)
        selected = evidence_by_id.get(selected_id or "")
        selected_links = [
            link
            for link in links_by_claim[claim_id]
            if link.evidence_id == selected_id and link.support is EvidenceSupport.DIRECT
        ]
        if selected is None or not selected_links or selected.status is not EvidenceStatus.PRIMARY:
            unresolved.append(claim_id)
    unknown = set(context.conflict_resolutions) - set(conflicts)
    if unknown:
        unresolved.extend(sorted(f"unexpected_resolution:{claim}" for claim in unknown))
    return unresolved


def _verify_temporal(context: CandidateVerificationContext) -> list[tuple[str, bool]]:
    if context.temporal_constraint is None:
        return []
    if not context.temporal_evidence:
        return [("temporal_evidence_missing", False)]
    return [
        (
            f"temporal:{decision.code.value}",
            decision.valid,
        )
        for item in context.temporal_evidence
        for decision in [validate_temporal_evidence(item, context.temporal_constraint)]
    ]


def _independence_gaps(context: CandidateVerificationContext) -> list[str]:
    if not context.require_independent_retrieval:
        return []
    evidence_by_id = {item.evidence_id: item for item in context.bundle.evidence}
    gaps: list[str] = []
    for claim in context.bundle.claims:
        if not claim.required:
            continue
        authorities: set[tuple[str, str]] = set()
        for link in context.bundle.links:
            if link.claim_id != claim.claim_id or link.support is not EvidenceSupport.DIRECT:
                continue
            evidence = evidence_by_id[link.evidence_id]
            if evidence.status not in {
                EvidenceStatus.PRIMARY,
                EvidenceStatus.INDEPENDENTLY_VERIFIED,
            }:
                continue
            if evidence.source_url is not None:
                host = urlsplit(str(evidence.source_url)).hostname or "unknown"
            else:
                host = f"artifact:{evidence.artifact_id}"
            authorities.add((host.casefold(), evidence.extraction_method.casefold()))
        if len(authorities) < 2:
            gaps.append(claim.claim_id)
    return gaps


__all__ = [
    "CandidateVerificationContext",
    "CriticVerdict",
    "IndependentCritique",
    "VerificationOutcome",
    "assess_verification_risk",
    "verify_candidate",
]
