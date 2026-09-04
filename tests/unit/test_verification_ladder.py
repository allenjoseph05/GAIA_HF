from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from gaia_max.domain import Evidence, TemporalConstraint, Verdict
from gaia_max.retrieval import (
    CandidateEvidenceBundle,
    ClaimEvidenceLink,
    EvidenceSupport,
    ResearchClaim,
)
from gaia_max.retrieval.archive import TemporalEvidence, TemporalEvidenceKind
from gaia_max.verification import (
    CandidateVerificationContext,
    CriticVerdict,
    IndependentCritique,
    assess_verification_risk,
    verify_candidate,
)

NOW = datetime(2026, 8, 30, tzinfo=UTC)


def _evidence(
    evidence_id: str,
    *,
    host: str = "primary.example",
    method: str = "exact_page",
    status: str = "primary",
) -> Evidence:
    return Evidence.model_validate(
        {
            "evidence_id": evidence_id,
            "claim": "Synthetic exact statement.",
            "source_url": f"https://{host}/{evidence_id}",
            "source_type": "primary_document",
            "retrieved_at": NOW,
            "is_primary_source": status == "primary",
            "excerpt": "Synthetic exact statement.",
            "locator": "page 1",
            "extraction_method": method,
            "status": status,
        }
    )


def _bundle(
    *evidence: Evidence,
    contradictory: bool = False,
    deterministic: bool = False,
) -> CandidateEvidenceBundle:
    links = [
        ClaimEvidenceLink(
            claim_id="answer",
            evidence_id=item.evidence_id,
            support=EvidenceSupport.DIRECT,
            locator="page 1",
        )
        for item in evidence
    ]
    if contradictory:
        links[-1] = ClaimEvidenceLink(
            claim_id="answer",
            evidence_id=evidence[-1].evidence_id,
            support=EvidenceSupport.CONTRADICTS,
            locator="page 1",
        )
    return CandidateEvidenceBundle(
        candidate_id="candidate",
        deterministic=deterministic,
        claims=(ResearchClaim(claim_id="answer", statement="The answer claim."),),
        evidence=evidence,
        links=tuple(links),
    )


def test_direct_and_independent_evidence_approves_low_risk_candidate() -> None:
    bundle = _bundle(
        _evidence("primary"),
        _evidence(
            "independent",
            host="independent.example",
            method="independent_query",
            status="independently_verified",
        ),
    )
    critique = IndependentCritique(
        critic="critic-b",
        verdict=CriticVerdict.APPROVE,
        reviewed_claim_ids=("answer",),
        rationale="Both exact passages support the claim.",
    )

    result = verify_candidate(
        CandidateVerificationContext(
            bundle=bundle,
            passed_structural_checks=("output_contract_valid",),
            require_independent_retrieval=True,
            critique=critique,
        )
    )

    assert result.verification.verdict is Verdict.APPROVE
    assert result.risk.level.value == "low"
    assert "independent_retrieval_confirmed" in result.verification.passed_checks


def test_missing_direct_evidence_is_a_non_overridable_hard_failure() -> None:
    result = verify_candidate(CandidateVerificationContext(bundle=_bundle()))
    assert result.verification.verdict is Verdict.REJECT
    assert result.risk.level.value == "blocked"
    assert "missing_direct_claim_evidence" in result.risk.hard_failures


def test_deterministic_candidate_requires_authority_replay() -> None:
    bundle = _bundle(deterministic=True)
    blocked = verify_candidate(CandidateVerificationContext(bundle=bundle))
    passed = verify_candidate(
        CandidateVerificationContext(bundle=bundle, deterministic_authority_passed=True)
    )
    assert blocked.verification.verdict is Verdict.REJECT
    assert passed.verification.verdict is Verdict.APPROVE


def test_current_unqualified_source_fails_historical_constraint() -> None:
    temporal = TemporalEvidence(
        evidence_id="page",
        kind=TemporalEvidenceKind.CURRENT_UNQUALIFIED,
        retrieved_at=NOW,
    )
    constraint = TemporalConstraint(kind="as_of", end=datetime(2020, 1, 1, tzinfo=UTC))
    result = verify_candidate(
        CandidateVerificationContext(
            bundle=_bundle(_evidence("page")),
            temporal_constraint=constraint,
            temporal_evidence=(temporal,),
        )
    )
    assert "temporal:current_state_unqualified" in result.risk.hard_failures


def test_conflict_needs_explicit_primary_evidence_resolution() -> None:
    bundle = _bundle(
        _evidence("primary"),
        _evidence("conflict", status="independently_verified"),
        contradictory=True,
    )
    blocked = verify_candidate(CandidateVerificationContext(bundle=bundle))
    resolved = verify_candidate(
        CandidateVerificationContext(
            bundle=bundle,
            conflict_resolutions={"answer": "primary"},
        )
    )
    assert blocked.verification.verdict is Verdict.REJECT
    assert resolved.verification.verdict is Verdict.APPROVE


def test_required_independent_retrieval_cannot_repeat_one_authority() -> None:
    bundle = _bundle(_evidence("only"))
    result = verify_candidate(
        CandidateVerificationContext(bundle=bundle, require_independent_retrieval=True)
    )
    assert result.verification.verdict is Verdict.REJECT
    assert result.verification.missing_evidence == ["answer"]


def test_critic_cannot_approve_without_reviewed_claims() -> None:
    with pytest.raises(ValidationError, match="reviewed claims"):
        IndependentCritique(
            critic="critic-b",
            verdict=CriticVerdict.APPROVE,
            reviewed_claim_ids=(),
            rationale="Looks plausible.",
        )


def test_risk_features_do_not_erase_hard_failures() -> None:
    risk = assess_verification_risk(
        ["many_positive_checks"],
        [],
        ["authority_mismatch"],
    )
    assert risk.level.value == "blocked"
    assert risk.hard_failures == ["authority_mismatch"]
