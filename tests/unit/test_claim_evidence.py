from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from gaia_max.domain import Evidence
from gaia_max.retrieval import (
    CandidateEvidenceBundle,
    ClaimEvidenceLink,
    EvidenceSupport,
    ResearchClaim,
    assess_evidence_readiness,
)

NOW = datetime(2026, 8, 30, tzinfo=UTC)


def evidence(
    evidence_id: str,
    *,
    source_type: str = "primary_paper",
    status: str = "primary",
) -> Evidence:
    return Evidence.model_validate(
        {
            "evidence_id": evidence_id,
            "claim": "Synthetic source passage.",
            "source_url": f"https://example.org/{evidence_id}",
            "source_type": source_type,
            "retrieved_at": NOW,
            "is_primary_source": status == "primary",
            "excerpt": "Exact synthetic passage.",
            "locator": "page 2",
            "extraction_method": "pdf_exact_context",
            "status": status,
        }
    )


def claim(claim_id: str) -> ResearchClaim:
    return ResearchClaim(claim_id=claim_id, statement=f"Claim {claim_id}")


def link(
    claim_id: str,
    evidence_id: str,
    support: EvidenceSupport = EvidenceSupport.DIRECT,
) -> ClaimEvidenceLink:
    return ClaimEvidenceLink(
        claim_id=claim_id,
        evidence_id=evidence_id,
        support=support,
        locator="page 2",
    )


def test_every_required_research_claim_needs_direct_authoritative_evidence() -> None:
    bundle = CandidateEvidenceBundle(
        candidate_id="candidate-1",
        claims=(claim("identity"), claim("relationship")),
        evidence=(evidence("paper"),),
        links=(link("identity", "paper"), link("relationship", "paper")),
    )

    readiness = assess_evidence_readiness(bundle)

    assert readiness.ready is True
    assert readiness.directly_supported_claims == ("identity", "relationship")
    assert readiness.missing_direct_support == ()


def test_search_snippet_cannot_approve_even_if_mislabeled_primary() -> None:
    bundle = CandidateEvidenceBundle(
        candidate_id="candidate-1",
        claims=(claim("answer"),),
        evidence=(evidence("snippet", source_type="search_snippet"),),
        links=(link("answer", "snippet"),),
    )

    readiness = assess_evidence_readiness(bundle)

    assert readiness.ready is False
    assert readiness.missing_direct_support == ("answer",)


def test_indirect_evidence_is_audit_context_not_direct_support() -> None:
    bundle = CandidateEvidenceBundle(
        candidate_id="candidate-1",
        claims=(claim("answer"),),
        evidence=(evidence("secondary", status="indirect"),),
        links=(link("answer", "secondary", EvidenceSupport.INDIRECT),),
    )
    assert assess_evidence_readiness(bundle).ready is False


def test_conflicting_evidence_blocks_ready_candidate() -> None:
    bundle = CandidateEvidenceBundle(
        candidate_id="candidate-1",
        claims=(claim("answer"),),
        evidence=(evidence("primary"), evidence("conflict", status="independently_verified")),
        links=(
            link("answer", "primary"),
            link("answer", "conflict", EvidenceSupport.CONTRADICTS),
        ),
    )
    readiness = assess_evidence_readiness(bundle)
    assert readiness.ready is False
    assert readiness.conflicting_claims == ("answer",)


def test_deterministic_candidate_can_be_ready_without_external_evidence() -> None:
    bundle = CandidateEvidenceBundle(
        candidate_id="candidate-1",
        deterministic=True,
        claims=(claim("computed"),),
    )
    assert assess_evidence_readiness(bundle).ready is True


def test_graph_rejects_unknown_links_duplicate_nodes_and_unlocated_direct_edges() -> None:
    with pytest.raises(ValidationError, match="unknown node"):
        CandidateEvidenceBundle(
            candidate_id="candidate-1",
            claims=(claim("answer"),),
            evidence=(evidence("paper"),),
            links=(link("missing", "paper"),),
        )
    with pytest.raises(ValidationError, match="unique IDs"):
        CandidateEvidenceBundle(
            candidate_id="candidate-1",
            claims=(claim("answer"), claim("answer")),
        )
    with pytest.raises(ValidationError, match="excerpt or locator"):
        ClaimEvidenceLink(
            claim_id="answer",
            evidence_id="paper",
            support=EvidenceSupport.DIRECT,
        )
