from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gaia_max.domain import AnswerType, Modality, OutputContract, Question, TaskClass
from gaia_max.output_contracts import ContractParseStatus
from gaia_max.research.llamaindex_retrieval import PassageCatalog
from gaia_max.research.models import (
    ResearchClaimDraft,
    ResearchDraft,
    ResearchObjective,
    ResearchStatus,
    RetrievedPassage,
)
from gaia_max.research.solver import DeepResearchError, DeepResearchSolver
from gaia_max.solver_routing import SolverRoute
from gaia_max.task_analysis import AnalysisAuthority, ReconciledTaskAnalysis


def analysis() -> ReconciledTaskAnalysis:
    return ReconciledTaskAnalysis(
        task_id="research-1",
        modality=Modality.WEB,
        task_class=TaskClass.DEEP_RESEARCH,
        requested_operation="identify the supported entity",
        output_contract=OutputContract(answer_type=AnswerType.STRING),
        field_sources={
            "modality": AnalysisAuthority.PROFILE,
            "task_class": AnalysisAuthority.PROFILE,
            "requested_operation": AnalysisAuthority.PROFILE,
            "output_contract": AnalysisAuthority.PROFILE,
        },
        contract_parse_status=ContractParseStatus.COMPLETE,
        profile_used=True,
        model_used=False,
        requires_review=False,
    )


def passage(passage_id: str = "a" * 64) -> RetrievedPassage:
    return RetrievedPassage(
        passage_id=passage_id,
        node_id="node-1",
        document_id="official-record",
        model_text_artifact_id="b" * 64,
        source_url="https://official.example/record",
        page_number=4,
        retrieved_at=datetime.now(UTC),
        is_primary_source=True,
        excerpt="The official record names Ada as the recipient.",
        locator="page 4",
        lexical_score=2.0,
        fused_score=0.01,
        rank=1,
    )


def draft(citation_id: str, *, unresolved: bool = False) -> ResearchDraft:
    return ResearchDraft(
        semantic_answer={"kind": "string", "value": "Ada"},
        claims=(
            ResearchClaimDraft(
                claim_id="claim_recipient",
                statement="Ada is the recipient.",
                citation_ids=(citation_id,),
            ),
        ),
        unresolved_claim_ids=("claim_recipient",) if unresolved else (),
    )


class SequenceBackend:
    def __init__(self, drafts: list[ResearchDraft]) -> None:
        self._drafts = iter(drafts)
        self.objectives: list[ResearchObjective] = []

    async def research(self, objective: ResearchObjective) -> ResearchDraft:
        self.objectives.append(objective)
        return next(self._drafts)


@pytest.mark.asyncio
async def test_replans_unknown_citation_then_approves_observed_primary_evidence() -> None:
    catalog = PassageCatalog()
    catalog.add([passage()])
    backend = SequenceBackend([draft("f" * 64), draft("a" * 64)])
    solver = DeepResearchSolver(
        backend=backend,
        passage_catalog=catalog,
        max_replans=1,
    )

    answer = await solver.solve(
        Question(task_id="research-1", question="Who is the recipient?"),
        analysis(),
        SolverRoute.DEEP_RESEARCH_FALLBACK.value,
    )

    assert answer.value == "Ada"
    assert backend.objectives[1].attempt == 2
    assert "citation_not_observed" in backend.objectives[1].unresolved_claims
    record = solver.ledger.get("research-1")
    assert record is not None and record.status is ResearchStatus.APPROVED
    assert record.evidence_bundle is not None


@pytest.mark.asyncio
async def test_exhausts_instead_of_accepting_unresolved_or_indirect_claims() -> None:
    catalog = PassageCatalog()
    indirect = passage().model_copy(update={"is_primary_source": False})
    catalog.add([indirect])
    backend = SequenceBackend([draft("a" * 64, unresolved=True)])
    solver = DeepResearchSolver(
        backend=backend,
        passage_catalog=catalog,
        max_replans=0,
    )

    with pytest.raises(DeepResearchError, match="evidence_gate_exhausted"):
        await solver.solve(
            Question(task_id="research-1", question="Who is the recipient?"),
            analysis(),
            SolverRoute.DEEP_RESEARCH_FALLBACK.value,
        )

    record = solver.ledger.get("research-1")
    assert record is not None and record.status is ResearchStatus.EXHAUSTED
