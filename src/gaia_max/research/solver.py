"""Bounded deep-research solver with deterministic evidence gating and replanning."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from gaia_max.domain import Question, SemanticAnswer, Verdict
from gaia_max.observability import NoOpTelemetry, Telemetry
from gaia_max.research.llamaindex_retrieval import PassageCatalog
from gaia_max.research.models import (
    ResearchDraft,
    ResearchObjective,
    ResearchRunRecord,
    ResearchStatus,
    RetrievedPassage,
)
from gaia_max.retrieval.evidence import (
    CandidateEvidenceBundle,
    ClaimEvidenceLink,
    EvidenceSupport,
    ResearchClaim,
)
from gaia_max.solver_routing import SolverRoute
from gaia_max.task_analysis import ReconciledTaskAnalysis
from gaia_max.verification import CandidateVerificationContext, verify_candidate


class DeepResearchBackend(Protocol):
    async def research(self, objective: ResearchObjective) -> ResearchDraft: ...


class DeepResearchError(RuntimeError):
    """Research could not cross its evidence gate within the allowed attempts."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"deep research failed: {code}")


class ResearchLedger:
    """Thread-safe in-memory audit ledger kept outside submission-bearing graph state."""

    def __init__(self) -> None:
        self._records: dict[str, ResearchRunRecord] = {}
        self._lock = threading.Lock()

    def put(self, record: ResearchRunRecord) -> None:
        with self._lock:
            previous = self._records.get(record.task_id)
            if previous is not None and previous.status is ResearchStatus.APPROVED:
                raise ValueError("approved research records are immutable")
            self._records[record.task_id] = record

    def get(self, task_id: str) -> ResearchRunRecord | None:
        with self._lock:
            return self._records.get(task_id)


class DeepResearchSolver:
    """Specialist fallback: smolagents proposes; code-owned evidence policy decides."""

    def __init__(
        self,
        *,
        backend: DeepResearchBackend,
        passage_catalog: PassageCatalog,
        ledger: ResearchLedger | None = None,
        max_replans: int = 2,
        require_independent_retrieval: bool = False,
        telemetry: Telemetry | None = None,
    ) -> None:
        if not 0 <= max_replans <= 19:
            raise ValueError("max_replans must be between zero and nineteen")
        self._backend = backend
        self._catalog = passage_catalog
        self.ledger = ledger or ResearchLedger()
        self._max_replans = max_replans
        self._require_independent_retrieval = require_independent_retrieval
        self._telemetry = telemetry or NoOpTelemetry()

    async def solve(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        route: str,
    ) -> SemanticAnswer:
        if route != SolverRoute.DEEP_RESEARCH_FALLBACK.value:
            raise DeepResearchError("route_scope_violation")

        unresolved: tuple[str, ...] = ()
        last_draft: ResearchDraft | None = None
        last_bundle: CandidateEvidenceBundle | None = None
        reason_codes: tuple[str, ...] = ("research_not_started",)
        attempts = self._max_replans + 1

        with self._telemetry.span(
            "research.solver.run",
            {
                "research.max_attempts": attempts,
                "research.independence_required": self._require_independent_retrieval,
            },
        ) as span:
            for attempt in range(1, attempts + 1):
                objective = ResearchObjective(
                    task_id=question.task_id,
                    question=question.question,
                    required_output=analysis.output_contract.model_dump_json(),
                    unresolved_claims=unresolved,
                    attempt=attempt,
                )
                draft = await self._backend.research(objective)
                last_draft = draft
                try:
                    bundle = self._build_bundle(question.task_id, draft)
                except KeyError:
                    reason_codes = ("citation_not_observed",)
                    unresolved = self._repair_targets(draft, reason_codes)
                    continue
                last_bundle = bundle
                if draft.unresolved_claim_ids:
                    reason_codes = ("agent_reported_unresolved_claims",)
                    unresolved = self._repair_targets(draft, draft.unresolved_claim_ids)
                    continue

                outcome = verify_candidate(
                    CandidateVerificationContext(
                        bundle=bundle,
                        require_independent_retrieval=self._require_independent_retrieval,
                    )
                )
                reason_codes = tuple(outcome.verification.reason_codes)
                if outcome.verification.verdict is Verdict.APPROVE:
                    record = ResearchRunRecord(
                        task_id=question.task_id,
                        status=ResearchStatus.APPROVED,
                        attempts=attempt,
                        draft=draft,
                        evidence_bundle=bundle,
                        completed_at=datetime.now(UTC),
                    )
                    self.ledger.put(record)
                    span.set_attribute("research.attempts_used", attempt)
                    span.set_attribute("research.approved", True)
                    return draft.semantic_answer
                unresolved = self._repair_targets(
                    draft,
                    (*outcome.verification.missing_evidence, *reason_codes),
                )

            self.ledger.put(
                ResearchRunRecord(
                    task_id=question.task_id,
                    status=ResearchStatus.EXHAUSTED,
                    attempts=attempts,
                    draft=last_draft,
                    evidence_bundle=last_bundle,
                    reason_codes=reason_codes or ("evidence_gate_not_approved",),
                    completed_at=datetime.now(UTC),
                )
            )
            span.set_attribute("research.attempts_used", attempts)
            span.set_attribute("research.approved", False)
        raise DeepResearchError("evidence_gate_exhausted")

    def _build_bundle(self, task_id: str, draft: ResearchDraft) -> CandidateEvidenceBundle:
        citation_ids = tuple(
            dict.fromkeys(
                citation
                for claim in draft.claims
                for citation in claim.citation_ids
            )
        )
        passages = self._catalog.resolve(citation_ids)
        by_id = {passage.passage_id: passage for passage in passages}
        identity = hashlib.sha256(
            f"{task_id}\x00{draft.model_dump_json()}".encode()
        ).hexdigest()
        return CandidateEvidenceBundle(
            candidate_id=identity,
            claims=tuple(
                ResearchClaim(
                    claim_id=claim.claim_id,
                    statement=claim.statement,
                    required=claim.required,
                )
                for claim in draft.claims
            ),
            evidence=tuple(passage.to_evidence() for passage in passages),
            links=tuple(
                self._link(claim.claim_id, by_id[citation_id])
                for claim in draft.claims
                for citation_id in claim.citation_ids
            ),
        )

    @staticmethod
    def _link(claim_id: str, passage: RetrievedPassage) -> ClaimEvidenceLink:
        return ClaimEvidenceLink(
            claim_id=claim_id,
            evidence_id=passage.passage_id,
            support=EvidenceSupport.DIRECT,
            locator=passage.locator,
            excerpt=passage.excerpt,
        )

    @staticmethod
    def _repair_targets(
        draft: ResearchDraft,
        reasons: Sequence[str],
    ) -> tuple[str, ...]:
        claims = {claim.claim_id: claim.statement for claim in draft.claims}
        targets = [claims[item] for item in reasons if item in claims]
        targets.extend(item for item in reasons if item not in claims)
        return tuple(dict.fromkeys(targets))[:50]


__all__ = [
    "DeepResearchBackend",
    "DeepResearchError",
    "DeepResearchSolver",
    "ResearchLedger",
]
