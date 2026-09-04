"""Tier-1 verification by replaying deterministic specialist authorities."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError

from gaia_max.domain import (
    Question,
    SemanticAnswer,
    TaskClass,
    Verdict,
    VerificationResult,
)
from gaia_max.solver_routing import SolverRoute, SpecialistSolver
from gaia_max.task_analysis import ReconciledTaskAnalysis

_SEMANTIC_ANSWER_ADAPTER = TypeAdapter(SemanticAnswer)


class DeterministicVerificationRegistryError(ValueError):
    """A deterministic policy definition is invalid or duplicated."""


@dataclass(frozen=True, slots=True)
class _PolicyDefinition:
    task_class: TaskClass
    route: SolverRoute
    authority_checks: tuple[str, ...]


_MILESTONE_B_POLICIES: dict[SolverRoute, _PolicyDefinition] = {
    SolverRoute.DETERMINISTIC_TEXT: _PolicyDefinition(
        task_class=TaskClass.DETERMINISTIC_TEXT,
        route=SolverRoute.DETERMINISTIC_TEXT,
        authority_checks=(
            "reverse_round_trip",
            "supported_transform_recomputed",
        ),
    ),
    SolverRoute.OPERATION_TABLE: _PolicyDefinition(
        task_class=TaskClass.OPERATION_TABLE,
        route=SolverRoute.OPERATION_TABLE,
        authority_checks=(
            "operation_table_closed",
            "ordered_pairs_exhaustive",
        ),
    ),
    SolverRoute.STRUCTURED_TABLE: _PolicyDefinition(
        task_class=TaskClass.STRUCTURED_TABLE,
        route=SolverRoute.STRUCTURED_TABLE,
        authority_checks=(
            "required_rows_complete",
            "extreme_and_tie_break_recomputed",
        ),
    ),
    SolverRoute.CODE: _PolicyDefinition(
        task_class=TaskClass.CODE,
        route=SolverRoute.CODE,
        authority_checks=(
            "python_ast_policy",
            "fresh_process_replay",
        ),
    ),
    SolverRoute.SPREADSHEET: _PolicyDefinition(
        task_class=TaskClass.SPREADSHEET,
        route=SolverRoute.SPREADSHEET,
        authority_checks=(
            "workbook_structure_audited",
            "openpyxl_calamine_exact_parity",
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class _DeterministicReplayPolicy:
    """Run one specialist's authority again and compare its typed answer."""

    definition: _PolicyDefinition
    solver: SpecialistSolver

    async def verify(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        answer: SemanticAnswer,
    ) -> VerificationResult:
        verifier_name = f"deterministic-replay:{self.definition.route.value}"
        try:
            raw_expected = await self.solver.solve(
                question,
                analysis,
                self.definition.route.value,
            )
            expected = _SEMANTIC_ANSWER_ADAPTER.validate_python(raw_expected)
        except (ValidationError, ValueError):
            return VerificationResult(
                verifier=verifier_name,
                tier=1,
                verdict=Verdict.UNCERTAIN,
                reason_codes=["deterministic_authority_failed"],
                critique="deterministic authority did not produce a trusted result",
                requested_next_action="repair deterministic input or authority",
            )

        passed_checks = list(self.definition.authority_checks)
        if expected != answer:
            return VerificationResult(
                verifier=verifier_name,
                tier=1,
                verdict=Verdict.REJECT,
                passed_checks=passed_checks,
                reason_codes=["candidate_authority_mismatch"],
                critique="candidate disagrees with deterministic authority",
                requested_next_action="replace candidate from deterministic authority",
            )
        return VerificationResult(
            verifier=verifier_name,
            tier=1,
            verdict=Verdict.APPROVE,
            passed_checks=[*passed_checks, "candidate_matches_authority"],
            critique="deterministic authority reproduced the candidate",
        )


class DeterministicVerificationRegistry:
    """TaskGraph verifier containing only explicitly registered B policies."""

    def __init__(self) -> None:
        self._policies: dict[TaskClass, _DeterministicReplayPolicy] = {}

    def register(self, route: SolverRoute | str, solver: SpecialistSolver) -> None:
        """Bind a real B specialist to its fixed deterministic authority policy."""

        parsed_route = SolverRoute(route)
        definition = _MILESTONE_B_POLICIES.get(parsed_route)
        if definition is None:
            raise DeterministicVerificationRegistryError(
                f"route {parsed_route.value!r} has no Milestone B deterministic policy"
            )
        if definition.task_class in self._policies:
            raise DeterministicVerificationRegistryError(
                f"route {parsed_route.value!r} already has a deterministic policy"
            )
        if not callable(getattr(solver, "solve", None)):
            raise DeterministicVerificationRegistryError(
                "deterministic policy requires a specialist solver"
            )
        self._policies[definition.task_class] = _DeterministicReplayPolicy(
            definition=definition,
            solver=solver,
        )

    async def verify(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        answer: SemanticAnswer,
    ) -> VerificationResult:
        """Dispatch by trusted task class; missing policy can never approve."""

        policy = self._policies.get(analysis.task_class)
        if policy is None:
            return VerificationResult(
                verifier="deterministic-policy-registry",
                tier=1,
                verdict=Verdict.UNCERTAIN,
                reason_codes=["deterministic_policy_unavailable"],
                critique="no deterministic authority is registered for this task class",
                requested_next_action="run the task's later verification ladder",
            )
        return await policy.verify(question, analysis, answer)


__all__ = [
    "DeterministicVerificationRegistry",
    "DeterministicVerificationRegistryError",
]
