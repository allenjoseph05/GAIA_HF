from __future__ import annotations

import pytest

from gaia_max.answer_validation import FinalAnswerValidator
from gaia_max.domain import (
    AnswerType,
    IntegerAnswer,
    Modality,
    OutputContract,
    Question,
    RiskLevel,
    SemanticAnswer,
    TaskClass,
    TaskProfile,
    TaskStatus,
    Verdict,
    VerificationResult,
)
from gaia_max.serialization import AnswerSerializer
from gaia_max.snapshot import question_sha256
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.task_analysis import ReconciledTaskAnalysis
from gaia_max.task_graph import (
    TaskGraphErrorCode,
    TaskGraphInput,
    TaskGraphState,
    task_result_from_state,
)
from gaia_max.task_graph import (
    build_task_graph as compile_task_graph,
)


def build_task_graph(
    *,
    solver: StubSolver,
    verifier: StubVerifier,
    **kwargs: object,
) -> object:
    """Keep A15 tests concise while exercising the real A17 registry boundary."""

    registry = SolverRegistry()
    registry.register(SolverRoute.STRUCTURED_TABLE, solver)
    return compile_task_graph(
        solver_registry=registry,
        verifier=verifier,
        **kwargs,
    )


class StubSolver:
    def __init__(
        self,
        answer: object | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.answer = answer if answer is not None else IntegerAnswer(value=42)
        self.error = error
        self.calls = 0
        self.routes: list[str] = []

    async def solve(
        self,
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
        route: str,
    ) -> object:
        self.calls += 1
        self.routes.append(route)
        if self.error is not None:
            raise self.error
        return self.answer


class StubVerifier:
    def __init__(
        self,
        verdict: Verdict = Verdict.APPROVE,
        *,
        error: Exception | None = None,
    ) -> None:
        self.verdict = verdict
        self.error = error
        self.calls = 0

    async def verify(
        self,
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
        _answer: SemanticAnswer,
    ) -> object:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return VerificationResult(
            verifier="synthetic-verifier",
            tier=0,
            verdict=self.verdict,
            reason_codes=[] if self.verdict is Verdict.APPROVE else ["synthetic_reject"],
        )


class PoisonedSerializer(AnswerSerializer):
    def serialize(self, answer: SemanticAnswer, contract: OutputContract) -> str:
        super().serialize(answer, contract)
        return "Answer: 42"


def question() -> Question:
    return Question(task_id="task-graph-1", question="How many records are there?")


def profile(runtime_question: Question | None = None) -> TaskProfile:
    resolved = runtime_question or question()
    return TaskProfile(
        task_id=resolved.task_id,
        question=resolved.question,
        question_sha256=question_sha256(resolved),
        file_name=resolved.file_name,
        modality=Modality.WEB,
        task_class=TaskClass.STRUCTURED_TABLE,
        route="structured_table",
        risk_level=RiskLevel.LOW,
        requested_operation="count records deterministically",
        required_sources=["synthetic_table"],
        output_contract=OutputContract(answer_type=AnswerType.INTEGER),
        verification_policy="synthetic_deterministic_replay",
    )


def initial_state(
    runtime_question: Question | None = None,
    runtime_profile: TaskProfile | None = None,
) -> TaskGraphState:
    resolved_question = runtime_question or question()
    resolved_profile = runtime_profile or profile(resolved_question)
    return TaskGraphInput(
        question=resolved_question,
        profile=resolved_profile,
    ).initial_state()


@pytest.mark.asyncio
async def test_successful_graph_runs_every_node_in_order() -> None:
    solver = StubSolver()
    verifier = StubVerifier()
    graph = build_task_graph(solver=solver, verifier=verifier)

    result = await graph.ainvoke(initial_state())

    assert result["status"] is TaskStatus.READY
    assert result["node_history"] == [
        "prepare",
        "analyze",
        "route",
        "solve",
        "verify",
        "serialize",
        "validate",
    ]
    assert result["route"] == "structured_table"
    assert result["route_decision"] is not None
    assert result["route_decision"].route is SolverRoute.STRUCTURED_TABLE
    assert result["serialized_answer"] == "42"
    assert result["output_validation"] is not None
    assert result["output_validation"].valid is True
    assert result["errors"] == []
    assert solver.routes == ["structured_table"]
    assert verifier.calls == 1


@pytest.mark.asyncio
async def test_graph_accepts_json_like_semantic_answer_from_solver_boundary() -> None:
    graph = build_task_graph(
        solver=StubSolver({"kind": "integer", "value": 7}),
        verifier=StubVerifier(),
    )

    result = await graph.ainvoke(initial_state())

    assert result["status"] is TaskStatus.READY
    assert result["semantic_answer"] == IntegerAnswer(value=7)
    assert result["serialized_answer"] == "7"


@pytest.mark.asyncio
async def test_changed_question_blocks_in_analysis_before_solver() -> None:
    original = question()
    changed = Question(task_id=original.task_id, question="How many changed rows are there?")
    solver = StubSolver()
    graph = build_task_graph(solver=solver, verifier=StubVerifier())

    result = await graph.ainvoke(initial_state(changed, profile(original)))

    assert result["status"] is TaskStatus.BLOCKED
    assert result["node_history"] == ["prepare", "analyze", "block"]
    assert result["errors"][0].code is TaskGraphErrorCode.ANALYSIS_FAILED
    assert solver.calls == 0


@pytest.mark.asyncio
async def test_solver_exception_is_sanitized_and_routes_to_block() -> None:
    solver = StubSolver(error=RuntimeError("secret solver candidate"))
    verifier = StubVerifier()
    graph = build_task_graph(solver=solver, verifier=verifier)

    result = await graph.ainvoke(initial_state())

    assert result["status"] is TaskStatus.BLOCKED
    assert result["node_history"][-2:] == ["solve", "block"]
    assert result["errors"][0].code is TaskGraphErrorCode.SOLVER_FAILED
    assert "secret solver candidate" not in str(result["errors"])
    assert verifier.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_answer", ["42", {"kind": "integer", "value": True}])
async def test_invalid_solver_schema_never_reaches_verifier(invalid_answer: object) -> None:
    verifier = StubVerifier()
    graph = build_task_graph(solver=StubSolver(invalid_answer), verifier=verifier)

    result = await graph.ainvoke(initial_state())

    assert result["status"] is TaskStatus.BLOCKED
    assert result["errors"][0].code is TaskGraphErrorCode.SOLVER_OUTPUT_INVALID
    assert verifier.calls == 0


@pytest.mark.asyncio
async def test_rejected_verification_blocks_before_serialization() -> None:
    graph = build_task_graph(
        solver=StubSolver(),
        verifier=StubVerifier(Verdict.REJECT),
    )

    result = await graph.ainvoke(initial_state())

    assert result["status"] is TaskStatus.BLOCKED
    assert result["serialized_answer"] is None
    assert result["node_history"][-2:] == ["verify", "block"]
    assert result["errors"][0].code is TaskGraphErrorCode.VERIFICATION_NOT_APPROVED
    assert result["errors"][0].detail_codes == ("synthetic_reject",)


@pytest.mark.asyncio
async def test_verifier_exception_is_sanitized_and_blocks() -> None:
    graph = build_task_graph(
        solver=StubSolver(),
        verifier=StubVerifier(error=RuntimeError("secret verifier payload")),
    )

    result = await graph.ainvoke(initial_state())

    assert result["status"] is TaskStatus.BLOCKED
    assert result["errors"][0].code is TaskGraphErrorCode.VERIFIER_FAILED
    assert "secret verifier payload" not in str(result["errors"])
    assert result["serialized_answer"] is None


@pytest.mark.asyncio
async def test_incompatible_semantic_type_blocks_during_serialization() -> None:
    graph = build_task_graph(
        solver=StubSolver({"kind": "string", "value": "forty-two"}),
        verifier=StubVerifier(),
    )

    result = await graph.ainvoke(initial_state())

    assert result["status"] is TaskStatus.BLOCKED
    assert result["node_history"][-2:] == ["serialize", "block"]
    assert result["errors"][0].code is TaskGraphErrorCode.SERIALIZATION_FAILED
    assert result["serialized_answer"] is None


@pytest.mark.asyncio
async def test_independent_validation_failure_routes_to_block_with_reason_codes() -> None:
    graph = build_task_graph(
        solver=StubSolver(),
        verifier=StubVerifier(),
        serializer=PoisonedSerializer(),
        answer_validator=FinalAnswerValidator(),
    )

    result = await graph.ainvoke(initial_state())

    assert result["status"] is TaskStatus.BLOCKED
    assert result["serialized_answer"] == "Answer: 42"
    assert result["output_validation"] is not None
    assert result["output_validation"].valid is False
    assert result["errors"][0].code is TaskGraphErrorCode.OUTPUT_INVALID
    assert "forbidden_prefix" in result["errors"][0].detail_codes
    assert result["node_history"][-2:] == ["validate", "block"]


def test_graph_renders_all_nodes_and_conditional_paths_as_mermaid() -> None:
    graph = build_task_graph(solver=StubSolver(), verifier=StubVerifier())

    mermaid = graph.get_graph().draw_mermaid()

    for node in (
        "prepare",
        "analyze",
        "route",
        "solve",
        "verify",
        "serialize",
        "validate",
        "block",
    ):
        assert node in mermaid
    assert "__start__" in mermaid
    assert "__end__" in mermaid
    assert "continue" in mermaid
    assert "ready" in mermaid
    assert "block --> __end__" in mermaid


@pytest.mark.asyncio
async def test_terminal_state_converts_to_existing_task_result_model() -> None:
    graph = build_task_graph(solver=StubSolver(), verifier=StubVerifier())
    state = await graph.ainvoke(initial_state())

    result = task_result_from_state(state)

    assert result.task_id == question().task_id
    assert result.status is TaskStatus.READY
    assert result.semantic_answer == IntegerAnswer(value=42)
    assert result.serialized_answer == "42"


@pytest.mark.asyncio
async def test_blocked_state_converts_to_reason_coded_task_result() -> None:
    graph = build_task_graph(
        solver=StubSolver(error=RuntimeError("do not expose")),
        verifier=StubVerifier(),
    )
    state = await graph.ainvoke(initial_state())

    result = task_result_from_state(state)

    assert result.status is TaskStatus.BLOCKED
    assert result.errors == ["solve:solver_failed"]


def test_task_graph_input_rejects_unexpected_state_fields() -> None:
    with pytest.raises(ValueError):
        TaskGraphInput.model_validate(
            {
                "question": question(),
                "profile": profile(),
                "gold_answer": "must-not-enter",
            }
        )


def test_graph_state_has_no_gold_answer_field() -> None:
    assert "gold_answer" not in TaskGraphState.__annotations__
    assert "submitted_answer" not in TaskGraphState.__annotations__


@pytest.mark.asyncio
async def test_graph_blocks_when_selected_specialist_is_not_registered() -> None:
    graph = compile_task_graph(
        solver_registry=SolverRegistry(),
        verifier=StubVerifier(),
    )

    result = await graph.ainvoke(initial_state())

    assert result["status"] is TaskStatus.BLOCKED
    assert result["node_history"] == ["prepare", "analyze", "route", "block"]
    assert result["errors"][0].code is TaskGraphErrorCode.ROUTE_UNAVAILABLE
    assert result["errors"][0].detail_codes == (
        "solver_not_registered",
        "structured_table",
    )
