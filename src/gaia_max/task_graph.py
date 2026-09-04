"""Minimal typed LangGraph control flow for one GAIA task."""

from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Annotated, Literal, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from gaia_max.answer_validation import (
    FinalAnswerValidationResult,
    FinalAnswerValidator,
)
from gaia_max.domain import (
    Question,
    SemanticAnswer,
    TaskProfile,
    TaskResult,
    TaskStatus,
    Verdict,
    VerificationResult,
)
from gaia_max.observability import NoOpTelemetry, Telemetry, TraceSpan
from gaia_max.serialization import AnswerSerializationError, AnswerSerializer
from gaia_max.solver_routing import (
    DeterministicRoutePolicy,
    RouteDecision,
    RouteFailureCode,
    SolverRegistry,
    SolverRoutingError,
    SpecialistSolver,
)
from gaia_max.task_analysis import ReconciledTaskAnalysis, TaskAnalysisError, TaskAnalyzer

_SEMANTIC_ANSWER_ADAPTER = TypeAdapter(SemanticAnswer)


class TaskGraphErrorCode(StrEnum):
    """Stable task-control failures used by later retry and repair policies."""

    ANALYSIS_FAILED = "analysis_failed"
    ROUTE_UNAVAILABLE = "route_unavailable"
    SOLVER_FAILED = "solver_failed"
    SOLVER_OUTPUT_INVALID = "solver_output_invalid"
    VERIFIER_FAILED = "verifier_failed"
    VERIFICATION_NOT_APPROVED = "verification_not_approved"
    SERIALIZATION_FAILED = "serialization_failed"
    OUTPUT_INVALID = "output_invalid"
    INVALID_TERMINAL_STATE = "invalid_terminal_state"


class TaskGraphError(BaseModel):
    """Answer-free error information safe for graph control and reporting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: TaskGraphErrorCode
    node: str
    message: str
    detail_codes: tuple[str, ...] = ()


class TaskNodeExecutionError(RuntimeError):
    """Sanitized transient node failure that a checkpointer may resume."""

    def __init__(self, code: TaskGraphErrorCode, node: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.node = node


class TaskGraphInput(BaseModel):
    """Validated input boundary for a profiled or newly analyzed task run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: Question
    profile: TaskProfile | None = None

    def initial_state(self) -> TaskGraphState:
        return {
            "question": self.question,
            "profile": self.profile,
            "status": TaskStatus.NEW,
            "analysis": None,
            "route": None,
            "route_decision": None,
            "semantic_answer": None,
            "verification": None,
            "serialized_answer": None,
            "output_validation": None,
            "errors": [],
            "node_history": [],
        }


class TaskGraphState(TypedDict):
    """Compact state carried between the first TaskGraph nodes."""

    question: Question
    profile: TaskProfile | None
    status: TaskStatus
    analysis: ReconciledTaskAnalysis | None
    route: str | None
    route_decision: RouteDecision | None
    semantic_answer: SemanticAnswer | None
    verification: VerificationResult | None
    serialized_answer: str | None
    output_validation: FinalAnswerValidationResult | None
    errors: Annotated[list[TaskGraphError], operator.add]
    node_history: Annotated[list[str], operator.add]


TaskSolver = SpecialistSolver


class TaskVerifier(Protocol):
    """Verification port; real policies are added in the specialist milestones."""

    async def verify(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        answer: SemanticAnswer,
    ) -> object: ...


class MinimalTaskGraph:
    """Own the nodes and compile their explicit deterministic control flow."""

    def __init__(
        self,
        *,
        solver_registry: SolverRegistry,
        verifier: TaskVerifier,
        route_policy: DeterministicRoutePolicy | None = None,
        analyzer: TaskAnalyzer | None = None,
        serializer: AnswerSerializer | None = None,
        answer_validator: FinalAnswerValidator | None = None,
        propagate_transient_exceptions: bool = False,
        telemetry: Telemetry | None = None,
    ) -> None:
        self._solver_registry = solver_registry
        self._verifier = verifier
        self._route_policy = route_policy or DeterministicRoutePolicy(
            solver_registry.catalog
        )
        self._analyzer = analyzer or TaskAnalyzer()
        self._serializer = serializer or AnswerSerializer()
        self._answer_validator = answer_validator or FinalAnswerValidator()
        self._propagate_transient_exceptions = propagate_transient_exceptions
        self._telemetry = telemetry or NoOpTelemetry()

    def compile(
        self,
        *,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> CompiledStateGraph[TaskGraphState, None, TaskGraphState, TaskGraphState]:
        """Compile with optional persistence; A15 stays usable without a saver."""

        builder = StateGraph(TaskGraphState)
        builder.add_node("prepare", self._traced_prepare)
        builder.add_node("analyze", self._traced_analyze)
        builder.add_node("route", self._traced_route)
        builder.add_node("solve", self._traced_solve)
        builder.add_node("verify", self._traced_verify)
        builder.add_node("serialize", self._traced_serialize)
        builder.add_node("validate", self._traced_validate)
        builder.add_node("block", self._traced_block)

        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "analyze")
        builder.add_conditional_edges(
            "analyze",
            self._continue_or_block,
            {"continue": "route", "block": "block"},
        )
        builder.add_conditional_edges(
            "route",
            self._continue_or_block,
            {"continue": "solve", "block": "block"},
        )
        builder.add_conditional_edges(
            "solve",
            self._continue_or_block,
            {"continue": "verify", "block": "block"},
        )
        builder.add_conditional_edges(
            "verify",
            self._continue_or_block,
            {"continue": "serialize", "block": "block"},
        )
        builder.add_conditional_edges(
            "serialize",
            self._continue_or_block,
            {"continue": "validate", "block": "block"},
        )
        builder.add_conditional_edges(
            "validate",
            self._ready_or_block,
            {"ready": END, "block": "block"},
        )
        builder.add_edge("block", END)
        return builder.compile(
            checkpointer=checkpointer,
            name="gaia_minimal_task_graph",
        )

    def _traced_prepare(self, state: TaskGraphState) -> dict[str, object]:
        return self._run_traced_sync("prepare", self._prepare, state)

    async def _traced_analyze(self, state: TaskGraphState) -> dict[str, object]:
        return await self._run_traced_async("analyze", self._analyze, state)

    def _traced_route(self, state: TaskGraphState) -> dict[str, object]:
        return self._run_traced_sync("route", self._route, state)

    async def _traced_solve(self, state: TaskGraphState) -> dict[str, object]:
        return await self._run_traced_async("solve", self._solve, state)

    async def _traced_verify(self, state: TaskGraphState) -> dict[str, object]:
        return await self._run_traced_async("verify", self._verify, state)

    def _traced_serialize(self, state: TaskGraphState) -> dict[str, object]:
        return self._run_traced_sync("serialize", self._serialize, state)

    def _traced_validate(self, state: TaskGraphState) -> dict[str, object]:
        return self._run_traced_sync("validate", self._validate, state)

    def _traced_block(self, state: TaskGraphState) -> dict[str, object]:
        return self._run_traced_sync("block", self._block, state)

    def _run_traced_sync(
        self,
        node_name: str,
        node: Callable[[TaskGraphState], dict[str, object]],
        state: TaskGraphState,
    ) -> dict[str, object]:
        with self._telemetry.span(
            f"task_graph.{node_name}", self._trace_attributes(state)
        ) as span:
            result = node(state)
            self._set_trace_result(span, result)
            return result

    async def _run_traced_async(
        self,
        node_name: str,
        node: Callable[[TaskGraphState], Awaitable[dict[str, object]]],
        state: TaskGraphState,
    ) -> dict[str, object]:
        with self._telemetry.span(
            f"task_graph.{node_name}", self._trace_attributes(state)
        ) as span:
            result = await node(state)
            self._set_trace_result(span, result)
            return result

    @staticmethod
    def _set_trace_result(span: TraceSpan, result: dict[str, object]) -> None:
        status = result.get("status")
        if isinstance(status, TaskStatus):
            span.set_attribute("graph.result_status", status.value)

    @staticmethod
    def _trace_attributes(state: TaskGraphState) -> dict[str, str]:
        return {
            "graph.state_before": state["status"].value,
            "graph.route": state["route"] or "unselected",
        }

    @staticmethod
    def _prepare(state: TaskGraphState) -> dict[str, object]:
        del state
        return {
            "status": TaskStatus.ANALYZING,
            "node_history": ["prepare"],
        }

    async def _analyze(self, state: TaskGraphState) -> dict[str, object]:
        try:
            analysis = await self._analyzer.analyze(
                state["question"],
                profile=state["profile"],
            )
        except (TaskAnalysisError, ValidationError, ValueError):
            return self._failure(
                TaskGraphErrorCode.ANALYSIS_FAILED,
                "analyze",
                "task analysis did not produce trusted routing metadata",
            )
        except Exception:
            if self._propagate_transient_exceptions:
                raise TaskNodeExecutionError(
                    TaskGraphErrorCode.ANALYSIS_FAILED,
                    "analyze",
                    "task analyzer failed unexpectedly",
                ) from None
            return self._failure(
                TaskGraphErrorCode.ANALYSIS_FAILED,
                "analyze",
                "task analyzer failed unexpectedly",
            )
        return {"analysis": analysis, "node_history": ["analyze"]}

    def _route(self, state: TaskGraphState) -> dict[str, object]:
        analysis = state["analysis"]
        if analysis is None:
            return self._failure(
                TaskGraphErrorCode.ROUTE_UNAVAILABLE,
                "route",
                "trusted task analysis is unavailable for routing",
            )
        try:
            decision = self._route_policy.decide(
                state["question"],
                analysis,
                profile=state["profile"],
            )
        except SolverRoutingError as exc:
            return self._failure(
                TaskGraphErrorCode.ROUTE_UNAVAILABLE,
                "route",
                "deterministic routing could not choose one safe specialist",
                detail_codes=(exc.code.value, *exc.detail_codes),
            )
        if not self._solver_registry.has(decision.route):
            return self._failure(
                TaskGraphErrorCode.ROUTE_UNAVAILABLE,
                "route",
                "selected specialist solver is not registered",
                detail_codes=(
                    RouteFailureCode.SOLVER_NOT_REGISTERED.value,
                    decision.route.value,
                ),
            )
        return {
            "route": decision.route.value,
            "route_decision": decision,
            "status": TaskStatus.SOLVING,
            "node_history": ["route"],
        }

    async def _solve(self, state: TaskGraphState) -> dict[str, object]:
        analysis = state["analysis"]
        decision = state["route_decision"]
        if analysis is None or decision is None:
            return self._failure(
                TaskGraphErrorCode.ROUTE_UNAVAILABLE,
                "solve",
                "solver prerequisites are missing",
            )
        try:
            raw_answer = await self._solver_registry.solve(
                decision.route,
                state["question"],
                analysis,
            )
        except Exception:
            if self._propagate_transient_exceptions:
                raise TaskNodeExecutionError(
                    TaskGraphErrorCode.SOLVER_FAILED,
                    "solve",
                    "selected solver failed transiently",
                ) from None
            return self._failure(
                TaskGraphErrorCode.SOLVER_FAILED,
                "solve",
                "selected solver failed without producing a trusted candidate",
            )
        try:
            answer = _SEMANTIC_ANSWER_ADAPTER.validate_python(raw_answer)
        except ValidationError:
            return self._failure(
                TaskGraphErrorCode.SOLVER_OUTPUT_INVALID,
                "solve",
                "selected solver returned an invalid semantic-answer schema",
            )
        return {
            "semantic_answer": answer,
            "status": TaskStatus.VERIFYING,
            "node_history": ["solve"],
        }

    async def _verify(self, state: TaskGraphState) -> dict[str, object]:
        analysis = state["analysis"]
        answer = state["semantic_answer"]
        if analysis is None or answer is None:
            return self._failure(
                TaskGraphErrorCode.VERIFIER_FAILED,
                "verify",
                "verifier prerequisites are missing",
            )
        try:
            raw_result = await self._verifier.verify(state["question"], analysis, answer)
            result = VerificationResult.model_validate(raw_result)
        except (ValidationError, ValueError):
            return self._failure(
                TaskGraphErrorCode.VERIFIER_FAILED,
                "verify",
                "verifier returned an invalid result schema",
            )
        except Exception:
            if self._propagate_transient_exceptions:
                raise TaskNodeExecutionError(
                    TaskGraphErrorCode.VERIFIER_FAILED,
                    "verify",
                    "verifier failed transiently",
                ) from None
            return self._failure(
                TaskGraphErrorCode.VERIFIER_FAILED,
                "verify",
                "verifier failed unexpectedly",
            )
        if result.verdict is not Verdict.APPROVE:
            return {
                **self._failure(
                    TaskGraphErrorCode.VERIFICATION_NOT_APPROVED,
                    "verify",
                    "candidate did not receive verification approval",
                    detail_codes=tuple(result.reason_codes),
                ),
                "verification": result,
            }
        return {
            "verification": result,
            "status": TaskStatus.ACCEPTED,
            "node_history": ["verify"],
        }

    def _serialize(self, state: TaskGraphState) -> dict[str, object]:
        analysis = state["analysis"]
        answer = state["semantic_answer"]
        verification = state["verification"]
        if (
            analysis is None
            or answer is None
            or verification is None
            or verification.verdict is not Verdict.APPROVE
        ):
            return self._failure(
                TaskGraphErrorCode.SERIALIZATION_FAILED,
                "serialize",
                "serialization requires an approved typed candidate",
            )
        try:
            serialized = self._serializer.serialize(answer, analysis.output_contract)
        except (AnswerSerializationError, ValidationError, ValueError):
            return self._failure(
                TaskGraphErrorCode.SERIALIZATION_FAILED,
                "serialize",
                "candidate cannot satisfy the reconciled output contract",
            )
        return {
            "serialized_answer": serialized,
            "status": TaskStatus.SERIALIZED,
            "node_history": ["serialize"],
        }

    def _validate(self, state: TaskGraphState) -> dict[str, object]:
        analysis = state["analysis"]
        serialized = state["serialized_answer"]
        if analysis is None or serialized is None:
            return self._failure(
                TaskGraphErrorCode.OUTPUT_INVALID,
                "validate",
                "final-answer validation prerequisites are missing",
            )
        result = self._answer_validator.validate(serialized, analysis.output_contract)
        if not result.valid:
            return {
                **self._failure(
                    TaskGraphErrorCode.OUTPUT_INVALID,
                    "validate",
                    "serialized answer failed final structural validation",
                    detail_codes=tuple(issue.code.value for issue in result.issues),
                ),
                "output_validation": result,
            }
        return {
            "output_validation": result,
            "status": TaskStatus.READY,
            "node_history": ["validate"],
        }

    @staticmethod
    def _block(state: TaskGraphState) -> dict[str, object]:
        del state
        return {"status": TaskStatus.BLOCKED, "node_history": ["block"]}

    @staticmethod
    def _continue_or_block(state: TaskGraphState) -> Literal["continue", "block"]:
        return "block" if state["status"] is TaskStatus.BLOCKED else "continue"

    @staticmethod
    def _ready_or_block(state: TaskGraphState) -> Literal["ready", "block"]:
        return "ready" if state["status"] is TaskStatus.READY else "block"

    @staticmethod
    def _failure(
        code: TaskGraphErrorCode,
        node: str,
        message: str,
        *,
        detail_codes: tuple[str, ...] = (),
    ) -> dict[str, object]:
        return {
            "status": TaskStatus.BLOCKED,
            "errors": [
                TaskGraphError(
                    code=code,
                    node=node,
                    message=message,
                    detail_codes=detail_codes,
                )
            ],
            "node_history": [node],
        }


def build_task_graph(
    *,
    solver_registry: SolverRegistry,
    verifier: TaskVerifier,
    route_policy: DeterministicRoutePolicy | None = None,
    analyzer: TaskAnalyzer | None = None,
    serializer: AnswerSerializer | None = None,
    answer_validator: FinalAnswerValidator | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    propagate_transient_exceptions: bool = False,
    telemetry: Telemetry | None = None,
) -> CompiledStateGraph[TaskGraphState, None, TaskGraphState, TaskGraphState]:
    """Build the typed task graph with explicit A17 capability boundaries."""

    return MinimalTaskGraph(
        solver_registry=solver_registry,
        verifier=verifier,
        route_policy=route_policy,
        analyzer=analyzer,
        serializer=serializer,
        answer_validator=answer_validator,
        propagate_transient_exceptions=propagate_transient_exceptions,
        telemetry=telemetry,
    ).compile(checkpointer=checkpointer)


def task_result_from_state(state: TaskGraphState) -> TaskResult:
    """Convert only a terminal graph state into the existing domain result model."""

    if state["status"] is TaskStatus.READY:
        return TaskResult(
            task_id=state["question"].task_id,
            status=TaskStatus.READY,
            semantic_answer=state["semantic_answer"],
            serialized_answer=state["serialized_answer"],
        )
    if state["status"] is TaskStatus.BLOCKED:
        errors = [f"{error.node}:{error.code.value}" for error in state["errors"]]
        if not errors:
            errors = [f"graph:{TaskGraphErrorCode.INVALID_TERMINAL_STATE.value}"]
        return TaskResult(
            task_id=state["question"].task_id,
            status=TaskStatus.BLOCKED,
            errors=errors,
        )
    raise ValueError("task graph state is not terminal")
