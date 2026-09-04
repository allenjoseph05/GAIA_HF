"""Typed LangGraph coordinator for snapshot-safe concurrent task dispatch."""

from __future__ import annotations

import asyncio
import operator
from enum import StrEnum
from typing import Annotated, Literal, Protocol, TypedDict, TypeVar

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gaia_max.checkpoint_serde import build_safe_checkpoint_serializer
from gaia_max.checkpointing import (
    TASK_CHECKPOINT_TYPES,
    TaskCheckpointInspection,
    TaskThread,
)
from gaia_max.domain import (
    Question,
    QuestionSnapshot,
    RiskAssessment,
    TaskResult,
    TaskStatus,
)
from gaia_max.profiles import (
    ProfileReconciliation,
    TaskProfileRecord,
    TaskProfileRegistry,
    reconcile_profile_registry,
)
from gaia_max.snapshot import build_question_snapshot, question_sha256
from gaia_max.task_graph import TaskGraphInput, TaskGraphState, task_result_from_state

_RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
ValueT = TypeVar("ValueT", bound=BaseModel)


class RunStatus(StrEnum):
    """Lifecycle of one run-level coordination pass."""

    NEW = "new"
    VALIDATING = "validating"
    PLANNING = "planning"
    DISPATCHING = "dispatching"
    COLLECTING = "collecting"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class TaskDispatchAction(StrEnum):
    """Idempotent action selected from one task checkpoint lifecycle."""

    START = "start"
    RESUME = "resume"
    REUSE = "reuse"


class RunErrorCode(StrEnum):
    """Stable answer-free failures used by run-level repair and reporting."""

    SNAPSHOT_COUNT_MISMATCH = "snapshot_count_mismatch"
    SNAPSHOT_INVENTORY_MISMATCH = "snapshot_inventory_mismatch"
    SNAPSHOT_HASH_MISMATCH = "snapshot_hash_mismatch"
    PROFILE_REGISTRY_MISMATCH = "profile_registry_mismatch"
    UNKNOWN_TASK_SELECTION = "unknown_task_selection"
    DISPATCH_PLANNING_FAILED = "dispatch_planning_failed"
    TASK_EXECUTION_INTERRUPTED = "task_execution_interrupted"
    TASK_RESULT_MISMATCH = "task_result_mismatch"
    INCOMPLETE_COLLECTION = "incomplete_collection"
    INVALID_TERMINAL_STATE = "invalid_terminal_state"


class RunError(BaseModel):
    """Sanitized run-control error with no candidate-answer content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: RunErrorCode
    node: str = Field(min_length=1)
    task_id: str | None = None
    detail_codes: tuple[str, ...] = ()


class RunTaskSelection(BaseModel):
    """Optional explicit subset; an empty tuple means the full snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_ids: tuple[str, ...] = ()

    @field_validator("task_ids")
    @classmethod
    def validate_task_ids(cls, task_ids: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("selected task IDs must be unique")
        if any(not task_id.strip() for task_id in task_ids):
            raise ValueError("selected task IDs cannot be empty")
        return task_ids


class RunGraphInput(BaseModel):
    """Validated public inputs needed to coordinate one task run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1, max_length=64, pattern=_RUN_ID_PATTERN)
    snapshot: QuestionSnapshot
    questions: tuple[Question, ...] = Field(min_length=1)
    profile_registry: TaskProfileRegistry
    selection: RunTaskSelection = Field(default_factory=RunTaskSelection)
    expected_question_count: int = Field(default=20, ge=1, le=1000)

    def initial_state(self) -> GaiaRunState:
        return {
            "run_id": self.run_id,
            "snapshot": self.snapshot,
            "questions": self.questions,
            "profile_registry": self.profile_registry,
            "selection": self.selection,
            "expected_question_count": self.expected_question_count,
            "status": RunStatus.NEW,
            "profile_reconciliation": None,
            "selected_task_ids": (),
            "dispatch_plans": (),
            "task_results": {},
            "dispatch_records": {},
            "run_errors": [],
            "node_history": [],
        }


class TaskDispatchPlan(BaseModel):
    """One immutable task action planned from its checkpoint state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    action: TaskDispatchAction
    task_input: TaskGraphInput
    thread: TaskThread

    @model_validator(mode="after")
    def validate_identity(self) -> TaskDispatchPlan:
        if self.task_id != self.task_input.question.task_id:
            raise ValueError("dispatch plan task ID must match task input")
        if self.task_id != self.thread.task_id:
            raise ValueError("dispatch plan task ID must match checkpoint thread")
        return self


class TaskDispatchRecord(BaseModel):
    """Answer-free record of what RunGraph did for one task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    action: TaskDispatchAction
    terminal_status: TaskStatus
    checkpoint_resumable: bool = False
    error_code: RunErrorCode | None = None


class GaiaRunResult(BaseModel):
    """Terminal run result used by reporting and the later A19 preflight."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    status: RunStatus
    selected_task_ids: tuple[str, ...]
    task_results: dict[str, TaskResult]
    dispatch_records: dict[str, TaskDispatchRecord]
    errors: tuple[RunError, ...] = ()

    @model_validator(mode="after")
    def validate_terminal_inventory(self) -> GaiaRunResult:
        selected = set(self.selected_task_ids)
        if self.status is RunStatus.COMPLETE:
            if set(self.task_results) != selected or set(self.dispatch_records) != selected:
                raise ValueError("complete run requires one result and record per selected task")
        elif self.status is RunStatus.BLOCKED and not self.errors:
            raise ValueError("blocked run requires at least one run error")
        elif self.status not in {RunStatus.COMPLETE, RunStatus.BLOCKED}:
            raise ValueError("run result requires a terminal status")
        return self


def build_run_checkpoint_serializer() -> JsonPlusSerializer:
    """Build the strict serializer required by an optional RunGraph checkpointer."""

    return build_safe_checkpoint_serializer(
        (
            *TASK_CHECKPOINT_TYPES,
            QuestionSnapshot,
            RiskAssessment,
            TaskResult,
            TaskProfileRecord,
            TaskProfileRegistry,
            ProfileReconciliation,
            TaskThread,
            TaskGraphInput,
            RunStatus,
            TaskDispatchAction,
            RunErrorCode,
            RunError,
            RunTaskSelection,
            RunGraphInput,
            TaskDispatchPlan,
            TaskDispatchRecord,
            GaiaRunResult,
        )
    )


def merge_task_results(
    left: dict[str, TaskResult],
    right: dict[str, TaskResult],
) -> dict[str, TaskResult]:
    """Merge parallel task outputs while rejecting conflicting duplicate writes."""

    return _merge_unique_mapping(left, right, label="task result")


def merge_dispatch_records(
    left: dict[str, TaskDispatchRecord],
    right: dict[str, TaskDispatchRecord],
) -> dict[str, TaskDispatchRecord]:
    """Merge parallel answer-free dispatch records deterministically."""

    return _merge_unique_mapping(left, right, label="dispatch record")


def merge_run_errors(left: list[RunError], right: list[RunError]) -> list[RunError]:
    """Deduplicate and deterministically order errors from parallel task branches."""

    merged = list(left)
    for error in right:
        if error not in merged:
            merged.append(error)
    return sorted(
        merged,
        key=lambda error: (
            error.task_id or "",
            error.node,
            error.code.value,
            error.detail_codes,
        ),
    )


class GaiaRunState(TypedDict):
    """Compact shared state for the first executable RunGraph."""

    run_id: str
    snapshot: QuestionSnapshot
    questions: tuple[Question, ...]
    profile_registry: TaskProfileRegistry
    selection: RunTaskSelection
    expected_question_count: int
    status: RunStatus
    profile_reconciliation: ProfileReconciliation | None
    selected_task_ids: tuple[str, ...]
    dispatch_plans: tuple[TaskDispatchPlan, ...]
    task_results: Annotated[dict[str, TaskResult], merge_task_results]
    dispatch_records: Annotated[
        dict[str, TaskDispatchRecord],
        merge_dispatch_records,
    ]
    run_errors: Annotated[list[RunError], merge_run_errors]
    node_history: Annotated[list[str], operator.add]


class TaskDispatchBranch(TypedDict):
    """Narrow per-Send payload consumed by a parallel dispatch node."""

    dispatch_plan: TaskDispatchPlan


class CheckpointedTaskDispatcher(Protocol):
    """RunGraph boundary implemented by AsyncSQLiteTaskRuntime."""

    async def inspect(self, thread: TaskThread) -> TaskCheckpointInspection: ...

    async def start(
        self,
        task_input: TaskGraphInput,
        thread: TaskThread,
    ) -> TaskGraphState: ...

    async def resume(self, thread: TaskThread) -> TaskGraphState: ...

    async def terminal_result(self, thread: TaskThread) -> TaskResult: ...


class RunGraph:
    """Validate, plan, fan out TaskGraphs, and collect isolated terminal results."""

    def __init__(
        self,
        *,
        dispatcher: CheckpointedTaskDispatcher,
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency < 1 or max_concurrency > 64:
            raise ValueError("max_concurrency must be between 1 and 64")
        self._dispatcher = dispatcher
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.BoundedSemaphore(max_concurrency)

    def compile(
        self,
        *,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> CompiledStateGraph[GaiaRunState, None, GaiaRunState, GaiaRunState]:
        """Compile the run-level graph with optional coordinator checkpointing."""

        builder = StateGraph(GaiaRunState)
        builder.add_node("validate_snapshot", self._validate_snapshot)
        builder.add_node("plan_dispatch", self._plan_dispatch)
        builder.add_node("dispatch_task", self._dispatch_task)
        builder.add_node("collect_results", self._collect_results)
        builder.add_node("block", self._block)

        builder.add_edge(START, "validate_snapshot")
        builder.add_conditional_edges(
            "validate_snapshot",
            self._continue_or_block,
            {"continue": "plan_dispatch", "block": "block"},
        )
        builder.add_conditional_edges(
            "plan_dispatch",
            self._fan_out,
            ["dispatch_task", "block"],
        )
        builder.add_edge("dispatch_task", "collect_results")
        builder.add_edge("collect_results", END)
        builder.add_edge("block", END)
        return builder.compile(checkpointer=checkpointer, name="gaia_run_graph")

    def _validate_snapshot(self, state: GaiaRunState) -> dict[str, object]:
        errors: list[RunError] = []
        snapshot = state["snapshot"]
        questions = state["questions"]
        if snapshot.count != state["expected_question_count"]:
            errors.append(
                RunError(
                    code=RunErrorCode.SNAPSHOT_COUNT_MISMATCH,
                    node="validate_snapshot",
                )
            )

        by_id = {question.task_id: question for question in questions}
        if len(by_id) != len(questions) or set(by_id) != set(snapshot.task_ids):
            errors.append(
                RunError(
                    code=RunErrorCode.SNAPSHOT_INVENTORY_MISMATCH,
                    node="validate_snapshot",
                )
            )
        else:
            current_hashes = {
                task_id: question_sha256(question) for task_id, question in by_id.items()
            }
            rebuilt = build_question_snapshot(
                questions,
                source_url=str(snapshot.source_url),
                task_profiles_version=snapshot.task_profiles_version,
                retrieved_at=snapshot.retrieved_at,
            )
            if (
                current_hashes != snapshot.task_hashes
                or rebuilt.snapshot_sha256 != snapshot.snapshot_sha256
            ):
                errors.append(
                    RunError(
                        code=RunErrorCode.SNAPSHOT_HASH_MISMATCH,
                        node="validate_snapshot",
                    )
                )

        audit = reconcile_profile_registry(state["profile_registry"], snapshot)
        if not audit.submission_safe:
            errors.append(
                RunError(
                    code=RunErrorCode.PROFILE_REGISTRY_MISMATCH,
                    node="validate_snapshot",
                    detail_codes=tuple(
                        [
                            *(("snapshot_hash",) if not audit.snapshot_hash_matches else ()),
                            *(("profile_version",) if not audit.profile_version_matches else ()),
                            *(("missing_profiles",) if audit.missing_profile_task_ids else ()),
                            *(("unknown_profiles",) if audit.unknown_profile_task_ids else ()),
                            *(("stale_profiles",) if audit.stale_profile_task_ids else ()),
                        ]
                    ),
                )
            )
        if errors:
            return {
                "status": RunStatus.BLOCKED,
                "profile_reconciliation": audit,
                "run_errors": errors,
                "node_history": ["validate_snapshot"],
            }
        return {
            "status": RunStatus.PLANNING,
            "profile_reconciliation": audit,
            "node_history": ["validate_snapshot"],
        }

    async def _plan_dispatch(self, state: GaiaRunState) -> dict[str, object]:
        snapshot_ids = tuple(state["snapshot"].task_ids)
        requested_ids = state["selection"].task_ids
        unknown = sorted(set(requested_ids) - set(snapshot_ids))
        if unknown:
            return self._failure(
                RunErrorCode.UNKNOWN_TASK_SELECTION,
                "plan_dispatch",
                detail_codes=tuple(unknown),
            )
        requested = set(requested_ids)
        selected = tuple(
            task_id for task_id in snapshot_ids if not requested_ids or task_id in requested
        )
        questions = {question.task_id: question for question in state["questions"]}
        profiles = state["profile_registry"].by_task_id()
        plans: list[TaskDispatchPlan] = []
        try:
            for task_id in selected:
                question = questions[task_id]
                task_input = TaskGraphInput(
                    question=question,
                    profile=profiles[task_id].materialize(question),
                )
                thread = TaskThread(run_id=state["run_id"], task_id=task_id)
                inspection = await self._dispatcher.inspect(thread)
                if inspection.thread_id != thread.thread_id:
                    raise ValueError("dispatcher inspection returned a mismatched thread")
                action = TaskDispatchAction.START
                if inspection.exists:
                    action = (
                        TaskDispatchAction.RESUME
                        if inspection.resumable
                        else TaskDispatchAction.REUSE
                    )
                plans.append(
                    TaskDispatchPlan(
                        task_id=task_id,
                        action=action,
                        task_input=task_input,
                        thread=thread,
                    )
                )
        except Exception:
            return self._failure(
                RunErrorCode.DISPATCH_PLANNING_FAILED,
                "plan_dispatch",
            )
        return {
            "status": RunStatus.DISPATCHING,
            "selected_task_ids": selected,
            "dispatch_plans": tuple(plans),
            "node_history": ["plan_dispatch"],
        }

    @staticmethod
    def _fan_out(state: GaiaRunState) -> list[Send] | Literal["block"]:
        if state["status"] is RunStatus.BLOCKED:
            return "block"
        return [
            Send("dispatch_task", {"dispatch_plan": plan})
            for plan in state["dispatch_plans"]
        ]

    async def _dispatch_task(self, state: TaskDispatchBranch) -> dict[str, object]:
        plan = state["dispatch_plan"]
        try:
            async with self._semaphore:
                if plan.action is TaskDispatchAction.START:
                    task_state = await self._dispatcher.start(plan.task_input, plan.thread)
                    result = task_result_from_state(task_state)
                elif plan.action is TaskDispatchAction.RESUME:
                    task_state = await self._dispatcher.resume(plan.thread)
                    result = task_result_from_state(task_state)
                else:
                    result = await self._dispatcher.terminal_result(plan.thread)
            if result.task_id != plan.task_id:
                raise ValueError("dispatcher returned a mismatched task result")
        except Exception:
            resumable = False
            try:
                inspection = await self._dispatcher.inspect(plan.thread)
                resumable = inspection.exists and inspection.resumable
            except Exception:
                pass
            result = TaskResult(
                task_id=plan.task_id,
                status=TaskStatus.BLOCKED,
                errors=[f"dispatch:{RunErrorCode.TASK_EXECUTION_INTERRUPTED.value}"],
            )
            record = TaskDispatchRecord(
                task_id=plan.task_id,
                action=plan.action,
                terminal_status=TaskStatus.BLOCKED,
                checkpoint_resumable=resumable,
                error_code=RunErrorCode.TASK_EXECUTION_INTERRUPTED,
            )
            return {
                "task_results": {plan.task_id: result},
                "dispatch_records": {plan.task_id: record},
                "run_errors": [
                    RunError(
                        code=RunErrorCode.TASK_EXECUTION_INTERRUPTED,
                        node="dispatch_task",
                        task_id=plan.task_id,
                    )
                ],
            }
        return {
            "task_results": {plan.task_id: result},
            "dispatch_records": {
                plan.task_id: TaskDispatchRecord(
                    task_id=plan.task_id,
                    action=plan.action,
                    terminal_status=result.status,
                )
            },
        }

    @staticmethod
    def _collect_results(state: GaiaRunState) -> dict[str, object]:
        selected = set(state["selected_task_ids"])
        result_ids = set(state["task_results"])
        record_ids = set(state["dispatch_records"])
        if result_ids != selected or record_ids != selected:
            return RunGraph._failure(
                RunErrorCode.INCOMPLETE_COLLECTION,
                "collect_results",
                detail_codes=tuple(sorted(selected - result_ids)),
            )
        return {
            "status": RunStatus.COMPLETE,
            "node_history": ["dispatch_tasks", "collect_results"],
        }

    @staticmethod
    def _block(state: GaiaRunState) -> dict[str, object]:
        del state
        return {"status": RunStatus.BLOCKED, "node_history": ["block"]}

    @staticmethod
    def _continue_or_block(state: GaiaRunState) -> Literal["continue", "block"]:
        return "block" if state["status"] is RunStatus.BLOCKED else "continue"

    @staticmethod
    def _failure(
        code: RunErrorCode,
        node: str,
        *,
        detail_codes: tuple[str, ...] = (),
    ) -> dict[str, object]:
        return {
            "status": RunStatus.BLOCKED,
            "run_errors": [
                RunError(code=code, node=node, detail_codes=detail_codes)
            ],
            "node_history": [node],
        }


def build_run_graph(
    *,
    dispatcher: CheckpointedTaskDispatcher,
    max_concurrency: int = 4,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[GaiaRunState, None, GaiaRunState, GaiaRunState]:
    """Build the A18 RunGraph around a durable per-task dispatcher."""

    return RunGraph(
        dispatcher=dispatcher,
        max_concurrency=max_concurrency,
    ).compile(checkpointer=checkpointer)


def run_result_from_state(state: GaiaRunState) -> GaiaRunResult:
    """Convert terminal graph state into a deterministic run result."""

    status = state["status"]
    if status not in {RunStatus.COMPLETE, RunStatus.BLOCKED}:
        raise ValueError("run graph state is not terminal")
    selected = state["selected_task_ids"]
    ordered_results = {
        task_id: state["task_results"][task_id]
        for task_id in selected
        if task_id in state["task_results"]
    }
    ordered_records = {
        task_id: state["dispatch_records"][task_id]
        for task_id in selected
        if task_id in state["dispatch_records"]
    }
    errors = tuple(state["run_errors"])
    if status is RunStatus.BLOCKED and not errors:
        errors = (
            RunError(
                code=RunErrorCode.INVALID_TERMINAL_STATE,
                node="run_graph",
            ),
        )
    return GaiaRunResult(
        run_id=state["run_id"],
        status=status,
        selected_task_ids=selected,
        task_results=ordered_results,
        dispatch_records=ordered_records,
        errors=errors,
    )


def _merge_unique_mapping(
    left: dict[str, ValueT],
    right: dict[str, ValueT],
    *,
    label: str,
) -> dict[str, ValueT]:
    merged = dict(left)
    for task_id, value in right.items():
        existing = merged.get(task_id)
        if existing is not None and existing != value:
            raise ValueError(f"conflicting duplicate {label} for task {task_id}")
        merged[task_id] = value
    return {task_id: merged[task_id] for task_id in sorted(merged)}
