"""Async SQLite ownership, task threads, resume, and safe history inspection."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaia_max.answer_validation import (
    AnswerValidationCode,
    AnswerValidationIssue,
    FinalAnswerValidationResult,
    FinalAnswerValidator,
)
from gaia_max.checkpoint_serde import build_safe_checkpoint_serializer
from gaia_max.domain.models import (
    AnswerType,
    ChessMoveAnswer,
    DecimalAnswer,
    EntityAnswer,
    IntegerAnswer,
    ListAnswer,
    Modality,
    NameScope,
    OutputContract,
    Question,
    RiskLevel,
    SortPolicy,
    StringAnswer,
    TaskClass,
    TaskProfile,
    TaskResult,
    TaskStatus,
    TemporalConstraint,
    UnitsPolicy,
    Verdict,
    VerificationResult,
)
from gaia_max.output_contracts import ContractParseStatus
from gaia_max.serialization import AnswerSerializer
from gaia_max.solver_routing import (
    RouteDecision,
    RouteDecisionSource,
    RouteSignal,
    RouteSignalCode,
    SolverRegistry,
    SolverRoute,
)
from gaia_max.task_analysis import (
    AnalysisAuthority,
    AnalysisDisagreement,
    ReconciledTaskAnalysis,
    TaskAnalyzer,
)
from gaia_max.task_graph import (
    TaskGraphError,
    TaskGraphErrorCode,
    TaskGraphInput,
    TaskGraphState,
    TaskVerifier,
    build_task_graph,
    task_result_from_state,
)

_THREAD_PART_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
TASK_CHECKPOINT_TYPES: tuple[type[Any], ...] = (
    AnswerType,
    ChessMoveAnswer,
    DecimalAnswer,
    EntityAnswer,
    IntegerAnswer,
    ListAnswer,
    Modality,
    NameScope,
    OutputContract,
    Question,
    RiskLevel,
    SortPolicy,
    StringAnswer,
    TaskClass,
    TaskProfile,
    TaskStatus,
    TemporalConstraint,
    UnitsPolicy,
    Verdict,
    VerificationResult,
    ContractParseStatus,
    AnalysisAuthority,
    AnalysisDisagreement,
    ReconciledTaskAnalysis,
    AnswerValidationCode,
    AnswerValidationIssue,
    FinalAnswerValidationResult,
    TaskGraphErrorCode,
    TaskGraphError,
    SolverRoute,
    RouteDecisionSource,
    RouteSignalCode,
    RouteSignal,
    RouteDecision,
)


class CheckpointRuntimeError(RuntimeError):
    """Base error for safe checkpoint lifecycle and thread operations."""


class CheckpointRuntimeClosedError(CheckpointRuntimeError):
    """The runtime operation needs an open async context."""


class CheckpointThreadExistsError(CheckpointRuntimeError):
    """A fresh run attempted to reuse an existing persistent thread."""


class CheckpointThreadNotFoundError(CheckpointRuntimeError):
    """No checkpoint exists for the requested thread."""


class CheckpointThreadNotResumableError(CheckpointRuntimeError):
    """The requested thread is already terminal."""


class CheckpointThreadNotTerminalError(CheckpointRuntimeError):
    """A terminal result was requested from an unfinished thread."""


class CheckpointTaskMismatchError(CheckpointRuntimeError):
    """The thread task identity does not match the supplied graph input."""


class TaskThread(BaseModel):
    """Stable, path-safe checkpoint identity for one task inside one run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1, max_length=64, pattern=_THREAD_PART_PATTERN)
    task_id: str = Field(min_length=1, max_length=128, pattern=_THREAD_PART_PATTERN)

    @property
    def thread_id(self) -> str:
        return f"gaia:{self.run_id}:{self.task_id}"

    def config(self) -> RunnableConfig:
        return {"configurable": {"thread_id": self.thread_id}}


class TaskCheckpointSnapshot(BaseModel):
    """Answer-free checkpoint summary for debugging and learner inspection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    thread_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    created_at: datetime | None = None
    step: int
    source: str
    next_nodes: tuple[str, ...] = ()
    status: TaskStatus | None = None
    node_history: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()


class TaskCheckpointInspection(BaseModel):
    """Answer-free information used by RunGraph dispatch planning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    thread_id: str = Field(min_length=1)
    exists: bool
    resumable: bool
    status: TaskStatus | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> TaskCheckpointInspection:
        if not self.exists and (self.resumable or self.status is not None):
            raise ValueError("missing checkpoint cannot be resumable or have status")
        return self


class AsyncSQLiteTaskRuntime:
    """Own one SQLite connection and a checkpointed typed task graph."""

    def __init__(
        self,
        database_path: Path,
        *,
        solver_registry: SolverRegistry,
        verifier: TaskVerifier,
        analyzer: TaskAnalyzer | None = None,
        serializer: AnswerSerializer | None = None,
        answer_validator: FinalAnswerValidator | None = None,
    ) -> None:
        resolved_path = database_path.resolve(strict=False)
        if resolved_path.suffix.casefold() not in {".db", ".sqlite", ".sqlite3"}:
            raise ValueError("checkpoint database must use .db, .sqlite, or .sqlite3")
        if resolved_path.exists() and resolved_path.is_dir():
            raise ValueError("checkpoint database path cannot be a directory")
        self.database_path = resolved_path
        self._solver_registry = solver_registry
        self._verifier = verifier
        self._analyzer = analyzer
        self._serializer = serializer
        self._answer_validator = answer_validator
        self._connection: aiosqlite.Connection | None = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._graph: (
            CompiledStateGraph[TaskGraphState, None, TaskGraphState, TaskGraphState] | None
        ) = None

    async def __aenter__(self) -> Self:
        if self._connection is not None:
            raise CheckpointRuntimeError("checkpoint runtime is already open")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(str(self.database_path))
        serde = build_safe_checkpoint_serializer(TASK_CHECKPOINT_TYPES)
        checkpointer = AsyncSqliteSaver(connection, serde=serde)
        try:
            await checkpointer.setup()
        except BaseException:
            await connection.close()
            raise
        self._connection = connection
        self._checkpointer = checkpointer
        self._graph = build_task_graph(
            solver_registry=self._solver_registry,
            verifier=self._verifier,
            analyzer=self._analyzer,
            serializer=self._serializer,
            answer_validator=self._answer_validator,
            checkpointer=checkpointer,
            propagate_transient_exceptions=True,
        )
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        connection = self._connection
        self._graph = None
        self._checkpointer = None
        self._connection = None
        if connection is not None:
            await connection.close()

    async def start(
        self,
        task_input: TaskGraphInput,
        thread: TaskThread,
    ) -> TaskGraphState:
        """Start exactly once; accidental thread reuse is rejected."""

        if task_input.question.task_id != thread.task_id:
            raise CheckpointTaskMismatchError(
                "checkpoint thread task ID does not match graph input"
            )
        graph = self._require_graph()
        current = await graph.aget_state(thread.config())
        if current.values:
            raise CheckpointThreadExistsError(
                f"checkpoint thread {thread.thread_id!r} already exists"
            )
        result = await graph.ainvoke(
            task_input.initial_state(),
            config=thread.config(),
            durability="sync",
        )
        return cast(TaskGraphState, result)

    async def resume(self, thread: TaskThread) -> TaskGraphState:
        """Continue from the latest nonterminal checkpoint using no new input."""

        graph = self._require_graph()
        current = await graph.aget_state(thread.config())
        if not current.values:
            raise CheckpointThreadNotFoundError(
                f"checkpoint thread {thread.thread_id!r} does not exist"
            )
        if not current.next:
            raise CheckpointThreadNotResumableError(
                f"checkpoint thread {thread.thread_id!r} is already terminal"
            )
        result = await graph.ainvoke(
            None,
            config=thread.config(),
            durability="sync",
        )
        return cast(TaskGraphState, result)

    async def latest(self, thread: TaskThread) -> TaskCheckpointSnapshot | None:
        """Return an answer-free summary of the newest checkpoint."""

        snapshot = await self._require_graph().aget_state(thread.config())
        if not snapshot.values:
            return None
        return _summarize_snapshot(thread, snapshot)

    async def inspect(self, thread: TaskThread) -> TaskCheckpointInspection:
        """Inspect lifecycle state without exposing any candidate answer."""

        snapshot = await self._require_graph().aget_state(thread.config())
        if not snapshot.values:
            return TaskCheckpointInspection(
                thread_id=thread.thread_id,
                exists=False,
                resumable=False,
            )
        return TaskCheckpointInspection(
            thread_id=thread.thread_id,
            exists=True,
            resumable=bool(snapshot.next),
            status=_snapshot_status(snapshot),
        )

    async def terminal_result(self, thread: TaskThread) -> TaskResult:
        """Restore one terminal task result for RunGraph collection without rerunning it."""

        snapshot = await self._require_graph().aget_state(thread.config())
        if not snapshot.values:
            raise CheckpointThreadNotFoundError(
                f"checkpoint thread {thread.thread_id!r} does not exist"
            )
        if snapshot.next:
            raise CheckpointThreadNotTerminalError(
                f"checkpoint thread {thread.thread_id!r} is not terminal"
            )
        return task_result_from_state(cast(TaskGraphState, snapshot.values))

    async def history(
        self,
        thread: TaskThread,
        *,
        limit: int | None = None,
    ) -> tuple[TaskCheckpointSnapshot, ...]:
        """Return newest-first, answer-free checkpoint history."""

        if limit is not None and limit < 1:
            raise ValueError("checkpoint history limit must be positive")
        snapshots = self._require_graph().aget_state_history(
            thread.config(),
            limit=limit,
        )
        return tuple(
            [_summarize_snapshot(thread, snapshot) async for snapshot in snapshots]
        )

    def _require_graph(
        self,
    ) -> CompiledStateGraph[TaskGraphState, None, TaskGraphState, TaskGraphState]:
        if self._graph is None:
            raise CheckpointRuntimeClosedError(
                "checkpoint runtime must be used inside 'async with'"
            )
        return self._graph


def _summarize_snapshot(
    thread: TaskThread,
    snapshot: StateSnapshot,
) -> TaskCheckpointSnapshot:
    values = snapshot.values if isinstance(snapshot.values, Mapping) else {}
    configurable = snapshot.config.get("configurable", {})
    checkpoint_id = str(configurable.get("checkpoint_id", "unknown"))
    metadata = snapshot.metadata or {}
    status = _snapshot_status(snapshot)
    raw_history = values.get("node_history", ())
    node_history = (
        tuple(str(node) for node in raw_history)
        if isinstance(raw_history, list | tuple)
        else ()
    )
    raw_errors = values.get("errors", ())
    error_codes: list[str] = []
    if isinstance(raw_errors, list | tuple):
        for error in raw_errors:
            if isinstance(error, TaskGraphError):
                error_codes.append(error.code.value)
            elif isinstance(error, Mapping) and "code" in error:
                error_codes.append(str(error["code"]))
    return TaskCheckpointSnapshot(
        thread_id=thread.thread_id,
        checkpoint_id=checkpoint_id,
        created_at=(
            datetime.fromisoformat(snapshot.created_at.replace("Z", "+00:00"))
            if snapshot.created_at is not None
            else None
        ),
        step=int(metadata.get("step", -1)),
        source=str(metadata.get("source", "unknown")),
        next_nodes=snapshot.next,
        status=status,
        node_history=node_history,
        error_codes=tuple(error_codes),
    )


def _snapshot_status(snapshot: StateSnapshot) -> TaskStatus | None:
    values = snapshot.values if isinstance(snapshot.values, Mapping) else {}
    raw_status = values.get("status")
    if isinstance(raw_status, TaskStatus):
        return raw_status
    if isinstance(raw_status, str):
        try:
            return TaskStatus(raw_status)
        except ValueError:
            return None
    return None
