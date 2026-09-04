from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gaia_max.checkpointing import (
    AsyncSQLiteTaskRuntime,
    CheckpointRuntimeClosedError,
    CheckpointTaskMismatchError,
    CheckpointThreadExistsError,
    CheckpointThreadNotFoundError,
    CheckpointThreadNotResumableError,
    CheckpointThreadNotTerminalError,
    TaskCheckpointInspection,
    TaskCheckpointSnapshot,
    TaskThread,
)
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
from gaia_max.snapshot import question_sha256
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.task_analysis import ReconciledTaskAnalysis
from gaia_max.task_graph import TaskGraphInput, TaskNodeExecutionError


class CountingSolver:
    def __init__(self, value: int = 42) -> None:
        self.value = value
        self.calls = 0

    async def solve(
        self,
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
        _route: str,
    ) -> object:
        self.calls += 1
        return IntegerAnswer(value=self.value)


class ApprovingVerifier:
    def __init__(self) -> None:
        self.calls = 0

    async def verify(
        self,
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
        _answer: SemanticAnswer,
    ) -> object:
        self.calls += 1
        return VerificationResult(
            verifier="checkpoint-test-verifier",
            tier=0,
            verdict=Verdict.APPROVE,
        )


class FailOnceVerifier(ApprovingVerifier):
    async def verify(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        answer: SemanticAnswer,
    ) -> object:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("private transient provider payload")
        return VerificationResult(
            verifier="checkpoint-test-verifier",
            tier=0,
            verdict=Verdict.APPROVE,
        )


def question(task_id: str = "checkpoint-task-1") -> Question:
    return Question(task_id=task_id, question="How many records are there?")


def profile(runtime_question: Question) -> TaskProfile:
    return TaskProfile(
        task_id=runtime_question.task_id,
        question=runtime_question.question,
        question_sha256=question_sha256(runtime_question),
        file_name=runtime_question.file_name,
        modality=Modality.WEB,
        task_class=TaskClass.STRUCTURED_TABLE,
        route="structured_table",
        risk_level=RiskLevel.LOW,
        requested_operation="count records deterministically",
        required_sources=["synthetic_table"],
        output_contract=OutputContract(answer_type=AnswerType.INTEGER),
        verification_policy="synthetic_deterministic_replay",
    )


def task_input(runtime_question: Question | None = None) -> TaskGraphInput:
    resolved = runtime_question or question()
    return TaskGraphInput(question=resolved, profile=profile(resolved))


def runtime(
    database_path: Path,
    *,
    solver: CountingSolver | None = None,
    verifier: ApprovingVerifier | None = None,
) -> AsyncSQLiteTaskRuntime:
    registry = SolverRegistry()
    registry.register(SolverRoute.STRUCTURED_TABLE, solver or CountingSolver())
    return AsyncSQLiteTaskRuntime(
        database_path,
        solver_registry=registry,
        verifier=verifier or ApprovingVerifier(),
    )


@pytest.mark.asyncio
async def test_successful_run_persists_answer_free_inspectable_history(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "checkpoints.sqlite3"
    task = question()
    thread = TaskThread(run_id="run-001", task_id=task.task_id)

    async with runtime(database_path) as checkpoint_runtime:
        state = await checkpoint_runtime.start(task_input(task), thread)
        latest = await checkpoint_runtime.latest(thread)
        history = await checkpoint_runtime.history(thread)
        inspection = await checkpoint_runtime.inspect(thread)
        restored_result = await checkpoint_runtime.terminal_result(thread)

    assert state["status"] is TaskStatus.READY
    assert state["serialized_answer"] == "42"
    assert database_path.is_file()
    assert database_path.stat().st_size > 0
    assert latest is not None
    assert latest.status is TaskStatus.READY
    assert latest.next_nodes == ()
    assert latest.node_history[-1] == "validate"
    assert len(history) >= 8
    assert history[0] == latest
    assert inspection == TaskCheckpointInspection(
        thread_id=thread.thread_id,
        exists=True,
        resumable=False,
        status=TaskStatus.READY,
    )
    assert restored_result.status is TaskStatus.READY
    assert restored_result.serialized_answer == "42"
    assert set(TaskCheckpointSnapshot.model_fields) == {
        "thread_id",
        "checkpoint_id",
        "created_at",
        "step",
        "source",
        "next_nodes",
        "status",
        "node_history",
        "error_codes",
    }


@pytest.mark.asyncio
async def test_failed_verifier_resumes_after_runtime_reopen_without_repeating_solver(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "resume.sqlite3"
    task = question()
    thread = TaskThread(run_id="resume-run", task_id=task.task_id)
    solver = CountingSolver()
    verifier = FailOnceVerifier()

    async with runtime(database_path, solver=solver, verifier=verifier) as first_runtime:
        with pytest.raises(TaskNodeExecutionError) as caught:
            await first_runtime.start(task_input(task), thread)
        failed_snapshot = await first_runtime.latest(thread)
        inspection = await first_runtime.inspect(thread)
        with pytest.raises(CheckpointThreadNotTerminalError):
            await first_runtime.terminal_result(thread)

    assert caught.value.node == "verify"
    assert "private transient provider payload" not in str(caught.value)
    assert solver.calls == 1
    assert verifier.calls == 1
    assert failed_snapshot is not None
    assert failed_snapshot.next_nodes == ("verify",)
    assert failed_snapshot.node_history == ("prepare", "analyze", "route", "solve")
    assert inspection.exists is True
    assert inspection.resumable is True
    assert inspection.status is TaskStatus.VERIFYING

    async with runtime(database_path, solver=solver, verifier=verifier) as second_runtime:
        state = await second_runtime.resume(thread)
        latest = await second_runtime.latest(thread)

    assert state["status"] is TaskStatus.READY
    assert state["node_history"] == [
        "prepare",
        "analyze",
        "route",
        "solve",
        "verify",
        "serialize",
        "validate",
    ]
    assert solver.calls == 1
    assert verifier.calls == 2
    assert latest is not None
    assert latest.status is TaskStatus.READY


@pytest.mark.asyncio
async def test_start_rejects_existing_thread_without_repeating_side_effects(
    tmp_path: Path,
) -> None:
    task = question()
    thread = TaskThread(run_id="one-shot-run", task_id=task.task_id)
    solver = CountingSolver()

    async with runtime(tmp_path / "one-shot.db", solver=solver) as checkpoint_runtime:
        await checkpoint_runtime.start(task_input(task), thread)
        with pytest.raises(CheckpointThreadExistsError):
            await checkpoint_runtime.start(task_input(task), thread)

    assert solver.calls == 1


@pytest.mark.asyncio
async def test_terminal_thread_cannot_be_resumed(tmp_path: Path) -> None:
    task = question()
    thread = TaskThread(run_id="terminal-run", task_id=task.task_id)

    async with runtime(tmp_path / "terminal.db") as checkpoint_runtime:
        await checkpoint_runtime.start(task_input(task), thread)
        with pytest.raises(CheckpointThreadNotResumableError):
            await checkpoint_runtime.resume(thread)


@pytest.mark.asyncio
async def test_blocked_thread_history_includes_reason_codes(tmp_path: Path) -> None:
    task = question()
    thread = TaskThread(run_id="blocked-run", task_id=task.task_id)

    class RejectingVerifier(ApprovingVerifier):
        async def verify(
            self,
            _question: Question,
            _analysis: ReconciledTaskAnalysis,
            _answer: SemanticAnswer,
        ) -> object:
            self.calls += 1
            return VerificationResult(
                verifier="checkpoint-test-verifier",
                tier=0,
                verdict=Verdict.REJECT,
                reason_codes=["fixture_reject"],
            )

    async with runtime(
        tmp_path / "blocked.db",
        verifier=RejectingVerifier(),
    ) as checkpoint_runtime:
        state = await checkpoint_runtime.start(task_input(task), thread)
        history = await checkpoint_runtime.history(thread)

    assert state["status"] is TaskStatus.BLOCKED
    assert history[0].status is TaskStatus.BLOCKED
    assert "verification_not_approved" in history[0].error_codes
    assert any(snapshot.next_nodes == ("block",) for snapshot in history)


@pytest.mark.asyncio
async def test_unknown_thread_cannot_be_resumed(tmp_path: Path) -> None:
    thread = TaskThread(run_id="missing-run", task_id=question().task_id)

    async with runtime(tmp_path / "missing.db") as checkpoint_runtime:
        assert await checkpoint_runtime.latest(thread) is None
        assert await checkpoint_runtime.history(thread) == ()
        assert await checkpoint_runtime.inspect(thread) == TaskCheckpointInspection(
            thread_id=thread.thread_id,
            exists=False,
            resumable=False,
        )
        with pytest.raises(CheckpointThreadNotFoundError):
            await checkpoint_runtime.resume(thread)
        with pytest.raises(CheckpointThreadNotFoundError):
            await checkpoint_runtime.terminal_result(thread)


@pytest.mark.asyncio
async def test_threads_are_isolated_in_the_same_database(tmp_path: Path) -> None:
    task = question()
    first = TaskThread(run_id="isolated-a", task_id=task.task_id)
    second = TaskThread(run_id="isolated-b", task_id=task.task_id)
    solver = CountingSolver()

    async with runtime(tmp_path / "isolated.sqlite", solver=solver) as checkpoint_runtime:
        await checkpoint_runtime.start(task_input(task), first)
        await checkpoint_runtime.start(task_input(task), second)
        first_history = await checkpoint_runtime.history(first)
        second_history = await checkpoint_runtime.history(second)

    assert solver.calls == 2
    assert first_history
    assert second_history
    assert {snapshot.thread_id for snapshot in first_history} == {first.thread_id}
    assert {snapshot.thread_id for snapshot in second_history} == {second.thread_id}


@pytest.mark.asyncio
async def test_thread_task_id_must_match_graph_input(tmp_path: Path) -> None:
    supplied = question("supplied-task")
    thread = TaskThread(run_id="mismatch-run", task_id="different-task")

    async with runtime(tmp_path / "mismatch.db") as checkpoint_runtime:
        with pytest.raises(CheckpointTaskMismatchError):
            await checkpoint_runtime.start(task_input(supplied), thread)


@pytest.mark.parametrize(
    ("run_id", "task_id"),
    [
        ("../escape", "task-1"),
        ("run:injected", "task-1"),
        ("run-1", "task/escape"),
        ("", "task-1"),
    ],
)
def test_thread_identity_rejects_unsafe_parts(run_id: str, task_id: str) -> None:
    with pytest.raises(ValidationError):
        TaskThread(run_id=run_id, task_id=task_id)


@pytest.mark.asyncio
async def test_runtime_must_be_open_for_operations(tmp_path: Path) -> None:
    checkpoint_runtime = runtime(tmp_path / "closed.db")
    thread = TaskThread(run_id="closed-run", task_id=question().task_id)

    with pytest.raises(CheckpointRuntimeClosedError):
        await checkpoint_runtime.latest(thread)


def test_checkpoint_path_requires_sqlite_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint database"):
        runtime(tmp_path / "checkpoints.txt")


@pytest.mark.asyncio
async def test_history_limit_is_validated_and_applied(tmp_path: Path) -> None:
    task = question()
    thread = TaskThread(run_id="history-run", task_id=task.task_id)

    async with runtime(tmp_path / "history.db") as checkpoint_runtime:
        await checkpoint_runtime.start(task_input(task), thread)
        assert len(await checkpoint_runtime.history(thread, limit=2)) == 2
        with pytest.raises(ValueError, match="must be positive"):
            await checkpoint_runtime.history(thread, limit=0)
