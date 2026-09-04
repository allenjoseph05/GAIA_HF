from __future__ import annotations

import asyncio
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from gaia_max.checkpointing import AsyncSQLiteTaskRuntime
from gaia_max.domain import (
    AnswerType,
    IntegerAnswer,
    Modality,
    OutputContract,
    Question,
    QuestionSnapshot,
    RiskLevel,
    SemanticAnswer,
    TaskClass,
    TaskResult,
    TaskStatus,
    Verdict,
    VerificationResult,
)
from gaia_max.profiles import TaskProfileRecord, TaskProfileRegistry
from gaia_max.run_graph import (
    GaiaRunState,
    RunErrorCode,
    RunGraphInput,
    RunStatus,
    RunTaskSelection,
    TaskDispatchAction,
    build_run_checkpoint_serializer,
    build_run_graph,
    merge_task_results,
    run_result_from_state,
)
from gaia_max.snapshot import build_question_snapshot, question_sha256
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.task_analysis import ReconciledTaskAnalysis

NOW = datetime(2026, 8, 28, tzinfo=UTC)
SOURCE_URL = "https://course.test/questions"
PROFILE_VERSION = "synthetic-run-v1"


class TrackingSolver:
    def __init__(
        self,
        *,
        fail_once_task_id: str | None = None,
    ) -> None:
        self.fail_once_task_id = fail_once_task_id
        self.calls: Counter[str] = Counter()
        self.active = 0
        self.max_active = 0

    async def solve(
        self,
        question: Question,
        _analysis: ReconciledTaskAnalysis,
        _route: str,
    ) -> object:
        self.calls[question.task_id] += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            if (
                question.task_id == self.fail_once_task_id
                and self.calls[question.task_id] == 1
            ):
                raise RuntimeError("private synthetic solver failure")
            return IntegerAnswer(value=1)
        finally:
            self.active -= 1


class ApprovingVerifier:
    async def verify(
        self,
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
        _answer: SemanticAnswer,
    ) -> object:
        return VerificationResult(
            verifier="synthetic-run-verifier",
            tier=0,
            verdict=Verdict.APPROVE,
        )


def synthetic_inventory(
    count: int = 20,
) -> tuple[tuple[Question, ...], QuestionSnapshot, TaskProfileRegistry]:
    questions = tuple(
        Question(
            task_id=f"task-{index:02d}",
            question=f"How many records are in table {index}?",
            level="1",
        )
        for index in range(count)
    )
    snapshot = build_question_snapshot(
        questions,
        source_url=SOURCE_URL,
        task_profiles_version=PROFILE_VERSION,
        retrieved_at=NOW,
    )
    records = [
        TaskProfileRecord(
            task_id=question.task_id,
            question_sha256=question_sha256(question),
            modality=Modality.WEB,
            task_class=TaskClass.STRUCTURED_TABLE,
            route=SolverRoute.STRUCTURED_TABLE.value,
            requested_operation="count complete table rows",
            output_contract=OutputContract(answer_type=AnswerType.INTEGER),
            verification_policy="synthetic_deterministic_replay",
            risk_level=RiskLevel.LOW,
        )
        for question in questions
    ]
    profile_registry = TaskProfileRegistry(
        schema_version=1,
        profiles_version=PROFILE_VERSION,
        question_snapshot_sha256=snapshot.snapshot_sha256,
        profile_count=count,
        profiles=records,
    )
    return questions, snapshot, profile_registry


def run_input(
    *,
    count: int = 20,
    run_id: str = "synthetic-run",
    selected: tuple[str, ...] = (),
) -> RunGraphInput:
    questions, snapshot, registry = synthetic_inventory(count)
    return RunGraphInput(
        run_id=run_id,
        snapshot=snapshot,
        questions=questions,
        profile_registry=registry,
        selection=RunTaskSelection(task_ids=selected),
        expected_question_count=count,
    )


def task_runtime(database_path: Path, solver: TrackingSolver) -> AsyncSQLiteTaskRuntime:
    registry = SolverRegistry()
    registry.register(SolverRoute.STRUCTURED_TABLE, solver)
    return AsyncSQLiteTaskRuntime(
        database_path,
        solver_registry=registry,
        verifier=ApprovingVerifier(),
    )


@pytest.mark.asyncio
async def test_synthetic_twenty_task_run_completes_with_bounded_concurrency(
    tmp_path: Path,
) -> None:
    solver = TrackingSolver()
    task_input = run_input()

    async with task_runtime(tmp_path / "twenty.sqlite3", solver) as runtime:
        graph = build_run_graph(dispatcher=runtime, max_concurrency=4)
        state = await graph.ainvoke(task_input.initial_state())

    result = run_result_from_state(state)
    assert result.status is RunStatus.COMPLETE
    assert len(result.selected_task_ids) == 20
    assert len(result.task_results) == 20
    assert all(item.status is TaskStatus.READY for item in result.task_results.values())
    assert {record.action for record in result.dispatch_records.values()} == {
        TaskDispatchAction.START
    }
    assert 1 < solver.max_active <= 4
    assert set(solver.calls.values()) == {1}
    assert state["node_history"] == [
        "validate_snapshot",
        "plan_dispatch",
        "dispatch_tasks",
        "collect_results",
    ]


@pytest.mark.asyncio
async def test_interrupted_task_resumes_while_terminal_tasks_are_reused(
    tmp_path: Path,
) -> None:
    failed_task = "task-03"
    solver = TrackingSolver(fail_once_task_id=failed_task)
    task_input = run_input(count=8, run_id="resume-run")

    async with task_runtime(tmp_path / "resume-run.sqlite", solver) as runtime:
        graph = build_run_graph(dispatcher=runtime, max_concurrency=3)
        first_state = await graph.ainvoke(task_input.initial_state())
        first_result = run_result_from_state(first_state)

        assert first_result.status is RunStatus.COMPLETE
        assert first_result.task_results[failed_task].status is TaskStatus.BLOCKED
        assert first_result.dispatch_records[failed_task].checkpoint_resumable is True
        assert first_result.errors[0].code is RunErrorCode.TASK_EXECUTION_INTERRUPTED
        assert sum(
            item.status is TaskStatus.READY
            for item in first_result.task_results.values()
        ) == 7

        second_state = await graph.ainvoke(task_input.initial_state())

    second_result = run_result_from_state(second_state)
    assert all(item.status is TaskStatus.READY for item in second_result.task_results.values())
    assert second_result.dispatch_records[failed_task].action is TaskDispatchAction.RESUME
    assert {
        record.action
        for task_id, record in second_result.dispatch_records.items()
        if task_id != failed_task
    } == {TaskDispatchAction.REUSE}
    assert solver.calls[failed_task] == 2
    assert all(
        calls == 1 for task_id, calls in solver.calls.items() if task_id != failed_task
    )
    assert second_result.errors == ()


@pytest.mark.asyncio
async def test_explicit_task_selection_dispatches_only_requested_subset(
    tmp_path: Path,
) -> None:
    selected = ("task-02", "task-07", "task-15")
    solver = TrackingSolver()
    task_input = run_input(selected=selected, run_id="selected-run")

    async with task_runtime(tmp_path / "selected.sqlite", solver) as runtime:
        state = await build_run_graph(
            dispatcher=runtime,
            max_concurrency=2,
        ).ainvoke(task_input.initial_state())

    result = run_result_from_state(state)
    assert result.selected_task_ids == selected
    assert set(result.task_results) == set(selected)
    assert set(solver.calls) == set(selected)


@pytest.mark.asyncio
async def test_run_graph_supports_coordinator_checkpoint_inspection(
    tmp_path: Path,
) -> None:
    solver = TrackingSolver()
    task_input = run_input(count=2, run_id="coordinator-checkpoint-run")
    coordinator_config = {"configurable": {"thread_id": "run:coordinator-checkpoint"}}

    async with task_runtime(tmp_path / "coordinator.sqlite", solver) as runtime:
        graph = build_run_graph(
            dispatcher=runtime,
            max_concurrency=2,
            checkpointer=InMemorySaver(serde=build_run_checkpoint_serializer()),
        )
        state = await graph.ainvoke(
            task_input.initial_state(),
            config=coordinator_config,
        )
        checkpoint = await graph.aget_state(coordinator_config)

    assert state["status"] is RunStatus.COMPLETE
    assert checkpoint.values["status"] is RunStatus.COMPLETE
    assert checkpoint.next == ()
    assert checkpoint.values["node_history"][-1] == "collect_results"


@pytest.mark.asyncio
async def test_snapshot_mutation_blocks_before_checkpoint_inspection(
    tmp_path: Path,
) -> None:
    original = run_input(count=3, run_id="bad-snapshot-run")
    changed_questions = list(original.questions)
    changed_questions[0] = changed_questions[0].model_copy(
        update={"question": "Changed public wording"}
    )
    poisoned = original.model_copy(update={"questions": tuple(changed_questions)})
    solver = TrackingSolver()

    async with task_runtime(tmp_path / "bad-snapshot.sqlite", solver) as runtime:
        state = await build_run_graph(dispatcher=runtime).ainvoke(
            poisoned.initial_state()
        )

    result = run_result_from_state(state)
    assert result.status is RunStatus.BLOCKED
    assert result.errors[0].code is RunErrorCode.SNAPSHOT_HASH_MISMATCH
    assert solver.calls == Counter()
    assert state["node_history"] == ["validate_snapshot", "block"]


@pytest.mark.asyncio
async def test_unknown_selected_task_blocks_without_dispatch(
    tmp_path: Path,
) -> None:
    task_input = run_input(
        count=3,
        run_id="unknown-selection-run",
        selected=("not-in-snapshot",),
    )
    solver = TrackingSolver()

    async with task_runtime(tmp_path / "unknown-selection.sqlite", solver) as runtime:
        state = await build_run_graph(dispatcher=runtime).ainvoke(
            task_input.initial_state()
        )

    result = run_result_from_state(state)
    assert result.status is RunStatus.BLOCKED
    assert result.errors[0].code is RunErrorCode.UNKNOWN_TASK_SELECTION
    assert solver.calls == Counter()
    assert state["node_history"] == ["validate_snapshot", "plan_dispatch", "block"]


def test_parallel_result_reducer_rejects_conflicting_duplicate_task_write() -> None:
    first = TaskResult(
        task_id="task-1",
        status=TaskStatus.BLOCKED,
        errors=["first"],
    )
    conflicting = TaskResult(
        task_id="task-1",
        status=TaskStatus.BLOCKED,
        errors=["second"],
    )

    with pytest.raises(ValueError, match="conflicting duplicate task result"):
        merge_task_results({"task-1": first}, {"task-1": conflicting})


def test_run_graph_renders_fanout_and_collection_nodes() -> None:
    solver = TrackingSolver()
    runtime = task_runtime(Path("synthetic.db"), solver)
    graph = build_run_graph(dispatcher=runtime, max_concurrency=2)

    mermaid = graph.get_graph().draw_mermaid()

    for node in (
        "validate_snapshot",
        "plan_dispatch",
        "dispatch_task",
        "collect_results",
        "block",
    ):
        assert node in mermaid
    assert "__start__" in mermaid
    assert "__end__" in mermaid


def test_run_graph_state_has_no_gold_or_submission_fields() -> None:
    assert "gold_answer" not in GaiaRunState.__annotations__
    assert "submitted_answer" not in GaiaRunState.__annotations__
    assert "submission_receipt" not in GaiaRunState.__annotations__


def test_run_input_rejects_answer_bearing_extra_fields() -> None:
    valid = run_input(count=2)
    payload = valid.model_dump()
    payload["gold_answer"] = "forbidden"

    with pytest.raises(ValueError):
        RunGraphInput.model_validate(payload)
