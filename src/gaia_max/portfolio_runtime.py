"""Runnable, answer-free portfolio scenario over the real GAIA graph stack.

The official evaluation involved private evidence and candidate records.  This module
provides a public reproducible vertical slice without publishing any benchmark answer:
two synthetic questions exercise routing, specialist execution, independent replay,
serialization, SQLite checkpoints, bounded fan-out, and terminal result collection.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from gaia_max.checkpointing import AsyncSQLiteTaskRuntime
from gaia_max.config import Settings
from gaia_max.deterministic_verification import DeterministicVerificationRegistry
from gaia_max.domain import (
    AnswerType,
    Modality,
    OutputContract,
    Question,
    RiskLevel,
    SortPolicy,
    TaskClass,
    TaskStatus,
)
from gaia_max.profiles import TaskProfileRecord, TaskProfileRegistry
from gaia_max.run_graph import (
    GaiaRunState,
    RunGraphInput,
    RunStatus,
    RunTaskSelection,
    TaskDispatchAction,
    build_run_graph,
    run_result_from_state,
)
from gaia_max.snapshot import build_question_snapshot, question_sha256
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.solvers import (
    register_operation_table_solver,
    register_transformed_text_solver,
    reverse_exact,
)

PORTFOLIO_DEMO_VERSION = "portfolio-demo-v1"
PORTFOLIO_DEMO_SOURCE = "https://example.invalid/gaia-max/portfolio-demo"


class PublicDemoTaskResult(BaseModel):
    """One intentionally public synthetic result; never used for official tasks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    route: SolverRoute
    status: TaskStatus
    checkpoint_action: TaskDispatchAction
    serialized_answer: str | None = None
    errors: tuple[str, ...] = ()


class PublicDemoRunReport(BaseModel):
    """Serializable proof that the production graph stack ran end to end."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    status: RunStatus
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_task_ids: tuple[str, ...]
    tasks: tuple[PublicDemoTaskResult, ...]
    graph_path: tuple[str, ...] = (
        "snapshot_validation",
        "profile_reconciliation",
        "deterministic_routing",
        "specialist_solver",
        "independent_replay",
        "exact_serialization",
        "checkpointed_collection",
    )
    answers_are_synthetic: bool = True
    submission_capability_present: bool = False


def _demo_questions() -> tuple[Question, ...]:
    transform_instruction = (
        'If you understand this sentence, write the opposite of the word "north" '
        "as the answer."
    )
    operation_table = """Given this table defining * on the set S = {z, a, m}

|*|z|a|m|
|---|---|---|---|
|m|z|z|z|
|z|z|z|z|
|a|a|z|z|

Provide the subset involved in counter-examples as a comma separated list in
alphabetical order.
"""
    return (
        Question(
            task_id="demo-transformed-text",
            question=reverse_exact(transform_instruction),
            level="synthetic",
        ),
        Question(
            task_id="demo-operation-table",
            question=operation_table,
            level="synthetic",
        ),
    )


def build_portfolio_demo_input(
    *,
    run_id: str,
    selected_task_ids: tuple[str, ...] = (),
) -> RunGraphInput:
    """Build a hash-bound, answer-free input for the public demonstration."""

    questions = _demo_questions()
    snapshot = build_question_snapshot(
        questions,
        source_url=PORTFOLIO_DEMO_SOURCE,
        task_profiles_version=PORTFOLIO_DEMO_VERSION,
        retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    records = [
        TaskProfileRecord(
            task_id=questions[0].task_id,
            question_sha256=question_sha256(questions[0]),
            modality=Modality.TEXT,
            task_class=TaskClass.DETERMINISTIC_TEXT,
            route=SolverRoute.DETERMINISTIC_TEXT.value,
            requested_operation="reverse exact text and evaluate the supported direction",
            output_contract=OutputContract(answer_type=AnswerType.STRING),
            verification_policy="independent_deterministic_replay",
            risk_level=RiskLevel.LOW,
            risk_flags=["exact_character_preservation"],
        ),
        TaskProfileRecord(
            task_id=questions[1].task_id,
            question_sha256=question_sha256(questions[1]),
            modality=Modality.TEXT,
            task_class=TaskClass.OPERATION_TABLE,
            route=SolverRoute.OPERATION_TABLE.value,
            requested_operation="exhaustively identify noncommutative elements",
            output_contract=OutputContract(
                answer_type=AnswerType.LIST,
                sort=SortPolicy.ALPHABETICAL,
                delimiter=", ",
            ),
            verification_policy="independent_deterministic_replay",
            risk_level=RiskLevel.LOW,
            risk_flags=["exhaustive_ordered_pairs"],
        ),
    ]
    registry = TaskProfileRegistry(
        schema_version=1,
        profiles_version=PORTFOLIO_DEMO_VERSION,
        question_snapshot_sha256=snapshot.snapshot_sha256,
        profile_count=len(records),
        profiles=records,
    )
    return RunGraphInput(
        run_id=run_id,
        snapshot=snapshot,
        questions=questions,
        profile_registry=registry,
        selection=RunTaskSelection(task_ids=selected_task_ids),
        expected_question_count=len(questions),
    )


def build_portfolio_solver_stack() -> tuple[SolverRegistry, DeterministicVerificationRegistry]:
    """Register only specialists that are complete and independently replayable."""

    solvers = SolverRegistry()
    verifier = DeterministicVerificationRegistry()
    transformed = register_transformed_text_solver(solvers)
    operation_table = register_operation_table_solver(solvers)
    verifier.register(SolverRoute.DETERMINISTIC_TEXT, transformed)
    verifier.register(SolverRoute.OPERATION_TABLE, operation_table)
    return solvers, verifier


async def run_portfolio_demo(
    settings: Settings,
    *,
    run_id: str = "portfolio-demo",
    selected_task_ids: tuple[str, ...] = (),
) -> PublicDemoRunReport:
    """Execute or resume the public scenario with durable per-task checkpoints."""

    graph_input = build_portfolio_demo_input(
        run_id=run_id,
        selected_task_ids=selected_task_ids,
    )
    solvers, verifier = build_portfolio_solver_stack()
    database_path = Path(settings.runs_dir) / "portfolio-demo" / f"{run_id}.sqlite3"
    async with AsyncSQLiteTaskRuntime(
        database_path,
        solver_registry=solvers,
        verifier=verifier,
    ) as dispatcher:
        graph = build_run_graph(
            dispatcher=dispatcher,
            max_concurrency=settings.max_task_concurrency,
        )
        state = cast(GaiaRunState, await graph.ainvoke(graph_input.initial_state()))

    result = run_result_from_state(state)
    profiles = graph_input.profile_registry.by_task_id()
    tasks = tuple(
        PublicDemoTaskResult(
            task_id=task_id,
            route=SolverRoute(profiles[task_id].route),
            status=result.task_results[task_id].status,
            checkpoint_action=result.dispatch_records[task_id].action,
            serialized_answer=result.task_results[task_id].serialized_answer,
            errors=tuple(result.task_results[task_id].errors),
        )
        for task_id in result.selected_task_ids
    )
    return PublicDemoRunReport(
        run_id=run_id,
        status=result.status,
        snapshot_sha256=graph_input.snapshot.snapshot_sha256,
        selected_task_ids=result.selected_task_ids,
        tasks=tasks,
    )


__all__ = [
    "PORTFOLIO_DEMO_VERSION",
    "PublicDemoRunReport",
    "PublicDemoTaskResult",
    "build_portfolio_demo_input",
    "build_portfolio_solver_stack",
    "run_portfolio_demo",
]
