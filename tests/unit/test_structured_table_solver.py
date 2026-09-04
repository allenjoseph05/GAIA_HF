from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from gaia_max.domain import (
    AnswerType,
    EntityAnswer,
    IntegerAnswer,
    Modality,
    OutputContract,
    Question,
    TaskClass,
)
from gaia_max.output_contracts import ContractParseStatus
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.solvers.structured_table import (
    Extreme,
    StructuredTableSolver,
    StructuredTableSolverError,
    TableIntegrityError,
    TableReductionInput,
    UnresolvedTieError,
    coerce_table_integer,
    reduce_structured_table,
    register_structured_table_solver,
)
from gaia_max.task_analysis import AnalysisAuthority, ReconciledTaskAnalysis


def analysis(
    *,
    answer_type: AnswerType = AnswerType.INTEGER,
    task_class: TaskClass = TaskClass.STRUCTURED_TABLE,
    modality: Modality = Modality.WEB,
) -> ReconciledTaskAnalysis:
    return ReconciledTaskAnalysis(
        task_id="structured-table-task",
        modality=modality,
        task_class=task_class,
        requested_operation="reduce a complete sports table",
        output_contract=OutputContract(answer_type=answer_type),
        risk_flags=("complete_table", "tie_assertion", "same_row_field"),
        field_sources={
            "modality": AnalysisAuthority.PROFILE,
            "task_class": AnalysisAuthority.PROFILE,
            "requested_operation": AnalysisAuthority.PROFILE,
            "temporal_constraint": AnalysisAuthority.PROFILE,
            "filters": AnalysisAuthority.PROFILE,
            "output_contract": AnalysisAuthority.PROFILE,
            "risk_flags": AnalysisAuthority.PROFILE,
        },
        contract_parse_status=ContractParseStatus.COMPLETE,
        profile_used=True,
        model_used=False,
        requires_review=False,
    )


def yankees_like_input() -> TableReductionInput:
    return TableReductionInput(
        rows=(
            {"player": "Ada", "walks": "41", "at_bats": "311"},
            {"player": "Ben", "walks": " 58 ", "at_bats": "427"},
            {"player": "Cy", "walks": 29, "at_bats": 205},
        ),
        identity_field="player",
        metric_field="walks",
        result_field="at_bats",
        extreme=Extreme.MAXIMUM,
    )


def olympics_like_input() -> TableReductionInput:
    return TableReductionInput(
        rows=(
            {"country": "Zedland", "athletes": "2", "ioc_code": "ZED"},
            {"country": "Aland", "athletes": 2, "ioc_code": "ALA"},
            {"country": "Midland", "athletes": "14", "ioc_code": "MID"},
        ),
        identity_field="country",
        metric_field="athletes",
        result_field="ioc_code",
        extreme=Extreme.MINIMUM,
        tie_break_field="country",
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), (-12, -12), (" 42 ", 42), ("1,234", 1234), ("-2,000", -2000)],
)
def test_integer_coercion_is_explicit(value: object, expected: int) -> None:
    assert coerce_table_integer(value, field="metric") == expected


@pytest.mark.parametrize("value", [True, 1.0, None, "", "1.5", "1,00", "N/A"])
def test_invalid_numeric_cells_fail(value: object) -> None:
    with pytest.raises(TableIntegrityError):
        coerce_table_integer(value, field="metric")


def test_argmax_returns_requested_value_from_the_selected_same_row() -> None:
    result = reduce_structured_table(yankees_like_input())

    assert result.selected_identity == "Ben"
    assert result.selected_row_index == 1
    assert result.metric_value == 58
    assert result.result_value == "427"
    assert result.tie_count == 1
    assert result.rows_examined == 3


def test_argmin_uses_alphabetical_tie_break_before_returning_code() -> None:
    result = reduce_structured_table(olympics_like_input())

    assert result.selected_identity == "Aland"
    assert result.metric_value == 2
    assert result.result_value == "ALA"
    assert result.tie_count == 2
    assert result.rows_examined == 3


def test_unresolved_or_non_unique_ties_fail_instead_of_using_row_order() -> None:
    no_tie_break = olympics_like_input()
    no_tie_break = TableReductionInput(
        rows=no_tie_break.rows,
        identity_field=no_tie_break.identity_field,
        metric_field=no_tie_break.metric_field,
        result_field=no_tie_break.result_field,
        extreme=no_tie_break.extreme,
    )
    duplicate_tie_values = TableReductionInput(
        rows=(
            {"team": "One", "score": 3, "label": "same", "result": "x"},
            {"team": "Two", "score": 3, "label": "SAME", "result": "y"},
        ),
        identity_field="team",
        metric_field="score",
        result_field="result",
        extreme=Extreme.MINIMUM,
        tie_break_field="label",
    )

    with pytest.raises(UnresolvedTieError, match="no tie-break"):
        reduce_structured_table(no_tie_break)
    with pytest.raises(UnresolvedTieError, match="not unique"):
        reduce_structured_table(duplicate_tie_values)


@pytest.mark.parametrize(
    "specification",
    [
        TableReductionInput(
            rows=(),
            identity_field="name",
            metric_field="metric",
            result_field="result",
            extreme=Extreme.MAXIMUM,
        ),
        TableReductionInput(
            rows=({"name": "One", "metric": 1},),
            identity_field="name",
            metric_field="metric",
            result_field="result",
            extreme=Extreme.MAXIMUM,
        ),
        TableReductionInput(
            rows=(
                {"name": "One", "metric": 1, "result": "x"},
                {"name": "one", "metric": 2, "result": "y"},
            ),
            identity_field="name",
            metric_field="metric",
            result_field="result",
            extreme=Extreme.MAXIMUM,
        ),
    ],
)
def test_empty_incomplete_or_duplicate_identity_tables_fail(
    specification: TableReductionInput,
) -> None:
    with pytest.raises(TableIntegrityError):
        reduce_structured_table(specification)


def provider_for(
    specification: TableReductionInput,
) -> Callable[[Question, ReconciledTaskAnalysis], Awaitable[TableReductionInput]]:
    async def provide(
        question: Question,
        task_analysis: ReconciledTaskAnalysis,
    ) -> TableReductionInput:
        assert question.task_id == "structured-table-task"
        assert task_analysis.task_class is TaskClass.STRUCTURED_TABLE
        return specification

    return provide


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer_type", "specification", "expected"),
    [
        (AnswerType.INTEGER, yankees_like_input(), IntegerAnswer(value=427)),
        (AnswerType.IOC_CODE, olympics_like_input(), EntityAnswer(ioc_code="ALA")),
    ],
)
async def test_solver_returns_typed_answer_for_both_public_task_shapes(
    answer_type: AnswerType,
    specification: TableReductionInput,
    expected: IntegerAnswer | EntityAnswer,
) -> None:
    question = Question(task_id="structured-table-task", question="Synthetic question")

    result = await StructuredTableSolver(provider_for(specification)).solve(
        question,
        analysis(answer_type=answer_type),
        SolverRoute.STRUCTURED_TABLE.value,
    )

    assert result == expected


@pytest.mark.asyncio
async def test_registered_solver_dispatches_through_existing_registry() -> None:
    question = Question(task_id="structured-table-task", question="Synthetic question")
    registry = SolverRegistry()
    registered = register_structured_table_solver(registry, provider_for(yankees_like_input()))

    result = await registry.solve(SolverRoute.STRUCTURED_TABLE, question, analysis())

    assert isinstance(registered, StructuredTableSolver)
    assert result == IntegerAnswer(value=427)


@pytest.mark.asyncio
async def test_wrong_route_or_metadata_is_rejected_before_provider_call() -> None:
    question = Question(task_id="structured-table-task", question="Synthetic question")
    called = False

    async def provider(
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
    ) -> TableReductionInput:
        nonlocal called
        called = True
        return yankees_like_input()

    solver = StructuredTableSolver(provider)
    with pytest.raises(StructuredTableSolverError, match="wrong route"):
        await solver.solve(question, analysis(), SolverRoute.OPERATION_TABLE.value)
    with pytest.raises(StructuredTableSolverError, match="metadata"):
        await solver.solve(
            question,
            analysis(task_class=TaskClass.DEEP_RESEARCH),
            SolverRoute.STRUCTURED_TABLE.value,
        )

    assert called is False
