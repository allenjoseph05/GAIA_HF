from __future__ import annotations

import pytest

from gaia_max.domain import (
    AnswerType,
    ListAnswer,
    Modality,
    OutputContract,
    Question,
    SortPolicy,
    TaskClass,
)
from gaia_max.output_contracts import ContractParseStatus
from gaia_max.serialization import AnswerSerializer
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.solvers.operation_table import (
    OperationTableParseError,
    OperationTableSolver,
    OperationTableSolverError,
    analyze_noncommutativity,
    parse_operation_table,
    register_operation_table_solver,
)
from gaia_max.task_analysis import AnalysisAuthority, ReconciledTaskAnalysis

NONCOMMUTATIVE_TABLE = """Given this table defining * on the set S = {z, a, m}

|*|z|a|m|
|---|---|---|---|
|m|z|z|z|
|z|z|z|z|
|a|a|z|z|

Provide the subset involved in counter-examples as a comma separated list in
alphabetical order.
"""


def analysis(
    *,
    task_class: TaskClass = TaskClass.OPERATION_TABLE,
    modality: Modality = Modality.TEXT,
    answer_type: AnswerType = AnswerType.LIST,
) -> ReconciledTaskAnalysis:
    return ReconciledTaskAnalysis(
        task_id="operation-table-task",
        modality=modality,
        task_class=task_class,
        requested_operation="collect noncommutativity counterexamples",
        output_contract=OutputContract(
            answer_type=answer_type,
            sort=SortPolicy.ALPHABETICAL,
            delimiter=", ",
        ),
        risk_flags=("exhaustive_pairs",),
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


def test_parser_normalizes_rows_to_header_order_and_validates_declared_set() -> None:
    table = parse_operation_table(NONCOMMUTATIVE_TABLE)

    assert table.operator == "*"
    assert table.elements == ("z", "a", "m")
    assert table.rows == (("z", "z", "z"), ("a", "z", "z"), ("z", "z", "z"))
    assert table.apply("a", "z") == "a"


def test_analysis_checks_all_ordered_pairs_and_records_each_counterexample() -> None:
    result = analyze_noncommutativity(parse_operation_table(NONCOMMUTATIVE_TABLE))

    assert result.comparisons_checked == 3**2
    assert result.counterexample_pairs == (("z", "a"), ("a", "z"))
    assert result.elements == ("z", "a")


def test_commutative_table_still_checks_all_ordered_pairs() -> None:
    table = parse_operation_table(
        """|+|0|1|
|---|---|---|
|0|0|1|
|1|1|0|
"""
    )

    result = analyze_noncommutativity(table)

    assert result.elements == ()
    assert result.counterexample_pairs == ()
    assert result.comparisons_checked == 4


@pytest.mark.parametrize(
    "table_text",
    [
        "No Markdown table here.",
        "|*|a|a|\n|---|---|---|\n|a|a|a|",
        "|*|a|b|\n|---|---|---|\n|a|a|b|",
        "|*|a|b|\n|---|---|---|\n|a|a|b|\n|a|b|a|",
        "|*|a|b|\n|---|---|---|\n|a|a|b|\n|b|a|",
        "|*|a|b|\n|---|---|---|\n|a|a|b|\n|b|b|x|",
        "On set S = {a, c}\n|*|a|b|\n|---|---|---|\n|a|a|b|\n|b|b|a|",
    ],
)
def test_malformed_or_incomplete_tables_fail_loudly(table_text: str) -> None:
    with pytest.raises(OperationTableParseError):
        parse_operation_table(table_text)


@pytest.mark.asyncio
async def test_solver_returns_typed_semantics_and_serializer_owns_formatting() -> None:
    question = Question(task_id="operation-table-task", question=NONCOMMUTATIVE_TABLE)
    result = await OperationTableSolver().solve(
        question,
        analysis(),
        SolverRoute.OPERATION_TABLE.value,
    )

    assert result == ListAnswer(items=["z", "a"])
    assert AnswerSerializer().serialize(result, analysis().output_contract) == "a, z"


@pytest.mark.asyncio
async def test_registered_solver_dispatches_through_existing_registry() -> None:
    question = Question(task_id="operation-table-task", question=NONCOMMUTATIVE_TABLE)
    registry = SolverRegistry()
    registered = register_operation_table_solver(registry)

    result = await registry.solve(SolverRoute.OPERATION_TABLE, question, analysis())

    assert isinstance(registered, OperationTableSolver)
    assert result == ListAnswer(items=["z", "a"])


@pytest.mark.asyncio
async def test_wrong_route_or_metadata_is_rejected() -> None:
    question = Question(task_id="operation-table-task", question=NONCOMMUTATIVE_TABLE)
    solver = OperationTableSolver()

    with pytest.raises(OperationTableSolverError, match="wrong route"):
        await solver.solve(question, analysis(), SolverRoute.STRUCTURED_TABLE.value)
    with pytest.raises(OperationTableSolverError, match="metadata"):
        await solver.solve(
            question,
            analysis(task_class=TaskClass.STRUCTURED_TABLE, modality=Modality.WEB),
            SolverRoute.OPERATION_TABLE.value,
        )
    with pytest.raises(OperationTableSolverError, match="metadata"):
        await solver.solve(
            question,
            analysis(answer_type=AnswerType.STRING),
            SolverRoute.OPERATION_TABLE.value,
        )
