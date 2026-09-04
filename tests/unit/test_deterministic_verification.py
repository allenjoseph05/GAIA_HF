from __future__ import annotations

import io
from collections.abc import Awaitable, Callable
from typing import cast

import pytest
from openpyxl import Workbook

from gaia_max.deterministic_verification import (
    DeterministicVerificationRegistry,
    DeterministicVerificationRegistryError,
)
from gaia_max.domain import (
    AnswerType,
    IntegerAnswer,
    Modality,
    OutputContract,
    Question,
    RiskLevel,
    SemanticAnswer,
    SortPolicy,
    StringAnswer,
    TaskClass,
    TaskProfile,
    TaskStatus,
    Verdict,
)
from gaia_max.output_contracts import ContractParseStatus
from gaia_max.snapshot import question_sha256
from gaia_max.solver_routing import SolverRegistry, SolverRoute, SpecialistSolver
from gaia_max.solvers import (
    Extreme,
    HiddenRowPolicy,
    OperationTableSolver,
    PythonCodeSolver,
    SpreadsheetCalculationPlan,
    SpreadsheetSolver,
    StructuredTableSolver,
    TableReductionInput,
    TransformedTextSolver,
    reverse_exact,
)
from gaia_max.task_analysis import AnalysisAuthority, ReconciledTaskAnalysis
from gaia_max.task_graph import TaskGraphInput, build_task_graph


def analysis(
    task_class: TaskClass,
    modality: Modality,
    answer_type: AnswerType,
    *,
    sort: SortPolicy = SortPolicy.NONE,
) -> ReconciledTaskAnalysis:
    return ReconciledTaskAnalysis(
        task_id=f"{task_class.value}-verification-task",
        modality=modality,
        task_class=task_class,
        requested_operation="synthetic deterministic operation",
        output_contract=OutputContract(answer_type=answer_type, sort=sort),
        risk_flags=(),
        field_sources={
            field: AnalysisAuthority.PROFILE
            for field in (
                "modality",
                "task_class",
                "requested_operation",
                "temporal_constraint",
                "filters",
                "output_contract",
                "risk_flags",
            )
        },
        contract_parse_status=ContractParseStatus.COMPLETE,
        profile_used=True,
        model_used=False,
        requires_review=False,
    )


def byte_provider(
    content: bytes,
) -> Callable[[Question, ReconciledTaskAnalysis], Awaitable[bytes]]:
    async def provide(
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
    ) -> bytes:
        return content

    return provide


def sales_workbook() -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    sheet.append(("Item", "Category", "Sales"))
    sheet.append(("Soup", "Food", "10.25"))
    sheet.append(("Tea", "Drink", "2.50"))
    sheet.append(("Total", "", "12.75"))
    output = io.BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


@pytest.mark.asyncio
async def test_every_milestone_b_specialist_is_approved_by_its_authority() -> None:
    decoded = 'Write the opposite of the word "north" as the answer.'
    transformed_question = Question(
        task_id="deterministic_text-verification-task",
        question=reverse_exact(decoded),
    )
    transformed_analysis = analysis(
        TaskClass.DETERMINISTIC_TEXT,
        Modality.TEXT,
        AnswerType.STRING,
    )

    table_text = """Find the noncommuting elements in this table.
|*|a|b|
|---|---|---|
|a|a|a|
|b|b|a|
"""
    operation_question = Question(
        task_id="operation_table-verification-task",
        question=table_text,
    )
    operation_analysis = analysis(
        TaskClass.OPERATION_TABLE,
        Modality.TEXT,
        AnswerType.LIST,
        sort=SortPolicy.ALPHABETICAL,
    )

    table_input = TableReductionInput(
        rows=(
            {"player": "Ada", "walks": 12, "at_bats": 100},
            {"player": "Ben", "walks": 20, "at_bats": 150},
        ),
        identity_field="player",
        metric_field="walks",
        result_field="at_bats",
        extreme=Extreme.MAXIMUM,
    )

    async def table_provider(
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
    ) -> TableReductionInput:
        return table_input

    structured_question = Question(
        task_id="structured_table-verification-task",
        question="Which row has the maximum?",
    )
    structured_analysis = analysis(
        TaskClass.STRUCTURED_TABLE,
        Modality.WEB,
        AnswerType.INTEGER,
    )

    code_question = Question(
        task_id="code-verification-task",
        question="What is the final numeric output?",
        file_name="program.py",
    )
    code_analysis = analysis(TaskClass.CODE, Modality.PYTHON, AnswerType.STRING)
    code_solver = PythonCodeSolver(byte_provider(b"value = 6 * 7\nprint(value)\n"))

    workbook_content = sales_workbook()

    async def plan_provider(
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
        _audit: object,
    ) -> SpreadsheetCalculationPlan:
        return SpreadsheetCalculationPlan(
            sheet_name="Sales",
            header_row=1,
            item_column="Item",
            category_column="Category",
            sales_column="Sales",
            included_categories=("Food",),
            excluded_categories=("Drink",),
            hidden_row_policy=HiddenRowPolicy.INCLUDE,
        )

    spreadsheet_question = Question(
        task_id="spreadsheet-verification-task",
        question="What are the food sales?",
        file_name="sales.xlsx",
    )
    spreadsheet_analysis = analysis(
        TaskClass.SPREADSHEET,
        Modality.XLSX,
        AnswerType.CURRENCY,
    )
    spreadsheet_solver = SpreadsheetSolver(
        byte_provider(workbook_content),
        plan_provider,
    )

    cases: tuple[
        tuple[
            SolverRoute,
            SpecialistSolver,
            Question,
            ReconciledTaskAnalysis,
        ],
        ...,
    ] = (
        (
            SolverRoute.DETERMINISTIC_TEXT,
            TransformedTextSolver(),
            transformed_question,
            transformed_analysis,
        ),
        (
            SolverRoute.OPERATION_TABLE,
            OperationTableSolver(),
            operation_question,
            operation_analysis,
        ),
        (
            SolverRoute.STRUCTURED_TABLE,
            StructuredTableSolver(table_provider),
            structured_question,
            structured_analysis,
        ),
        (SolverRoute.CODE, code_solver, code_question, code_analysis),
        (
            SolverRoute.SPREADSHEET,
            spreadsheet_solver,
            spreadsheet_question,
            spreadsheet_analysis,
        ),
    )

    for route, solver, question, task_analysis in cases:
        candidate = cast(
            SemanticAnswer,
            await solver.solve(question, task_analysis, route.value),
        )
        verifier = DeterministicVerificationRegistry()
        verifier.register(route, solver)

        result = await verifier.verify(question, task_analysis, candidate)

        assert result.verdict is Verdict.APPROVE, route
        assert result.tier == 1
        assert result.passed_checks[-1] == "candidate_matches_authority"
        assert result.reason_codes == []


@pytest.mark.asyncio
async def test_authority_mismatch_rejects_candidate_without_model_override() -> None:
    decoded = 'Write the opposite of the word "up" as the answer.'
    question = Question(
        task_id="deterministic_text-verification-task",
        question=reverse_exact(decoded),
    )
    task_analysis = analysis(
        TaskClass.DETERMINISTIC_TEXT,
        Modality.TEXT,
        AnswerType.STRING,
    )
    verifier = DeterministicVerificationRegistry()
    verifier.register(SolverRoute.DETERMINISTIC_TEXT, TransformedTextSolver())

    result = await verifier.verify(question, task_analysis, StringAnswer(value="up"))

    assert result.verdict is Verdict.REJECT
    assert result.reason_codes == ["candidate_authority_mismatch"]
    assert "candidate_matches_authority" not in result.passed_checks


@pytest.mark.asyncio
async def test_failed_or_missing_authority_is_uncertain_and_never_approves() -> None:
    malformed = Question(
        task_id="operation_table-verification-task",
        question="No complete operation table.",
    )
    task_analysis = analysis(
        TaskClass.OPERATION_TABLE,
        Modality.TEXT,
        AnswerType.LIST,
    )
    verifier = DeterministicVerificationRegistry()
    verifier.register(SolverRoute.OPERATION_TABLE, OperationTableSolver())

    failed = await verifier.verify(malformed, task_analysis, StringAnswer(value="guess"))
    missing = await DeterministicVerificationRegistry().verify(
        malformed,
        task_analysis,
        StringAnswer(value="guess"),
    )

    assert failed.verdict is Verdict.UNCERTAIN
    assert failed.reason_codes == ["deterministic_authority_failed"]
    assert missing.verdict is Verdict.UNCERTAIN
    assert missing.reason_codes == ["deterministic_policy_unavailable"]


def test_registry_rejects_duplicate_and_non_milestone_b_routes() -> None:
    verifier = DeterministicVerificationRegistry()
    solver = TransformedTextSolver()
    verifier.register(SolverRoute.DETERMINISTIC_TEXT, solver)

    with pytest.raises(DeterministicVerificationRegistryError, match="already"):
        verifier.register(SolverRoute.DETERMINISTIC_TEXT, solver)
    with pytest.raises(DeterministicVerificationRegistryError, match="no Milestone B"):
        verifier.register(SolverRoute.CHESS, solver)


@pytest.mark.asyncio
async def test_task_graph_uses_deterministic_registry_before_serialization() -> None:
    decoded = 'Write the opposite of the word "east" as the answer.'
    question = Question(task_id="graph-transform", question=reverse_exact(decoded))
    profile = TaskProfile(
        task_id=question.task_id,
        question=question.question,
        question_sha256=question_sha256(question),
        modality=Modality.TEXT,
        task_class=TaskClass.DETERMINISTIC_TEXT,
        route=SolverRoute.DETERMINISTIC_TEXT.value,
        risk_level=RiskLevel.LOW,
        requested_operation="exact string transform",
        required_sources=["question_text"],
        output_contract=OutputContract(answer_type=AnswerType.STRING),
        verification_policy="deterministic_replay",
    )
    solver = TransformedTextSolver()
    solvers = SolverRegistry()
    solvers.register(SolverRoute.DETERMINISTIC_TEXT, solver)
    verifier = DeterministicVerificationRegistry()
    verifier.register(SolverRoute.DETERMINISTIC_TEXT, solver)
    graph = build_task_graph(solver_registry=solvers, verifier=verifier)

    result = await graph.ainvoke(TaskGraphInput(question=question, profile=profile).initial_state())

    assert result["status"] is TaskStatus.READY
    assert result["semantic_answer"] == StringAnswer(value="west")
    assert result["verification"].verdict is Verdict.APPROVE
    assert result["verification"].tier == 1
    assert result["verification"].passed_checks[-1] == "candidate_matches_authority"
    assert result["serialized_answer"] == "west"


@pytest.mark.asyncio
async def test_unexpected_authority_errors_remain_retryable_graph_failures() -> None:
    class BrokenSolver:
        async def solve(
            self,
            _question: Question,
            _analysis: ReconciledTaskAnalysis,
            _route: str,
        ) -> object:
            raise RuntimeError("temporary provider outage")

    verifier = DeterministicVerificationRegistry()
    verifier.register(SolverRoute.STRUCTURED_TABLE, BrokenSolver())
    task_analysis = analysis(
        TaskClass.STRUCTURED_TABLE,
        Modality.WEB,
        AnswerType.INTEGER,
    )

    with pytest.raises(RuntimeError, match="temporary provider outage"):
        await verifier.verify(
            Question(task_id=task_analysis.task_id, question="Synthetic"),
            task_analysis,
            IntegerAnswer(value=1),
        )
