from __future__ import annotations

import io
from collections.abc import Awaitable, Callable
from dataclasses import replace
from decimal import Decimal

import pytest
from openpyxl import Workbook

from gaia_max.domain import (
    AnswerType,
    DecimalAnswer,
    Modality,
    OutputContract,
    Question,
    TaskClass,
    UnitsPolicy,
)
from gaia_max.output_contracts import ContractParseStatus
from gaia_max.serialization import AnswerSerializer
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.solvers.spreadsheet import (
    HiddenRowPolicy,
    RowDisposition,
    SpreadsheetCalculationError,
    SpreadsheetCalculationPlan,
    SpreadsheetCalculator,
    SpreadsheetPassResult,
    SpreadsheetSolver,
    WorkbookAudit,
    WorkbookInspector,
    decimal_from_spreadsheet,
    register_spreadsheet_solver,
)
from gaia_max.task_analysis import AnalysisAuthority, ReconciledTaskAnalysis


def sales_workbook_bytes(
    *,
    unknown_category: bool = False,
    unreviewed_total: bool = False,
    included_formula: bool = False,
) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    sheet.append(["Quarterly sales"])
    sheet.append(["Item", "Category", "Sales"])
    sheet.append(["Burger", "Food", 10.10])
    sheet.append(["Fries", "Food", "$2.20"])
    sheet.append(["Soda", "Drink", 5])
    sheet.append(["Secret Menu", "Food", 1])
    sheet.row_dimensions[6].hidden = True
    if unknown_category:
        sheet.append(["Mystery", "Unknown", 7])
    if unreviewed_total:
        sheet.append(["Food Total", "Food", 13.30])
    else:
        sheet.append(["Grand Total", "", "=SUM(C3:C6)"])
    if included_formula:
        sheet["C3"] = "=5+5.1"

    other = workbook.create_sheet("Notes")
    other.append(["This sheet proves pandas loads the full workbook"])
    output = io.BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


def plan(
    *,
    hidden_row_policy: HiddenRowPolicy = HiddenRowPolicy.INCLUDE,
) -> SpreadsheetCalculationPlan:
    return SpreadsheetCalculationPlan(
        sheet_name="Sales",
        header_row=2,
        item_column="Item",
        category_column="Category",
        sales_column="Sales",
        included_categories=("Food",),
        excluded_categories=("Drink",),
        hidden_row_policy=hidden_row_policy,
    )


def analysis(
    *,
    task_class: TaskClass = TaskClass.SPREADSHEET,
    modality: Modality = Modality.XLSX,
) -> ReconciledTaskAnalysis:
    return ReconciledTaskAnalysis(
        task_id="spreadsheet-task",
        modality=modality,
        task_class=task_class,
        requested_operation="sum food sales excluding drinks",
        output_contract=OutputContract(
            answer_type=AnswerType.CURRENCY,
            decimal_places=2,
            units_policy=UnitsPolicy.QUESTION_SPECIFIC,
            include_currency_symbol=False,
        ),
        risk_flags=(
            "food_drink_classification",
            "formula_cache",
            "hidden_or_total_rows",
            "currency_style",
        ),
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12, Decimal("12")),
        (12.3, Decimal("12.3")),
        ("$1,234.50", Decimal("1234.50")),
        ("(12.30)", Decimal("-12.30")),
        ("\u20ac7.25", Decimal("7.25")),
    ],
)
def test_currency_cells_convert_through_exact_decimal_text(
    value: object,
    expected: Decimal,
) -> None:
    assert decimal_from_spreadsheet(value, field_name="sales") == expected


@pytest.mark.parametrize(
    "value",
    [True, None, float("nan"), "", "12 dollars", "1,00", "(12.30", "12.30)"],
)
def test_invalid_currency_cells_fail(value: object) -> None:
    with pytest.raises(SpreadsheetCalculationError):
        decimal_from_spreadsheet(value, field_name="sales")


def test_independent_parsers_match_exact_rows_and_decimal_total() -> None:
    content = sales_workbook_bytes()
    audit = WorkbookInspector().inspect(content, original_name="sales.xlsx")

    result = SpreadsheetCalculator().calculate(content, plan(), audit)

    assert result.total == Decimal("13.3")
    assert result.openpyxl_pass.total == result.pandas_calamine_pass.total
    assert result.openpyxl_pass.rows == result.pandas_calamine_pass.rows
    assert result.openpyxl_pass.included_row_count == 3
    assert [row.disposition for row in result.openpyxl_pass.rows] == [
        RowDisposition.FOOD,
        RowDisposition.FOOD,
        RowDisposition.DRINK,
        RowDisposition.FOOD,
        RowDisposition.TOTAL,
    ]
    assert result.openpyxl_pass.rows[-1].sales is None


def test_total_row_is_excluded_instead_of_double_counted() -> None:
    content = sales_workbook_bytes()
    audit = WorkbookInspector().inspect(content)

    result = SpreadsheetCalculator().calculate(content, plan(), audit)

    assert result.total != Decimal("26.6")
    assert sum(
        row.disposition is RowDisposition.TOTAL for row in result.openpyxl_pass.rows
    ) == 1


def test_total_like_row_not_named_in_plan_blocks_instead_of_being_summed() -> None:
    content = sales_workbook_bytes(unreviewed_total=True)
    audit = WorkbookInspector().inspect(content)

    with pytest.raises(SpreadsheetCalculationError, match="unreviewed total"):
        SpreadsheetCalculator().calculate(content, plan(), audit)


def test_hidden_rows_require_policy_and_can_be_explicitly_excluded() -> None:
    content = sales_workbook_bytes()
    audit = WorkbookInspector().inspect(content)

    with pytest.raises(SpreadsheetCalculationError, match="hidden"):
        SpreadsheetCalculator().calculate(
            content,
            plan(hidden_row_policy=HiddenRowPolicy.ERROR),
            audit,
        )
    excluded = SpreadsheetCalculator().calculate(
        content,
        plan(hidden_row_policy=HiddenRowPolicy.EXCLUDE),
        audit,
    )

    assert excluded.total == Decimal("12.3")
    assert any(row.disposition is RowDisposition.HIDDEN for row in excluded.openpyxl_pass.rows)


def test_unknown_categories_and_uncached_included_formulas_fail() -> None:
    unknown = sales_workbook_bytes(unknown_category=True)
    unknown_audit = WorkbookInspector().inspect(unknown)
    formula = sales_workbook_bytes(included_formula=True)
    formula_audit = WorkbookInspector().inspect(formula)

    with pytest.raises(SpreadsheetCalculationError, match="unreviewed category"):
        SpreadsheetCalculator().calculate(unknown, plan(), unknown_audit)
    with pytest.raises(SpreadsheetCalculationError, match="currency value"):
        SpreadsheetCalculator().calculate(formula, plan(), formula_audit)


def test_mismatched_audit_or_parser_result_blocks_parity() -> None:
    content = sales_workbook_bytes()
    audit = WorkbookInspector().inspect(content)

    with pytest.raises(SpreadsheetCalculationError, match="do not match"):
        SpreadsheetCalculator().calculate(content + b"changed", plan(), audit)

    class DivergentCalculator(SpreadsheetCalculator):
        @staticmethod
        def _calculate_pandas_calamine(
            workbook_content: bytes,
            calculation_plan: SpreadsheetCalculationPlan,
            workbook_audit: WorkbookAudit,
        ) -> SpreadsheetPassResult:
            result = SpreadsheetCalculator._calculate_pandas_calamine(
                workbook_content,
                calculation_plan,
                workbook_audit,
            )
            return replace(result, total=result.total + Decimal("0.01"))

    with pytest.raises(SpreadsheetCalculationError, match="totals disagree"):
        DivergentCalculator().calculate(content, plan(), audit)


def providers(
    content: bytes,
) -> tuple[
    Callable[[Question, ReconciledTaskAnalysis], Awaitable[bytes]],
    Callable[
        [Question, ReconciledTaskAnalysis, WorkbookAudit],
        Awaitable[SpreadsheetCalculationPlan],
    ],
]:
    async def source(
        question: Question,
        task_analysis: ReconciledTaskAnalysis,
    ) -> bytes:
        assert question.file_name == "spreadsheet-task.xlsx"
        assert task_analysis.task_class is TaskClass.SPREADSHEET
        return content

    async def calculation_plan(
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
        audit: WorkbookAudit,
    ) -> SpreadsheetCalculationPlan:
        assert audit.sheet_names == ("Sales", "Notes")
        return plan()

    return source, calculation_plan


@pytest.mark.asyncio
async def test_solver_returns_typed_decimal_and_serializer_owns_currency_format() -> None:
    content = sales_workbook_bytes()
    source, calculation_plan = providers(content)
    question = Question(
        task_id="spreadsheet-task",
        question="What were total food sales in USD with two decimal places?",
        file_name="spreadsheet-task.xlsx",
    )

    result = await SpreadsheetSolver(source, calculation_plan).solve(
        question,
        analysis(),
        SolverRoute.SPREADSHEET.value,
    )

    assert result == DecimalAnswer(value=Decimal("13.3"))
    assert AnswerSerializer().serialize(result, analysis().output_contract) == "13.30"


@pytest.mark.asyncio
async def test_registered_solver_dispatches_through_existing_registry() -> None:
    content = sales_workbook_bytes()
    source, calculation_plan = providers(content)
    question = Question(
        task_id="spreadsheet-task",
        question="Calculate food sales.",
        file_name="spreadsheet-task.xlsx",
    )
    registry = SolverRegistry()
    registered = register_spreadsheet_solver(registry, source, calculation_plan)

    result = await registry.solve(SolverRoute.SPREADSHEET, question, analysis())

    assert isinstance(registered, SpreadsheetSolver)
    assert result == DecimalAnswer(value=Decimal("13.3"))


@pytest.mark.asyncio
async def test_wrong_route_or_metadata_fails_before_provider_access() -> None:
    called = False

    async def source(
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
    ) -> bytes:
        nonlocal called
        called = True
        return sales_workbook_bytes()

    async def calculation_plan(
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
        _audit: WorkbookAudit,
    ) -> SpreadsheetCalculationPlan:
        return plan()

    question = Question(
        task_id="spreadsheet-task",
        question="Calculate food sales.",
        file_name="spreadsheet-task.xlsx",
    )
    solver = SpreadsheetSolver(source, calculation_plan)

    with pytest.raises(SpreadsheetCalculationError, match="wrong route"):
        await solver.solve(question, analysis(), SolverRoute.CODE.value)
    with pytest.raises(SpreadsheetCalculationError, match="metadata"):
        await solver.solve(
            question,
            analysis(task_class=TaskClass.DEEP_RESEARCH, modality=Modality.WEB),
            SolverRoute.SPREADSHEET.value,
        )
    assert called is False
