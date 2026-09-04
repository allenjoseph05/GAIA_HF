"""Complete Markdown operation-table parsing and exhaustive comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass

from gaia_max.domain import AnswerType, ListAnswer, Modality, Question, TaskClass
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.task_analysis import ReconciledTaskAnalysis


class OperationTableSolverError(ValueError):
    """Base failure for an invalid or incompatible operation-table task."""


class OperationTableParseError(OperationTableSolverError):
    """The question does not contain one complete, closed operation table."""


_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
_DECLARED_SET = re.compile(
    r"\bset\s+[A-Za-z][A-Za-z0-9_]*\s*=\s*\{(?P<elements>[^{}]+)\}",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class OperationTable:
    """A closed binary operation whose rows are normalized to header order."""

    operator: str
    elements: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def apply(self, left: str, right: str) -> str:
        """Look up one operation result, failing for elements outside the table."""

        try:
            row = self.elements.index(left)
            column = self.elements.index(right)
        except ValueError as exc:
            raise OperationTableSolverError("operation lookup used an unknown element") from exc
        return self.rows[row][column]


@dataclass(frozen=True, slots=True)
class NoncommutativityResult:
    """Elements and audit facts produced by an exhaustive ordered-pair pass."""

    elements: tuple[str, ...]
    counterexample_pairs: tuple[tuple[str, str], ...]
    comparisons_checked: int


def _markdown_cells(line: str) -> tuple[str, ...] | None:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    return tuple(cell.strip() for cell in stripped[1:-1].split("|"))


def _table_blocks(text: str) -> list[list[tuple[str, ...]]]:
    blocks: list[list[tuple[str, ...]]] = []
    current: list[tuple[str, ...]] = []
    for line in text.splitlines():
        cells = _markdown_cells(line)
        if cells is not None:
            current.append(cells)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return [
        block
        for block in blocks
        if len(block) >= 2
        and all(_SEPARATOR_CELL.fullmatch(cell) for cell in block[1])
    ]


def parse_operation_table(question_text: str) -> OperationTable:
    """Parse and validate exactly one pipe-bounded Markdown operation table."""

    blocks = _table_blocks(question_text)
    if len(blocks) != 1:
        raise OperationTableParseError(
            "question must contain exactly one Markdown operation table"
        )

    header, separator, *data_rows = blocks[0]
    if len(header) < 2 or any(not cell for cell in header):
        raise OperationTableParseError("operation-table header contains an empty cell")
    if len(separator) != len(header):
        raise OperationTableParseError("operation-table separator width does not match header")

    operator = header[0]
    elements = header[1:]
    if len(set(elements)) != len(elements):
        raise OperationTableParseError("operation-table header elements must be unique")

    declared_match = _DECLARED_SET.search(question_text)
    if declared_match is not None:
        declared = tuple(
            item.strip() for item in declared_match.group("elements").split(",")
        )
        if any(not item for item in declared) or len(set(declared)) != len(declared):
            raise OperationTableParseError("declared set contains empty or duplicate elements")
        if set(declared) != set(elements):
            raise OperationTableParseError("declared set does not match table header")

    rows_by_element: dict[str, tuple[str, ...]] = {}
    for row in data_rows:
        if len(row) != len(header):
            raise OperationTableParseError("operation-table row width does not match header")
        label, *values = row
        if not label:
            raise OperationTableParseError("operation-table row label cannot be empty")
        if label in rows_by_element:
            raise OperationTableParseError("operation-table row labels must be unique")
        if any(not value for value in values):
            raise OperationTableParseError("operation-table result cells cannot be empty")
        rows_by_element[label] = tuple(values)

    if set(rows_by_element) != set(elements):
        raise OperationTableParseError(
            "operation-table rows must contain every header element exactly once"
        )
    allowed = set(elements)
    if any(value not in allowed for row in rows_by_element.values() for value in row):
        raise OperationTableParseError("operation-table result lies outside the declared set")

    return OperationTable(
        operator=operator,
        elements=elements,
        rows=tuple(rows_by_element[element] for element in elements),
    )


def analyze_noncommutativity(table: OperationTable) -> NoncommutativityResult:
    """Compare every ordered pair and collect all elements in unequal reversals."""

    involved: set[str] = set()
    counterexamples: list[tuple[str, str]] = []
    comparisons_checked = 0
    for left in table.elements:
        for right in table.elements:
            comparisons_checked += 1
            if table.apply(left, right) != table.apply(right, left):
                involved.update((left, right))
                counterexamples.append((left, right))
    return NoncommutativityResult(
        elements=tuple(element for element in table.elements if element in involved),
        counterexample_pairs=tuple(counterexamples),
        comparisons_checked=comparisons_checked,
    )


class OperationTableSolver:
    """Solve the reviewed noncommutativity task without model inference."""

    async def solve(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        route: str,
    ) -> object:
        if route != SolverRoute.OPERATION_TABLE.value:
            raise OperationTableSolverError("operation-table solver received wrong route")
        if (
            analysis.task_class is not TaskClass.OPERATION_TABLE
            or analysis.modality is not Modality.TEXT
            or analysis.output_contract.answer_type is not AnswerType.LIST
            or question.file_name is not None
        ):
            raise OperationTableSolverError(
                "operation-table solver received incompatible task metadata"
            )

        table = parse_operation_table(question.question)
        result = analyze_noncommutativity(table)
        expected_comparisons = len(table.elements) ** 2
        if result.comparisons_checked != expected_comparisons:
            raise OperationTableSolverError("operation-table comparison was not exhaustive")
        if not result.elements:
            raise OperationTableSolverError(
                "table has no noncommutativity counterexamples for a non-empty list answer"
            )
        return ListAnswer(items=list(result.elements))


def register_operation_table_solver(
    registry: SolverRegistry,
    solver: OperationTableSolver | None = None,
) -> OperationTableSolver:
    """Register B02 explicitly and return the concrete solver instance."""

    instance = solver or OperationTableSolver()
    registry.register(SolverRoute.OPERATION_TABLE, instance)
    return instance
