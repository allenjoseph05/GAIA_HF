"""Deterministic min/max reduction over complete normalized table rows."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from gaia_max.domain import (
    AnswerType,
    EntityAnswer,
    IntegerAnswer,
    Modality,
    Question,
    TaskClass,
)
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.task_analysis import ReconciledTaskAnalysis


class StructuredTableSolverError(ValueError):
    """Base failure for invalid table data or incompatible solver input."""


class TableIntegrityError(StructuredTableSolverError):
    """The normalized table is incomplete, ambiguous, or internally invalid."""


class UnresolvedTieError(StructuredTableSolverError):
    """Multiple extreme rows remain without a deterministic tie-break."""


class Extreme(StrEnum):
    """Supported deterministic reduction directions."""

    MINIMUM = "minimum"
    MAXIMUM = "maximum"


TableRow = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class TableReductionInput:
    """Full normalized rows plus the exact reduction requested by the question."""

    rows: tuple[TableRow, ...]
    identity_field: str
    metric_field: str
    result_field: str
    extreme: Extreme
    tie_break_field: str | None = None


@dataclass(frozen=True, slots=True)
class TableReductionResult:
    """Selected same-row value and audit facts for later verification."""

    selected_identity: str
    selected_row_index: int
    metric_value: int
    result_value: object
    tie_count: int
    rows_examined: int


TableProvider = Callable[
    [Question, ReconciledTaskAnalysis],
    Awaitable[TableReductionInput],
]


_PLAIN_INTEGER = re.compile(r"[+-]?\d+")
_GROUPED_INTEGER = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+")


def coerce_table_integer(value: object, *, field: str) -> int:
    """Coerce an integer cell while rejecting booleans, floats, and bad grouping."""

    if isinstance(value, bool):
        raise TableIntegrityError(f"field {field!r} contains a boolean, not an integer")
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise TableIntegrityError(f"field {field!r} is not an integer cell")
    text = value.strip()
    if not (_PLAIN_INTEGER.fullmatch(text) or _GROUPED_INTEGER.fullmatch(text)):
        raise TableIntegrityError(f"field {field!r} is not a valid integer literal")
    return int(text.replace(",", ""))


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TableIntegrityError(f"field {field!r} must contain normalized non-empty text")
    return value


def reduce_structured_table(specification: TableReductionInput) -> TableReductionResult:
    """Inspect every row, choose the extreme, resolve ties, and return that row's value."""

    if not specification.rows:
        raise TableIntegrityError("structured table cannot be empty")
    if not isinstance(specification.extreme, Extreme):
        raise TableIntegrityError("structured table has an unsupported extreme direction")
    fields = {
        specification.identity_field,
        specification.metric_field,
        specification.result_field,
    }
    if specification.tie_break_field is not None:
        fields.add(specification.tie_break_field)
    if any(not field for field in fields):
        raise TableIntegrityError("table field names cannot be empty")

    identities: list[str] = []
    metrics: list[int] = []
    seen_identities: set[str] = set()
    for index, row in enumerate(specification.rows):
        missing = fields.difference(row)
        if missing:
            raise TableIntegrityError(
                f"row {index} is missing required fields: {', '.join(sorted(missing))}"
            )
        if row[specification.result_field] is None:
            raise TableIntegrityError(
                f"row {index} has no value for result field {specification.result_field!r}"
            )
        identity = _required_text(
            row[specification.identity_field],
            field=specification.identity_field,
        )
        identity_key = identity.casefold()
        if identity_key in seen_identities:
            raise TableIntegrityError("table identity values must be unique")
        seen_identities.add(identity_key)
        identities.append(identity)
        metrics.append(
            coerce_table_integer(
                row[specification.metric_field],
                field=specification.metric_field,
            )
        )

    extreme_value = (
        min(metrics) if specification.extreme is Extreme.MINIMUM else max(metrics)
    )
    candidates = [index for index, value in enumerate(metrics) if value == extreme_value]
    if len(candidates) > 1:
        tie_field = specification.tie_break_field
        if tie_field is None:
            raise UnresolvedTieError(
                "multiple rows share the extreme and no tie-break field was supplied"
            )
        tie_values = {
            index: _required_text(specification.rows[index][tie_field], field=tie_field)
            for index in candidates
        }
        folded = [tie_values[index].casefold() for index in candidates]
        if len(set(folded)) != len(folded):
            raise UnresolvedTieError("tie-break values are not unique case-insensitively")
        selected_index = min(
            candidates,
            key=lambda index: (tie_values[index].casefold(), tie_values[index]),
        )
    else:
        selected_index = candidates[0]

    return TableReductionResult(
        selected_identity=identities[selected_index],
        selected_row_index=selected_index,
        metric_value=extreme_value,
        result_value=specification.rows[selected_index][specification.result_field],
        tie_count=len(candidates),
        rows_examined=len(specification.rows),
    )


class StructuredTableSolver:
    """Obtain normalized rows from an injected source and reduce them deterministically."""

    def __init__(self, table_provider: TableProvider) -> None:
        self._table_provider = table_provider

    async def solve(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        route: str,
    ) -> object:
        if route != SolverRoute.STRUCTURED_TABLE.value:
            raise StructuredTableSolverError("structured-table solver received wrong route")
        if (
            analysis.task_class is not TaskClass.STRUCTURED_TABLE
            or analysis.modality not in {Modality.WEB, Modality.TEXT}
            or analysis.output_contract.answer_type
            not in {AnswerType.INTEGER, AnswerType.IOC_CODE}
            or question.file_name is not None
        ):
            raise StructuredTableSolverError(
                "structured-table solver received incompatible task metadata"
            )

        specification = await self._table_provider(question, analysis)
        if not isinstance(specification, TableReductionInput):
            raise StructuredTableSolverError(
                "table provider did not return a TableReductionInput"
            )
        result = reduce_structured_table(specification)
        if result.rows_examined != len(specification.rows):
            raise StructuredTableSolverError("structured-table reduction skipped rows")

        if analysis.output_contract.answer_type is AnswerType.INTEGER:
            return IntegerAnswer(
                value=coerce_table_integer(
                    result.result_value,
                    field=specification.result_field,
                )
            )
        code = _required_text(result.result_value, field=specification.result_field)
        return EntityAnswer(ioc_code=code)


def register_structured_table_solver(
    registry: SolverRegistry,
    table_provider: TableProvider,
) -> StructuredTableSolver:
    """Register B03 explicitly with its required table-source dependency."""

    instance = StructuredTableSolver(table_provider)
    registry.register(SolverRoute.STRUCTURED_TABLE, instance)
    return instance
