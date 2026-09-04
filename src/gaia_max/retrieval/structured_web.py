"""Defensive HTML-table normalization for deterministic web reductions."""

from __future__ import annotations

from html.parser import HTMLParser

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictStr, field_validator

from gaia_max.domain.models import SHA256_PATTERN
from gaia_max.solvers.structured_table import (
    Extreme,
    TableReductionInput,
    TableReductionResult,
    reduce_structured_table,
)


class StructuredWebTableError(ValueError):
    """HTML table selection or normalization is incomplete or ambiguous."""


class StructuredWebTable(BaseModel):
    """One rectangular table tied to immutable source provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_artifact_id: str = Field(pattern=SHA256_PATTERN)
    table_index: int = Field(ge=0)
    caption: StrictStr | None = None
    headers: tuple[str, ...] = Field(min_length=1)
    rows: tuple[tuple[str, ...], ...] = Field(min_length=1)
    snapshot_at: AwareDatetime | None = None

    @field_validator("headers")
    @classmethod
    def unique_headers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not header for header in value):
            raise ValueError("structured web table headers cannot be empty")
        if len({header.casefold() for header in value}) != len(value):
            raise ValueError("structured web table headers must be unique")
        return value

    @field_validator("rows")
    @classmethod
    def rectangular_rows(cls, value: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
        widths = {len(row) for row in value}
        if len(widths) != 1:
            raise ValueError("structured web table must be rectangular")
        return value

    def records(self) -> tuple[dict[str, str], ...]:
        if any(len(row) != len(self.headers) for row in self.rows):
            raise StructuredWebTableError("table row width does not match headers")
        return tuple(dict(zip(self.headers, row, strict=True)) for row in self.rows)

    def unique_record(self, criteria: dict[str, str]) -> dict[str, str]:
        if not criteria or not set(criteria).issubset(self.headers):
            raise StructuredWebTableError("unique table criteria are empty or unknown")
        matches = [
            row
            for row in self.records()
            if all(
                row[field].casefold() == expected.casefold()
                for field, expected in criteria.items()
            )
        ]
        if len(matches) != 1:
            raise StructuredWebTableError("table criteria must select exactly one row")
        return matches[0]

    def reduce(
        self,
        *,
        identity_field: str,
        metric_field: str,
        result_field: str,
        extreme: Extreme,
        tie_break_field: str | None = None,
    ) -> TableReductionResult:
        return reduce_structured_table(
            TableReductionInput(
                rows=self.records(),
                identity_field=identity_field,
                metric_field=metric_field,
                result_field=result_field,
                extreme=extreme,
                tie_break_field=tie_break_field,
            )
        )


class _Cell:
    def __init__(self, text: str, rowspan: int, colspan: int, header: bool) -> None:
        self.text = text
        self.rowspan = rowspan
        self.colspan = colspan
        self.header = header


class _RawTable:
    def __init__(self) -> None:
        self.caption = ""
        self.rows: list[list[_Cell]] = []


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[_RawTable] = []
        self._depth = 0
        self._table: _RawTable | None = None
        self._row: list[_Cell] | None = None
        self._cell_parts: list[str] | None = None
        self._cell_attrs: tuple[int, int, bool] | None = None
        self._caption_parts: list[str] | None = None
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag in {"script", "style"}:
            self._ignored_depth += 1
            return
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._table = _RawTable()
            return
        if self._depth != 1 or self._ignored_depth:
            return
        if tag == "caption":
            self._caption_parts = []
        elif tag == "tr":
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            values = dict(attrs)
            rowspan = _span(values.get("rowspan"))
            colspan = _span(values.get("colspan"))
            self._cell_parts = []
            self._cell_attrs = (rowspan, colspan, tag == "th")
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if tag == "table":
            if self._depth == 1 and self._table is not None:
                self.tables.append(self._table)
                self._table = None
            self._depth = max(0, self._depth - 1)
            return
        if self._depth != 1:
            return
        if tag in {"th", "td"} and self._cell_parts is not None and self._row is not None:
            if self._cell_attrs is None:
                raise RuntimeError("missing HTML table cell attributes")
            rowspan, colspan, header = self._cell_attrs
            self._row.append(_Cell(_text(self._cell_parts), rowspan, colspan, header))
            self._cell_parts = None
            self._cell_attrs = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.rows.append(self._row)
            self._row = None
        elif tag == "caption" and self._caption_parts is not None and self._table is not None:
            self._table.caption = _text(self._caption_parts)
            self._caption_parts = None

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._cell_parts is not None:
            self._cell_parts.append(data)
        if self._caption_parts is not None:
            self._caption_parts.append(data)


def parse_html_tables(
    html: str,
    *,
    source_artifact_id: str,
    snapshot_at: object | None = None,
) -> tuple[StructuredWebTable, ...]:
    """Parse top-level HTML tables, expanding row/column spans deterministically."""

    parser = _TableParser()
    parser.feed(html)
    parsed: list[StructuredWebTable] = []
    for table_index, raw in enumerate(parser.tables):
        rows, header_flags = _expand_rows(raw.rows)
        if len(rows) < 2:
            continue
        header_indexes = [index for index, flags in enumerate(header_flags) if any(flags)]
        if not header_indexes:
            continue
        header_index = header_indexes[-1]
        if any(index > header_index for index in header_indexes):
            raise StructuredWebTableError("HTML table has header rows inside data")
        headers = rows[header_index]
        data_rows = tuple(rows[header_index + 1 :])
        if not data_rows:
            continue
        payload: dict[str, object] = {
            "source_artifact_id": source_artifact_id,
            "table_index": table_index,
            "caption": raw.caption or None,
            "headers": headers,
            "rows": data_rows,
        }
        if snapshot_at is not None:
            payload["snapshot_at"] = snapshot_at
        parsed.append(StructuredWebTable.model_validate(payload))
    return tuple(parsed)


def select_html_table(
    tables: tuple[StructuredWebTable, ...],
    *,
    required_headers: tuple[str, ...],
) -> StructuredWebTable:
    """Select exactly one table containing every requested header."""

    required = {header.casefold() for header in required_headers}
    if not required:
        raise ValueError("required table headers cannot be empty")
    matches = [
        table
        for table in tables
        if required.issubset({header.casefold() for header in table.headers})
    ]
    if len(matches) != 1:
        raise StructuredWebTableError("required headers must identify exactly one table")
    return matches[0]


def _expand_rows(
    raw_rows: list[list[_Cell]],
) -> tuple[list[tuple[str, ...]], list[tuple[bool, ...]]]:
    rows: list[tuple[str, ...]] = []
    flags: list[tuple[bool, ...]] = []
    pending: dict[int, tuple[int, str, bool]] = {}
    for raw_row in raw_rows:
        values: list[str] = []
        header_values: list[bool] = []
        column = 0

        for cell in raw_row:
            column = _fill_pending(values, header_values, pending, column)
            for _offset in range(cell.colspan):
                values.append(cell.text)
                header_values.append(cell.header)
                if cell.rowspan > 1:
                    pending[column] = (cell.rowspan - 1, cell.text, cell.header)
                column += 1
        column = _fill_pending(values, header_values, pending, column)
        rows.append(tuple(values))
        flags.append(tuple(header_values))
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise StructuredWebTableError("expanded HTML table is not rectangular")
    return rows, flags


def _fill_pending(
    values: list[str],
    header_values: list[bool],
    pending: dict[int, tuple[int, str, bool]],
    column: int,
) -> int:
    while column in pending:
        remaining, text, header = pending[column]
        values.append(text)
        header_values.append(header)
        if remaining <= 1:
            del pending[column]
        else:
            pending[column] = (remaining - 1, text, header)
        column += 1
    return column


def _span(value: str | None) -> int:
    if value is None:
        return 1
    try:
        parsed = int(value)
    except ValueError as exc:
        raise StructuredWebTableError("HTML table span is not an integer") from exc
    if not 1 <= parsed <= 100:
        raise StructuredWebTableError("HTML table span is outside safe bounds")
    return parsed


def _text(parts: list[str]) -> str:
    return " ".join("".join(parts).split())


__all__ = [
    "StructuredWebTable",
    "StructuredWebTableError",
    "parse_html_tables",
    "select_html_table",
]
