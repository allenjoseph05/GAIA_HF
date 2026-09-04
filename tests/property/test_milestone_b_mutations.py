"""Generative oracles that kill common deterministic-solver mutations."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from string import ascii_letters

import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from gaia_max.domain import AnswerType, ListAnswer, OutputContract, SortPolicy
from gaia_max.serialization import serialize_answer
from gaia_max.solvers.operation_table import (
    OperationTableParseError,
    analyze_noncommutativity,
    parse_operation_table,
)
from gaia_max.solvers.spreadsheet import decimal_from_spreadsheet
from gaia_max.solvers.structured_table import (
    Extreme,
    TableReductionInput,
    reduce_structured_table,
)

PROPERTY_SETTINGS = settings(max_examples=50, deadline=None)


def _operation_table_text(size: int, *, missing_row: int | None = None) -> str:
    elements = tuple(f"e{index}" for index in range(size))
    lines = [
        f"|+|{'|'.join(elements)}|",
        f"|---|{'|'.join('---' for _ in elements)}|",
    ]
    for row_index, element in enumerate(elements):
        if row_index == missing_row:
            continue
        values = tuple(elements[(row_index + column) % size] for column in range(size))
        lines.append(f"|{element}|{'|'.join(values)}|")
    return "\n".join(lines)


def _poisoned_partial_table_accepts(text: str) -> bool:
    """Mutation: check cell widths but forget the complete row inventory."""

    rows = [line.strip()[1:-1].split("|") for line in text.splitlines()]
    width = len(rows[0])
    return bool(rows[2:]) and all(len(row) == width for row in rows[2:])


@PROPERTY_SETTINGS
@example(prefix_metrics=[5, 10], margin=1)
@given(
    prefix_metrics=st.lists(
        st.integers(min_value=-100_000, max_value=100_000),
        min_size=1,
        max_size=20,
    ),
    margin=st.integers(min_value=1, max_value=10_000),
)
def test_last_row_extreme_kills_off_by_one_row_scan_mutant(
    prefix_metrics: list[int],
    margin: int,
) -> None:
    last_metric = max(prefix_metrics) + margin
    metrics = (*prefix_metrics, last_metric)
    specification = TableReductionInput(
        rows=tuple(
            {"identity": f"row-{index}", "metric": metric, "result": index}
            for index, metric in enumerate(metrics)
        ),
        identity_field="identity",
        metric_field="metric",
        result_field="result",
        extreme=Extreme.MAXIMUM,
    )

    result = reduce_structured_table(specification)
    poisoned_skip_last = max(prefix_metrics)

    assert result.selected_row_index == len(prefix_metrics)
    assert result.metric_value == last_metric
    assert result.rows_examined == len(metrics)
    assert poisoned_skip_last != result.metric_value


@PROPERTY_SETTINGS
@given(size=st.integers(min_value=2, max_value=12))
def test_operation_table_always_checks_exactly_n_squared_pairs(size: int) -> None:
    table = parse_operation_table(_operation_table_text(size))

    result = analyze_noncommutativity(table)

    assert result.comparisons_checked == size**2


@PROPERTY_SETTINGS
@example(order=("Zulu", "Mike", "Alpha"))
@given(order=st.permutations(("Zulu", "Alpha", "Mike")))
def test_tie_break_is_invariant_to_source_row_order(order: Sequence[str]) -> None:
    codes = {"Alpha": "ALP", "Mike": "MIK", "Zulu": "ZUL"}
    specification = TableReductionInput(
        rows=tuple(
            {
                "country": country,
                "athletes": 2,
                "ioc_code": codes[country],
            }
            for country in order
        ),
        identity_field="country",
        metric_field="athletes",
        result_field="ioc_code",
        extreme=Extreme.MINIMUM,
        tie_break_field="country",
    )

    result = reduce_structured_table(specification)

    assert result.result_value == "ALP"
    assert result.tie_count == 3
    if order[0] != "Alpha":
        poisoned_first_tied_row = codes[order[0]]
        assert poisoned_first_tied_row != result.result_value


_UNIQUE_WORDS = st.lists(
    st.text(alphabet=ascii_letters, min_size=1, max_size=8),
    min_size=2,
    max_size=15,
    unique_by=str.casefold,
)


@PROPERTY_SETTINGS
@example(items=["Zulu", "alpha", "Mike"])
@given(items=_UNIQUE_WORDS)
def test_alphabetical_contract_kills_preserve_input_order_mutant(items: list[str]) -> None:
    expected_items = sorted(items, key=lambda item: (item.casefold(), item))
    assume(items != expected_items)
    contract = OutputContract(
        answer_type=AnswerType.LIST,
        sort=SortPolicy.ALPHABETICAL,
        delimiter="; ",
    )

    rendered = serialize_answer(ListAnswer(items=items), contract)
    poisoned_preserve_order = "; ".join(items)

    assert rendered == "; ".join(expected_items)
    assert poisoned_preserve_order != rendered


@PROPERTY_SETTINGS
@example(cents=[10, 20])
@given(
    cents=st.lists(
        st.integers(min_value=-10_000_000, max_value=10_000_000),
        min_size=1,
        max_size=30,
    )
)
def test_currency_text_sum_remains_exact_for_generated_cent_values(cents: list[int]) -> None:
    exact_values = [Decimal(value).scaleb(-2) for value in cents]
    workbook_text = [format(value, "f") for value in exact_values]

    parsed_total = sum(
        (decimal_from_spreadsheet(value, field_name="sales") for value in workbook_text),
        start=Decimal(0),
    )

    assert parsed_total == sum(exact_values, start=Decimal(0))


def test_decimal_oracle_kills_binary_float_accumulation_mutant() -> None:
    cells = ("0.1", "0.2")
    exact_total = sum(
        (decimal_from_spreadsheet(value, field_name="sales") for value in cells),
        start=Decimal(0),
    )
    poisoned_float_total = Decimal(str(sum(float(value) for value in cells)))

    assert exact_total == Decimal("0.3")
    assert poisoned_float_total != exact_total


@PROPERTY_SETTINGS
@given(
    size=st.integers(min_value=2, max_value=12),
    data=st.data(),
)
def test_removed_operation_row_kills_incomplete_table_mutant(
    size: int,
    data: st.DataObject,
) -> None:
    missing_row = data.draw(st.integers(min_value=0, max_value=size - 1))
    incomplete = _operation_table_text(size, missing_row=missing_row)

    assert _poisoned_partial_table_accepts(incomplete) is True
    with pytest.raises(OperationTableParseError, match="every header element"):
        parse_operation_table(incomplete)
