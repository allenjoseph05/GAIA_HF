from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from gaia_max.retrieval import (
    StructuredWebTableError,
    parse_html_tables,
    select_html_table,
)
from gaia_max.solvers import Extreme

ARTIFACT_ID = "a" * 64


def test_selects_target_table_and_reduces_maximum_from_same_complete_row() -> None:
    html = """
    <table><tr><th>Noise</th></tr><tr><td>Ignore</td></tr></table>
    <table><caption>1977 batting</caption>
      <tr><th>Player</th><th>BB</th><th>AB</th></tr>
      <tr><td>Player A</td><td>45</td><td>500</td></tr>
      <tr><td>Player B</td><td>72</td><td>431</td></tr>
      <tr><td>Player C</td><td>61</td><td>489</td></tr>
    </table>
    """
    tables = parse_html_tables(html, source_artifact_id=ARTIFACT_ID)
    target = select_html_table(tables, required_headers=("Player", "BB", "AB"))

    result = target.reduce(
        identity_field="Player",
        metric_field="BB",
        result_field="AB",
        extreme=Extreme.MAXIMUM,
    )

    assert target.caption == "1977 batting"
    assert result.selected_identity == "Player B"
    assert result.metric_value == 72
    assert result.result_value == "431"
    assert result.rows_examined == 3


def test_rowspan_is_expanded_for_historical_roster_rows() -> None:
    html = """
    <table>
      <tr><th>Country</th><th>Athlete</th><th>Year</th></tr>
      <tr><td rowspan="2">Japan</td><td>Aiko One</td><td>1928</td></tr>
      <tr><td>Aiko Two</td><td>1928</td></tr>
      <tr><td>Poland</td><td>Person Three</td><td>1928</td></tr>
    </table>
    """
    table = parse_html_tables(html, source_artifact_id=ARTIFACT_ID)[0]

    assert table.records() == (
        {"Country": "Japan", "Athlete": "Aiko One", "Year": "1928"},
        {"Country": "Japan", "Athlete": "Aiko Two", "Year": "1928"},
        {"Country": "Poland", "Athlete": "Person Three", "Year": "1928"},
    )


def test_minimum_tie_uses_explicit_alphabetical_field() -> None:
    html = """
    <table><tr><th>Nation</th><th>Athletes</th><th>IOC</th></tr>
    <tr><td>Zeta</td><td>1</td><td>ZET</td></tr>
    <tr><td>Alpha</td><td>1</td><td>ALP</td></tr>
    <tr><td>Beta</td><td>2</td><td>BET</td></tr></table>
    """
    table = parse_html_tables(html, source_artifact_id=ARTIFACT_ID)[0]

    result = table.reduce(
        identity_field="Nation",
        metric_field="Athletes",
        result_field="IOC",
        extreme=Extreme.MINIMUM,
        tie_break_field="Nation",
    )

    assert result.tie_count == 2
    assert result.selected_identity == "Alpha"
    assert result.result_value == "ALP"


def test_unique_recipient_filter_rejects_zero_or_multiple_rows() -> None:
    html = """
    <table><tr><th>Month</th><th>Topic</th><th>Recipient</th></tr>
    <tr><td>2016-11</td><td>Dinosaur</td><td>One</td></tr>
    <tr><td>2016-11</td><td>Dinosaur</td><td>Two</td></tr></table>
    """
    table = parse_html_tables(html, source_artifact_id=ARTIFACT_ID)[0]
    with pytest.raises(StructuredWebTableError, match="exactly one"):
        table.unique_record({"Month": "2016-11", "Topic": "Dinosaur"})
    with pytest.raises(StructuredWebTableError, match="exactly one"):
        table.unique_record({"Recipient": "Missing"})


def test_table_selection_rejects_ambiguous_headers_and_bad_spans() -> None:
    html = """
    <table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>1</td></tr></table>
    <table><tr><th>Name</th><th>Value</th></tr><tr><td>B</td><td>2</td></tr></table>
    """
    tables = parse_html_tables(html, source_artifact_id=ARTIFACT_ID)
    with pytest.raises(StructuredWebTableError, match="exactly one"):
        select_html_table(tables, required_headers=("Name", "Value"))
    with pytest.raises(StructuredWebTableError, match="span"):
        parse_html_tables(
            "<table><tr><th>Name</th></tr><tr><td rowspan='0'>A</td></tr></table>",
            source_artifact_id=ARTIFACT_ID,
        )


def test_snapshot_timestamp_must_be_aware() -> None:
    html = "<table><tr><th>Name</th></tr><tr><td>A</td></tr></table>"
    table = parse_html_tables(
        html,
        source_artifact_id=ARTIFACT_ID,
        snapshot_at=datetime(2020, 1, 1, tzinfo=UTC),
    )[0]
    assert table.snapshot_at == datetime(2020, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError):
        parse_html_tables(
            html,
            source_artifact_id=ARTIFACT_ID,
            snapshot_at=datetime(2020, 1, 1),
        )
