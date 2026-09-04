from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

import pytest
from pydantic import ValidationError

from gaia_max.domain import (
    AnswerType,
    ChessMoveAnswer,
    DecimalAnswer,
    EntityAnswer,
    IntegerAnswer,
    ListAnswer,
    NameScope,
    OutputContract,
    SortPolicy,
    StringAnswer,
    UnitsPolicy,
)
from gaia_max.serialization import (
    IncompatibleSemanticAnswerError,
    IncompleteOutputContractError,
    InvalidListAnswerError,
    serialize_answer,
)


def contract(answer_type: AnswerType, **updates: object) -> OutputContract:
    return OutputContract(answer_type=answer_type, **updates)


def test_integer_uses_digits_only_unless_grouping_is_explicit() -> None:
    answer = IntegerAnswer(value=1234)

    assert serialize_answer(answer, contract(AnswerType.INTEGER)) == "1234"
    assert (
        serialize_answer(
            answer,
            contract(AnswerType.INTEGER, thousands_separator=True),
        )
        == "1,234"
    )


def test_integer_contract_does_not_coerce_decimal_answer() -> None:
    with pytest.raises(IncompatibleSemanticAnswerError):
        serialize_answer(DecimalAnswer(value=Decimal("3")), contract(AnswerType.INTEGER))


@pytest.mark.parametrize(
    ("value", "places", "expected"),
    [
        ("12.345", 2, "12.34"),
        ("12.355", 2, "12.36"),
        ("1234.50", None, "1234.50"),
        ("-0.004", 2, "0.00"),
    ],
)
def test_decimal_formatting_is_exact_and_uses_half_even(
    value: str,
    places: int | None,
    expected: str,
) -> None:
    assert (
        serialize_answer(
            DecimalAnswer(value=Decimal(value)),
            contract(AnswerType.DECIMAL, decimal_places=places),
        )
        == expected
    )


def test_decimal_grouping_is_contract_controlled() -> None:
    assert (
        serialize_answer(
            DecimalAnswer(value=Decimal("1234567.8")),
            contract(
                AnswerType.DECIMAL,
                decimal_places=2,
                thousands_separator=True,
            ),
        )
        == "1,234,567.80"
    )


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [(True, "$1,234.50"), (False, "1,234.50")],
)
def test_currency_symbol_is_never_guessed(symbol: bool, expected: str) -> None:
    output = serialize_answer(
        DecimalAnswer(value=Decimal("1234.5")),
        contract(
            AnswerType.CURRENCY,
            decimal_places=2,
            include_currency_symbol=symbol,
            thousands_separator=True,
            units_policy=UnitsPolicy.QUESTION_SPECIFIC,
        ),
    )

    assert output == expected


def test_negative_currency_places_sign_before_dollar_symbol() -> None:
    output = serialize_answer(
        DecimalAnswer(value=Decimal("-12.3")),
        contract(
            AnswerType.CURRENCY,
            decimal_places=2,
            include_currency_symbol=True,
            units_policy=UnitsPolicy.QUESTION_SPECIFIC,
        ),
    )

    assert output == "-$12.30"


@pytest.mark.parametrize(
    "incomplete_contract",
    [
        contract(
            AnswerType.CURRENCY,
            decimal_places=2,
            include_currency_symbol=None,
            units_policy=UnitsPolicy.QUESTION_SPECIFIC,
        ),
        contract(
            AnswerType.CURRENCY,
            include_currency_symbol=False,
            units_policy=UnitsPolicy.QUESTION_SPECIFIC,
        ),
    ],
)
def test_incomplete_currency_contract_is_rejected(
    incomplete_contract: OutputContract,
) -> None:
    with pytest.raises(IncompleteOutputContractError):
        serialize_answer(DecimalAnswer(value=Decimal("12.34")), incomplete_contract)


@pytest.mark.parametrize("answer_type", [AnswerType.STRING, AnswerType.EXACT_QUOTE])
def test_text_preserves_accents_quotes_articles_and_punctuation(
    answer_type: AnswerType,
) -> None:
    value = '"L\u2019\u00e9t\u00e9, the answer!"'

    assert serialize_answer(StringAnswer(value=value), contract(answer_type)) == value


@pytest.mark.parametrize(
    ("answer_type", "scope", "expected"),
    [
        (AnswerType.NAME, NameScope.FULL, "Ada Lovelace"),
        (AnswerType.FIRST_NAME, NameScope.FIRST, "Ada"),
        (AnswerType.SURNAME, NameScope.SURNAME, "Lovelace"),
        (AnswerType.CITY, NameScope.UNCHANGED, "Z\u00fcrich"),
        (AnswerType.IOC_CODE, NameScope.UNCHANGED, "SUI"),
    ],
)
def test_entity_contract_selects_resolved_component(
    answer_type: AnswerType,
    scope: NameScope,
    expected: str,
) -> None:
    answer = EntityAnswer(
        full_name="Ada Lovelace",
        first_name="Ada",
        surname="Lovelace",
        city="Z\u00fcrich",
        ioc_code="SUI",
    )

    assert serialize_answer(answer, contract(answer_type, name_scope=scope)) == expected


def test_missing_entity_component_is_not_inferred_from_full_name() -> None:
    with pytest.raises(IncompatibleSemanticAnswerError, match="first_name"):
        serialize_answer(
            EntityAnswer(full_name="Ada Lovelace"),
            contract(AnswerType.FIRST_NAME, name_scope=NameScope.FIRST),
        )


def test_entity_type_rejects_an_unrelated_name_scope() -> None:
    with pytest.raises(IncompleteOutputContractError, match="city answer type conflicts"):
        serialize_answer(
            EntityAnswer(city="Z\u00fcrich"),
            contract(AnswerType.CITY, name_scope=NameScope.SURNAME),
        )


def test_alphabetical_list_uses_casefold_key_but_preserves_source_spelling() -> None:
    answer = ListAnswer(items=["\u00c9clair", "banana", "Apple"])

    output = serialize_answer(
        answer,
        contract(AnswerType.LIST, sort=SortPolicy.ALPHABETICAL),
    )

    assert output == "Apple, banana, \u00c9clair"


def test_numeric_list_is_sorted_as_integers_not_lexicographically() -> None:
    output = serialize_answer(
        ListAnswer(items=[10, 2, 1]),
        contract(AnswerType.LIST, sort=SortPolicy.NUMERIC_ASCENDING),
    )

    assert output == "1, 2, 10"


def test_question_order_and_custom_delimiter_are_preserved() -> None:
    output = serialize_answer(
        ListAnswer(items=["second", "first"]),
        contract(
            AnswerType.LIST,
            sort=SortPolicy.QUESTION_ORDER,
            delimiter="; ",
        ),
    )

    assert output == "second; first"


def test_scoped_entity_list_selects_each_surname() -> None:
    answer = ListAnswer(
        items=[
            EntityAnswer(full_name="Mar\u00eda Garc\u00eda", surname="Garc\u00eda"),
            EntityAnswer(full_name="Zo\u00eb Smith", surname="Smith"),
        ]
    )

    output = serialize_answer(
        answer,
        contract(
            AnswerType.LIST,
            name_scope=NameScope.SURNAME,
            expected_item_count=2,
        ),
    )

    assert output == "Garc\u00eda, Smith"


def test_scoped_name_list_rejects_preflattened_strings() -> None:
    with pytest.raises(InvalidListAnswerError, match="requires EntityAnswer"):
        serialize_answer(
            ListAnswer(items=["Garc\u00eda", "Smith"]),
            contract(AnswerType.LIST, name_scope=NameScope.SURNAME),
        )


def test_list_count_must_match_contract() -> None:
    with pytest.raises(InvalidListAnswerError, match="expected 2, received 1"):
        serialize_answer(
            ListAnswer(items=["only one"]),
            contract(AnswerType.LIST, expected_item_count=2),
        )


def test_numeric_sort_rejects_text_numbers() -> None:
    with pytest.raises(InvalidListAnswerError, match="requires integer"):
        serialize_answer(
            ListAnswer(items=["10", "2"]),
            contract(AnswerType.LIST, sort=SortPolicy.NUMERIC_ASCENDING),
        )


def test_chess_returns_stored_engine_derived_san_only() -> None:
    answer = ChessMoveAnswer(uci="e7e8q", san="e8=Q+")

    assert (
        serialize_answer(
            answer,
            contract(AnswerType.CHESS_SAN, notation="algebraic"),
        )
        == "e8=Q+"
    )


def test_unsupported_chess_notation_is_rejected() -> None:
    with pytest.raises(IncompleteOutputContractError, match="unsupported chess notation"):
        serialize_answer(
            ChessMoveAnswer(uci="e2e4", san="e4"),
            contract(AnswerType.CHESS_SAN, notation="uci"),
        )


def test_required_period_is_added_once_without_other_cleanup() -> None:
    assert (
        serialize_answer(
            StringAnswer(value="answer"),
            contract(AnswerType.STRING, trailing_punctuation=True),
        )
        == "answer."
    )
    assert (
        serialize_answer(
            StringAnswer(value="already."),
            contract(AnswerType.STRING, trailing_punctuation=True),
        )
        == "already."
    )


def test_non_currency_literal_unit_contract_is_not_guessed() -> None:
    with pytest.raises(IncompleteOutputContractError, match="literal"):
        serialize_answer(
            DecimalAnswer(value=Decimal("12.5")),
            contract(AnswerType.DECIMAL, units_policy=UnitsPolicy.LITERAL_REQUIRED),
        )


@pytest.mark.parametrize(
    "invalid_answer",
    [
        lambda: IntegerAnswer(value=True),
        lambda: DecimalAnswer(value=0.1),
        lambda: DecimalAnswer(value=Decimal("NaN")),
        lambda: ListAnswer(items=[True]),
    ],
)
def test_semantic_numeric_models_reject_lossy_or_boolean_values(
    invalid_answer: Callable[[], object],
) -> None:
    with pytest.raises(ValidationError):
        invalid_answer()
