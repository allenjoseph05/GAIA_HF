from __future__ import annotations

from decimal import Decimal

import pytest

from gaia_max.answer_validation import (
    AnswerValidationCode,
    FinalAnswerValidationResult,
    validate_final_answer,
)
from gaia_max.domain import (
    AnswerType,
    ChessMoveAnswer,
    DecimalAnswer,
    EntityAnswer,
    IntegerAnswer,
    ListAnswer,
    NameScope,
    OutputContract,
    SemanticAnswer,
    SortPolicy,
    StringAnswer,
    UnitsPolicy,
)
from gaia_max.serialization import serialize_answer


def contract(answer_type: AnswerType, **updates: object) -> OutputContract:
    return OutputContract(answer_type=answer_type, **updates)


def codes(answer: object, output_contract: OutputContract) -> set[AnswerValidationCode]:
    return {
        issue.code
        for issue in validate_final_answer(answer, output_contract).issues
    }


@pytest.mark.parametrize("answer", ["", "   ", "\t\r\n"])
def test_empty_answer_is_rejected(answer: str) -> None:
    assert codes(answer, contract(AnswerType.STRING)) == {
        AnswerValidationCode.EMPTY_ANSWER
    }


def test_non_string_answer_is_rejected_without_echoing_it() -> None:
    result = validate_final_answer({"answer": "secret candidate"}, contract(AnswerType.STRING))

    assert result.valid is False
    assert result.issues[0].code is AnswerValidationCode.NON_STRING_ANSWER
    assert "secret candidate" not in result.model_dump_json()


@pytest.mark.parametrize(
    "answer",
    [
        "Answer: 42",
        "FINAL ANSWER - 42",
        "My answer: 42",
        "Result: 42",
        "The answer is 42",
        "Answer is 42",
    ],
)
def test_forbidden_answer_prefixes_are_reason_coded(answer: str) -> None:
    assert AnswerValidationCode.FORBIDDEN_PREFIX in codes(
        answer,
        contract(AnswerType.INTEGER),
    )


@pytest.mark.parametrize(
    "answer",
    ["I found that 42", "Based on the table, 42", "According to the source, 42"],
)
def test_explanatory_prose_is_rejected(answer: str) -> None:
    assert AnswerValidationCode.EXPLANATORY_PROSE in codes(
        answer,
        contract(AnswerType.INTEGER),
    )


@pytest.mark.parametrize(
    "answer",
    ["**42**", "`42`", "# 42", "[42](https://example.test)", "- 42"],
)
def test_markdown_is_rejected(answer: str) -> None:
    assert AnswerValidationCode.MARKDOWN in codes(answer, contract(AnswerType.STRING))


def test_exact_quote_is_exempt_from_unsafe_global_text_rewriting() -> None:
    answer = "Answer: **absolutely**, according to me!"

    result = validate_final_answer(answer, contract(AnswerType.EXACT_QUOTE))

    assert result.valid is True
    assert answer == "Answer: **absolutely**, according to me!"


def test_grader_ignored_outer_whitespace_is_a_valid_edge_case() -> None:
    answer = "  42\r\n"

    result = validate_final_answer(answer, contract(AnswerType.INTEGER))

    assert result.valid is True
    assert answer == "  42\r\n"


@pytest.mark.parametrize(
    ("answer", "grouping", "valid"),
    [
        ("1234", False, True),
        ("1,234", True, True),
        ("1,234", False, False),
        ("1234", True, False),
    ],
)
def test_integer_shape_and_grouping(answer: str, grouping: bool, valid: bool) -> None:
    result = validate_final_answer(
        answer,
        contract(AnswerType.INTEGER, thousands_separator=grouping),
    )

    assert result.valid is valid


@pytest.mark.parametrize("answer", ["3.0", "+3", "003", "-0", "three", "3e0"])
def test_integer_rejects_noncanonical_forms(answer: str) -> None:
    assert AnswerValidationCode.TYPE_MISMATCH in codes(
        answer,
        contract(AnswerType.INTEGER),
    )


def test_decimal_precision_is_checked_independently() -> None:
    output_contract = contract(AnswerType.DECIMAL, decimal_places=2)

    assert validate_final_answer("12.30", output_contract).valid is True
    assert AnswerValidationCode.DECIMAL_PLACES_MISMATCH in codes("12.3", output_contract)
    assert AnswerValidationCode.DECIMAL_PLACES_MISMATCH in codes("12", output_contract)


@pytest.mark.parametrize("answer", ["NaN", "Infinity", "1e3", ".50", "1."])
def test_decimal_rejects_non_fixed_point_forms(answer: str) -> None:
    assert AnswerValidationCode.TYPE_MISMATCH in codes(
        answer,
        contract(AnswerType.DECIMAL),
    )


@pytest.mark.parametrize(
    ("answer", "symbol", "valid"),
    [
        ("$12.30", True, True),
        ("-$12.30", True, True),
        ("12.30", False, True),
        ("$12.30", False, False),
        ("12.30", True, False),
        ("$-12.30", True, False),
        ("-$0.00", True, False),
    ],
)
def test_currency_symbol_policy(answer: str, symbol: bool, valid: bool) -> None:
    result = validate_final_answer(
        answer,
        contract(
            AnswerType.CURRENCY,
            decimal_places=2,
            include_currency_symbol=symbol,
            units_policy=UnitsPolicy.QUESTION_SPECIFIC,
        ),
    )

    assert result.valid is valid


def test_ambiguous_currency_contract_is_rejected_even_for_plausible_text() -> None:
    result = validate_final_answer(
        "12.30",
        contract(
            AnswerType.CURRENCY,
            decimal_places=2,
            include_currency_symbol=None,
            units_policy=UnitsPolicy.QUESTION_SPECIFIC,
        ),
    )

    assert AnswerValidationCode.CONTRACT_INCOMPLETE in {
        issue.code for issue in result.issues
    }


def test_required_trailing_period_is_checked_and_removed_only_for_parsing() -> None:
    output_contract = contract(AnswerType.INTEGER, trailing_punctuation=True)

    assert validate_final_answer("42.", output_contract).valid is True
    assert AnswerValidationCode.TRAILING_PUNCTUATION_MISMATCH in codes(
        "42",
        output_contract,
    )
    assert AnswerValidationCode.TRAILING_PUNCTUATION_MISMATCH in codes(
        "42..",
        output_contract,
    )


def test_list_delimiter_item_count_and_empty_items_are_checked() -> None:
    output_contract = contract(
        AnswerType.LIST,
        delimiter=", ",
        expected_item_count=3,
    )

    assert validate_final_answer("a, b, c", output_contract).valid is True
    wrong = codes("a,b,", output_contract)
    assert AnswerValidationCode.LIST_DELIMITER_MISMATCH in wrong
    assert AnswerValidationCode.ITEM_COUNT_MISMATCH in wrong
    assert AnswerValidationCode.LIST_ITEM_EMPTY in codes("a, , c", output_contract)


def test_single_list_item_with_punctuation_is_not_guessed_to_be_wrong_delimiter() -> None:
    output_contract = contract(AnswerType.LIST, delimiter="\n", expected_item_count=1)

    assert validate_final_answer("Washington, D.C.", output_contract).valid is True


def test_alphabetical_sort_and_case_insensitive_duplicates_are_checked() -> None:
    output_contract = contract(AnswerType.LIST, sort=SortPolicy.ALPHABETICAL)

    assert validate_final_answer("Apple, banana, \u00c9clair", output_contract).valid is True
    assert AnswerValidationCode.SORT_ORDER_MISMATCH in codes(
        "banana, Apple, \u00c9clair",
        output_contract,
    )
    assert AnswerValidationCode.DUPLICATE_ITEMS in codes(
        "Apple, apple",
        output_contract,
    )


def test_duplicate_list_items_can_be_explicitly_allowed() -> None:
    output_contract = contract(AnswerType.LIST, allow_duplicates=True)

    assert validate_final_answer("echo, echo", output_contract).valid is True


def test_numeric_list_requires_integer_items_and_ascending_order() -> None:
    output_contract = contract(AnswerType.LIST, sort=SortPolicy.NUMERIC_ASCENDING)

    assert validate_final_answer("1, 2, 10", output_contract).valid is True
    assert AnswerValidationCode.SORT_ORDER_MISMATCH in codes(
        "1, 10, 2",
        output_contract,
    )
    assert AnswerValidationCode.TYPE_MISMATCH in codes(
        "1, two, 3",
        output_contract,
    )


@pytest.mark.parametrize("answer", ["USA", "Pol", "gbr"])
def test_valid_ioc_code_shapes(answer: str) -> None:
    assert validate_final_answer(answer, contract(AnswerType.IOC_CODE)).valid is True


@pytest.mark.parametrize("answer", ["US", "USAA", "U1A", "U.S."])
def test_invalid_ioc_code_shapes(answer: str) -> None:
    assert AnswerValidationCode.IOC_CODE_SHAPE in codes(
        answer,
        contract(AnswerType.IOC_CODE),
    )


@pytest.mark.parametrize(
    "answer",
    ["e4", "Nf3", "Nbd2", "R1e2", "exd5", "e8=Q+", "Qh7#", "O-O", "O-O-O+"],
)
def test_supported_san_shapes(answer: str) -> None:
    assert validate_final_answer(answer, contract(AnswerType.CHESS_SAN)).valid is True


@pytest.mark.parametrize("answer", ["e2e4", "pawn e4", "0-0", "Qh7!!", "e9"])
def test_invalid_san_shapes(answer: str) -> None:
    assert AnswerValidationCode.CHESS_SAN_SHAPE in codes(
        answer,
        contract(AnswerType.CHESS_SAN),
    )


def test_chess_checkmate_marker_is_not_mistaken_for_markdown() -> None:
    assert validate_final_answer("Qh7#", contract(AnswerType.CHESS_SAN)).valid is True


def test_currency_and_percent_characters_are_not_globally_stripped() -> None:
    currency_contract = contract(
        AnswerType.CURRENCY,
        decimal_places=2,
        include_currency_symbol=True,
        units_policy=UnitsPolicy.QUESTION_SPECIFIC,
    )

    assert validate_final_answer("$12.50", currency_contract).valid is True
    assert validate_final_answer("50%", contract(AnswerType.STRING)).valid is True


@pytest.mark.parametrize(
    ("semantic_answer", "output_contract"),
    [
        (IntegerAnswer(value=42), contract(AnswerType.INTEGER)),
        (
            DecimalAnswer(value=Decimal("12.5")),
            contract(AnswerType.DECIMAL, decimal_places=2),
        ),
        (StringAnswer(value="L\u2019\u00e9t\u00e9!"), contract(AnswerType.EXACT_QUOTE)),
        (
            EntityAnswer(first_name="Ada"),
            contract(AnswerType.FIRST_NAME, name_scope=NameScope.FIRST),
        ),
        (
            ListAnswer(items=[3, 1, 2]),
            contract(AnswerType.LIST, sort=SortPolicy.NUMERIC_ASCENDING),
        ),
        (
            ChessMoveAnswer(uci="e7e8q", san="e8=Q+"),
            contract(AnswerType.CHESS_SAN, notation="algebraic"),
        ),
    ],
)
def test_every_representative_a13_output_passes_independent_a14_validation(
    semantic_answer: SemanticAnswer,
    output_contract: OutputContract,
) -> None:
    serialized = serialize_answer(semantic_answer, output_contract)

    assert validate_final_answer(serialized, output_contract).valid is True


def test_validation_result_contains_no_candidate_answer_field() -> None:
    field_names = set(FinalAnswerValidationResult.model_fields)

    assert field_names == {"valid", "issues"}
