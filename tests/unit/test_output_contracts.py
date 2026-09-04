from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaia_max.domain import (
    AnswerType,
    NameScope,
    OutputContract,
    SortPolicy,
    UnitsPolicy,
)
from gaia_max.output_contracts import (
    ContractIssueCode,
    ContractParseStatus,
    OutputContractParser,
    OutputContractParseResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = PROJECT_ROOT / "config" / "task_profiles.json"
CONTRACT_CASES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "output_contract_cases.json"


def test_all_current_questions_match_answer_free_profile_contracts() -> None:
    registry = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    cases = json.loads(CONTRACT_CASES_PATH.read_text(encoding="utf-8"))
    questions = {item["task_id"]: item["question_shape"] for item in cases}
    parser = OutputContractParser()

    assert set(questions) == {profile["task_id"] for profile in registry["profiles"]}

    for profile in registry["profiles"]:
        result = parser.parse(questions[profile["task_id"]])
        expected = OutputContract.model_validate(profile["output_contract"])

        assert result.contract == expected, profile["task_id"]
        if profile["task_id"] == "7bd855d8-463d-4ed5-93ca-5fe35145f733":
            assert result.status is ContractParseStatus.AMBIGUOUS
            assert result.issues[0].code is ContractIssueCode.CURRENCY_SYMBOL_UNSPECIFIED
        else:
            assert result.status is ContractParseStatus.COMPLETE, profile["task_id"]


def test_unknown_requirement_is_ambiguous_instead_of_defaulting_to_string() -> None:
    result = OutputContractParser().parse("Please provide the appropriate response.")

    assert result.status is ContractParseStatus.AMBIGUOUS
    assert result.contract is None
    assert result.issues[0].code is ContractIssueCode.MISSING_ANSWER_TYPE


def test_conflicting_name_scope_and_answer_type_are_blocked() -> None:
    result = OutputContractParser().parse(
        "Give only the first name and the surname as the answer."
    )

    assert result.status is ContractParseStatus.CONFLICT
    assert result.contract is None
    assert {issue.field for issue in result.issues} >= {"answer_type", "name_scope"}


def test_conflicting_list_sort_requirements_are_blocked() -> None:
    result = OutputContractParser().parse(
        "Return a comma-separated list in alphabetical order and ascending order."
    )

    assert result.status is ContractParseStatus.CONFLICT
    assert result.contract is None
    assert any(issue.field == "sort" for issue in result.issues)


def test_unsupported_descending_sort_is_explicitly_ambiguous() -> None:
    result = OutputContractParser().parse(
        "Return a comma-separated list in descending order."
    )

    assert result.status is ContractParseStatus.AMBIGUOUS
    assert result.contract is not None
    assert result.contract.sort is SortPolicy.NONE
    assert result.issues[0].code is ContractIssueCode.UNSUPPORTED_REQUIREMENT


def test_explicit_currency_symbol_precision_and_separator_are_complete() -> None:
    result = OutputContractParser().parse(
        "Express the answer in USD with three decimal places. Include the dollar sign "
        "and use thousands separators."
    )

    assert result.status is ContractParseStatus.COMPLETE
    assert result.contract == OutputContract(
        answer_type=AnswerType.CURRENCY,
        decimal_places=3,
        units_policy=UnitsPolicy.QUESTION_SPECIFIC,
        include_currency_symbol=True,
        thousands_separator=True,
    )


def test_semicolon_list_with_item_count_and_name_scope() -> None:
    result = OutputContractParser().parse(
        "Provide exactly 3 names as a semicolon-separated list in alphabetical order. "
        "Use full names."
    )

    assert result.status is ContractParseStatus.COMPLETE
    assert result.contract == OutputContract(
        answer_type=AnswerType.LIST,
        sort=SortPolicy.ALPHABETICAL,
        delimiter="; ",
        name_scope=NameScope.FULL,
        expected_item_count=3,
    )


def test_explicit_unit_requirements_and_punctuation_are_recorded() -> None:
    result = OutputContractParser().parse(
        "How many meters? Include the units and end with a period."
    )

    assert result.status is ContractParseStatus.COMPLETE
    assert result.contract is not None
    assert result.contract.answer_type is AnswerType.INTEGER
    assert result.contract.units_policy is UnitsPolicy.LITERAL_REQUIRED
    assert result.contract.trailing_punctuation is True


def test_parse_result_schema_contains_no_candidate_or_answer_value_field() -> None:
    field_names = set(OutputContractParseResult.model_fields)

    assert "candidate_answer" not in field_names
    assert "semantic_answer" not in field_names
    assert "serialized_answer" not in field_names
    assert "answer_value" not in field_names


@pytest.mark.parametrize("empty_question", ["", "   ", "\n\t"])
def test_empty_question_is_rejected(empty_question: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        OutputContractParser().parse(empty_question)
