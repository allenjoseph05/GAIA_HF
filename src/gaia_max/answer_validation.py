"""Independent structural validation of exact-match final-answer strings."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaia_max.domain import AnswerType, NameScope, OutputContract, SortPolicy, UnitsPolicy


class AnswerValidationCode(StrEnum):
    """Stable reason codes used by repair routing and submission preflight."""

    EMPTY_ANSWER = "empty_answer"
    NON_STRING_ANSWER = "non_string_answer"
    CONTRACT_INCOMPLETE = "contract_incomplete"
    FORBIDDEN_PREFIX = "forbidden_prefix"
    EXPLANATORY_PROSE = "explanatory_prose"
    MARKDOWN = "markdown"
    TYPE_MISMATCH = "type_mismatch"
    DECIMAL_PLACES_MISMATCH = "decimal_places_mismatch"
    CURRENCY_SYMBOL_MISMATCH = "currency_symbol_mismatch"
    THOUSANDS_SEPARATOR_MISMATCH = "thousands_separator_mismatch"
    TRAILING_PUNCTUATION_MISMATCH = "trailing_punctuation_mismatch"
    LIST_DELIMITER_MISMATCH = "list_delimiter_mismatch"
    LIST_ITEM_EMPTY = "list_item_empty"
    LIST_ITEM_WHITESPACE = "list_item_whitespace"
    ITEM_COUNT_MISMATCH = "item_count_mismatch"
    SORT_ORDER_MISMATCH = "sort_order_mismatch"
    DUPLICATE_ITEMS = "duplicate_items"
    IOC_CODE_SHAPE = "ioc_code_shape"
    CHESS_SAN_SHAPE = "chess_san_shape"


class AnswerValidationIssue(BaseModel):
    """One actionable structural defect, without copying the candidate answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: AnswerValidationCode
    field: str = Field(min_length=1)
    message: str = Field(min_length=1)


class FinalAnswerValidationResult(BaseModel):
    """Immutable validator result safe to place in future graph state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    valid: bool
    issues: tuple[AnswerValidationIssue, ...] = ()

    @model_validator(mode="after")
    def validate_consistency(self) -> FinalAnswerValidationResult:
        if self.valid == bool(self.issues):
            raise ValueError("valid must be true exactly when issues are empty")
        return self


class FinalAnswerValidator:
    """Check a final string against its contract without rewriting the string."""

    _FORBIDDEN_PREFIX = re.compile(
        r"^(?:(?:final\s+answer|answer|my\s+answer|result|response)\s*[:\-]"
        r"|(?:final\s+answer|the\s+answer|my\s+answer|answer)\s+is\b)",
        re.IGNORECASE,
    )
    _EXPLANATORY_PREFIX = re.compile(
        r"^(?:i\s+(?:found|calculated|determined|believe)\s+that|"
        r"based\s+on|according\s+to|therefore|thus|because)\b",
        re.IGNORECASE,
    )
    _MARKDOWN_PATTERNS: ClassVar[tuple[re.Pattern[str], ...]] = (
        re.compile(r"```"),
        re.compile(r"(?m)^\s{0,3}#{1,6}\s+"),
        re.compile(r"(?m)^\s*>\s+"),
        re.compile(r"(?m)^\s*[-*+]\s+"),
        re.compile(r"\[[^\]\n]+\]\([^\)\n]+\)"),
        re.compile(r"`[^`\n]+`"),
        re.compile(r"\*\*[^*\n]+\*\*"),
    )
    _INTEGER = re.compile(r"-?(?:0|[1-9]\d*)")
    _NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?")
    _GROUPED_INTEGER = re.compile(r"-?[1-9]\d{0,2}(?:,\d{3})+")
    _GROUPED_NUMBER = re.compile(r"-?[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?")
    _IOC_CODE = re.compile(r"[A-Za-z]{3}")
    _CHESS_SAN = re.compile(
        r"(?:"
        r"O-O(?:-O)?"
        r"|[a-h][1-8](?:=[QRBN])?"
        r"|[a-h]x[a-h][1-8](?:=[QRBN])?"
        r"|[KQRBN](?:[a-h]|[1-8]|[a-h][1-8])?x?[a-h][1-8]"
        r")[+#]?"
    )
    _KNOWN_LIST_DELIMITERS: ClassVar[tuple[str, ...]] = (", ", ",", "; ", ";", "\n")
    _TEXT_TYPES: ClassVar[frozenset[AnswerType]] = frozenset(
        {
            AnswerType.STRING,
            AnswerType.EXACT_QUOTE,
            AnswerType.NAME,
            AnswerType.FIRST_NAME,
            AnswerType.SURNAME,
            AnswerType.CITY,
        }
    )

    def validate(
        self,
        serialized_answer: object,
        contract: OutputContract,
    ) -> FinalAnswerValidationResult:
        """Return every independently detectable defect in stable check order."""

        issues: list[AnswerValidationIssue] = []
        if not isinstance(serialized_answer, str):
            self._add(
                issues,
                AnswerValidationCode.NON_STRING_ANSWER,
                "answer",
                "final answer must be a string",
            )
            return self._result(issues)

        payload = serialized_answer.strip()
        if not payload:
            self._add(
                issues,
                AnswerValidationCode.EMPTY_ANSWER,
                "answer",
                "final answer cannot be empty or whitespace-only",
            )
            return self._result(issues)

        self._validate_contract(contract, issues)
        self._validate_content_safety(payload, contract, issues)
        value_payload = self._validate_trailing_punctuation(payload, contract, issues)

        if contract.answer_type is AnswerType.INTEGER:
            self._validate_integer(value_payload, contract, issues)
        elif contract.answer_type in {AnswerType.DECIMAL, AnswerType.CURRENCY}:
            self._validate_decimal(value_payload, contract, issues)
        elif contract.answer_type is AnswerType.LIST:
            self._validate_list(value_payload, contract, issues)
        elif contract.answer_type is AnswerType.IOC_CODE:
            if self._IOC_CODE.fullmatch(value_payload) is None:
                self._add(
                    issues,
                    AnswerValidationCode.IOC_CODE_SHAPE,
                    "answer",
                    "IOC code must contain exactly three ASCII letters",
                )
        elif contract.answer_type is AnswerType.CHESS_SAN:
            if self._CHESS_SAN.fullmatch(value_payload) is None:
                self._add(
                    issues,
                    AnswerValidationCode.CHESS_SAN_SHAPE,
                    "answer",
                    "answer does not have supported standard algebraic notation shape",
                )
        elif contract.answer_type in self._TEXT_TYPES and "\n" in value_payload:
            self._add(
                issues,
                AnswerValidationCode.TYPE_MISMATCH,
                "answer",
                "single text or entity answer cannot contain multiple lines",
            )

        return self._result(issues)

    def _validate_contract(
        self,
        contract: OutputContract,
        issues: list[AnswerValidationIssue],
    ) -> None:
        if contract.answer_type is AnswerType.CURRENCY:
            if contract.decimal_places is None:
                self._add(
                    issues,
                    AnswerValidationCode.CONTRACT_INCOMPLETE,
                    "decimal_places",
                    "currency contract requires explicit decimal places",
                )
            if contract.include_currency_symbol is None:
                self._add(
                    issues,
                    AnswerValidationCode.CONTRACT_INCOMPLETE,
                    "include_currency_symbol",
                    "currency contract requires an explicit symbol decision",
                )
        if (
            contract.answer_type in {AnswerType.INTEGER, AnswerType.DECIMAL}
            and contract.units_policy is not UnitsPolicy.OMIT
        ):
            self._add(
                issues,
                AnswerValidationCode.CONTRACT_INCOMPLETE,
                "units_policy",
                "numeric unit literal and placement are not represented by the contract",
            )
        if contract.answer_type is AnswerType.LIST and not contract.delimiter:
            self._add(
                issues,
                AnswerValidationCode.CONTRACT_INCOMPLETE,
                "delimiter",
                "list delimiter cannot be empty",
            )
        if contract.answer_type is AnswerType.FIRST_NAME and contract.name_scope not in {
            NameScope.UNCHANGED,
            NameScope.FIRST,
        }:
            self._add(
                issues,
                AnswerValidationCode.CONTRACT_INCOMPLETE,
                "name_scope",
                "first-name answer type conflicts with name scope",
            )
        if contract.answer_type is AnswerType.SURNAME and contract.name_scope not in {
            NameScope.UNCHANGED,
            NameScope.SURNAME,
        }:
            self._add(
                issues,
                AnswerValidationCode.CONTRACT_INCOMPLETE,
                "name_scope",
                "surname answer type conflicts with name scope",
            )
        if contract.answer_type in {AnswerType.CITY, AnswerType.IOC_CODE} and (
            contract.name_scope is not NameScope.UNCHANGED
        ):
            self._add(
                issues,
                AnswerValidationCode.CONTRACT_INCOMPLETE,
                "name_scope",
                "place/code answer type conflicts with name scope",
            )

    def _validate_content_safety(
        self,
        payload: str,
        contract: OutputContract,
        issues: list[AnswerValidationIssue],
    ) -> None:
        if contract.answer_type is AnswerType.EXACT_QUOTE:
            return
        if self._FORBIDDEN_PREFIX.match(payload):
            self._add(
                issues,
                AnswerValidationCode.FORBIDDEN_PREFIX,
                "answer",
                "answer begins with a forbidden explanatory label",
            )
        if self._EXPLANATORY_PREFIX.match(payload):
            self._add(
                issues,
                AnswerValidationCode.EXPLANATORY_PROSE,
                "answer",
                "answer begins with explanatory prose",
            )
        if any(pattern.search(payload) for pattern in self._MARKDOWN_PATTERNS):
            self._add(
                issues,
                AnswerValidationCode.MARKDOWN,
                "answer",
                "answer contains Markdown formatting",
            )

    def _validate_trailing_punctuation(
        self,
        payload: str,
        contract: OutputContract,
        issues: list[AnswerValidationIssue],
    ) -> str:
        if not contract.trailing_punctuation:
            return payload
        if not payload.endswith("."):
            self._add(
                issues,
                AnswerValidationCode.TRAILING_PUNCTUATION_MISMATCH,
                "trailing_punctuation",
                "contract requires the answer to end with one period",
            )
            return payload
        if payload.endswith(".."):
            self._add(
                issues,
                AnswerValidationCode.TRAILING_PUNCTUATION_MISMATCH,
                "trailing_punctuation",
                "contract requires exactly one final period",
            )
        return payload[:-1]

    def _validate_integer(
        self,
        payload: str,
        contract: OutputContract,
        issues: list[AnswerValidationIssue],
    ) -> None:
        plain_match = self._INTEGER.fullmatch(payload)
        grouped_match = self._GROUPED_INTEGER.fullmatch(payload)
        if (plain_match is None and grouped_match is None) or payload == "-0":
            self._add(
                issues,
                AnswerValidationCode.TYPE_MISMATCH,
                "answer",
                "integer answer must contain only a canonical signed integer",
            )
            return
        self._validate_grouping(payload, contract, issues)

    def _validate_decimal(
        self,
        payload: str,
        contract: OutputContract,
        issues: list[AnswerValidationIssue],
    ) -> None:
        numeric_payload = payload
        if contract.answer_type is AnswerType.CURRENCY:
            has_symbol = payload.startswith("$") or payload.startswith("-$")
            if contract.include_currency_symbol is not None and (
                has_symbol is not contract.include_currency_symbol
            ):
                self._add(
                    issues,
                    AnswerValidationCode.CURRENCY_SYMBOL_MISMATCH,
                    "include_currency_symbol",
                    "currency symbol presence does not match the output contract",
                )
            if payload.startswith("$-"):
                self._add(
                    issues,
                    AnswerValidationCode.CURRENCY_SYMBOL_MISMATCH,
                    "include_currency_symbol",
                    "negative currency must place the sign before the symbol",
                )
            if payload.startswith("-$"):
                numeric_payload = f"-{payload[2:]}"
            elif payload.startswith("$"):
                numeric_payload = payload[1:]

        plain_match = self._NUMBER.fullmatch(numeric_payload)
        grouped_match = self._GROUPED_NUMBER.fullmatch(numeric_payload)
        numeric_without_grouping = numeric_payload.replace(",", "")
        is_negative_zero = numeric_without_grouping.startswith("-") and (
            set(numeric_without_grouping[1:].replace(".", "")) <= {"0"}
        )
        if (plain_match is None and grouped_match is None) or is_negative_zero:
            self._add(
                issues,
                AnswerValidationCode.TYPE_MISMATCH,
                "answer",
                "decimal answer must use canonical fixed-point digits",
            )
            return

        self._validate_grouping(numeric_payload, contract, issues)
        if contract.decimal_places is not None:
            fraction = numeric_payload.rpartition(".")[2] if "." in numeric_payload else ""
            if len(fraction) != contract.decimal_places:
                self._add(
                    issues,
                    AnswerValidationCode.DECIMAL_PLACES_MISMATCH,
                    "decimal_places",
                    "answer decimal-place count does not match the output contract",
                )

    def _validate_grouping(
        self,
        payload: str,
        contract: OutputContract,
        issues: list[AnswerValidationIssue],
    ) -> None:
        integer_part = payload.lstrip("-").partition(".")[0]
        contains_separator = "," in integer_part
        ungrouped_length = len(integer_part.replace(",", ""))
        required_but_missing = (
            contract.thousands_separator
            and ungrouped_length > 3
            and not contains_separator
        )
        forbidden_but_present = not contract.thousands_separator and contains_separator
        if required_but_missing or forbidden_but_present:
            self._add(
                issues,
                AnswerValidationCode.THOUSANDS_SEPARATOR_MISMATCH,
                "thousands_separator",
                "thousands-separator presence does not match the output contract",
            )

    def _validate_list(
        self,
        payload: str,
        contract: OutputContract,
        issues: list[AnswerValidationIssue],
    ) -> None:
        if not contract.delimiter:
            return
        items = payload.split(contract.delimiter)
        expects_multiple_items = (
            contract.expected_item_count is not None
            and contract.expected_item_count > 1
        )
        if (
            len(items) == 1
            and expects_multiple_items
            and self._contains_other_delimiter(payload, contract.delimiter)
        ):
            self._add(
                issues,
                AnswerValidationCode.LIST_DELIMITER_MISMATCH,
                "delimiter",
                "answer appears to use a delimiter different from the contract",
            )
        if any(item == "" for item in items):
            self._add(
                issues,
                AnswerValidationCode.LIST_ITEM_EMPTY,
                "items",
                "list contains an empty item",
            )
        if any(item != item.strip() for item in items):
            self._add(
                issues,
                AnswerValidationCode.LIST_ITEM_WHITESPACE,
                "items",
                "list item has whitespace not supplied by the delimiter",
            )
        if contract.expected_item_count is not None and (
            len(items) != contract.expected_item_count
        ):
            self._add(
                issues,
                AnswerValidationCode.ITEM_COUNT_MISMATCH,
                "expected_item_count",
                "serialized list item count does not match the output contract",
            )

        normalized_items = [item.casefold() for item in items]
        if not contract.allow_duplicates and len(set(normalized_items)) != len(items):
            self._add(
                issues,
                AnswerValidationCode.DUPLICATE_ITEMS,
                "items",
                "list contains duplicate items but the contract does not allow them",
            )

        if contract.sort is SortPolicy.ALPHABETICAL:
            expected = sorted(items, key=lambda item: (item.casefold(), item))
            if items != expected:
                self._add(
                    issues,
                    AnswerValidationCode.SORT_ORDER_MISMATCH,
                    "sort",
                    "list is not in deterministic alphabetical order",
                )
        elif contract.sort is SortPolicy.NUMERIC_ASCENDING:
            if any(self._INTEGER.fullmatch(item) is None for item in items):
                self._add(
                    issues,
                    AnswerValidationCode.TYPE_MISMATCH,
                    "items",
                    "numeric list contains a non-integer item",
                )
            else:
                values = [int(item) for item in items]
                if values != sorted(values):
                    self._add(
                        issues,
                        AnswerValidationCode.SORT_ORDER_MISMATCH,
                        "sort",
                        "list is not in numeric ascending order",
                    )

    @classmethod
    def _contains_other_delimiter(cls, payload: str, expected: str) -> bool:
        return any(
            delimiter != expected and delimiter in payload
            for delimiter in cls._KNOWN_LIST_DELIMITERS
        )

    @staticmethod
    def _add(
        issues: list[AnswerValidationIssue],
        code: AnswerValidationCode,
        field: str,
        message: str,
    ) -> None:
        issue = AnswerValidationIssue(code=code, field=field, message=message)
        if issue not in issues:
            issues.append(issue)

    @staticmethod
    def _result(issues: list[AnswerValidationIssue]) -> FinalAnswerValidationResult:
        return FinalAnswerValidationResult(valid=not issues, issues=tuple(issues))


def validate_final_answer(
    serialized_answer: object,
    contract: OutputContract,
) -> FinalAnswerValidationResult:
    """Convenience entry point for graph nodes and preflight."""

    return FinalAnswerValidator().validate(serialized_answer, contract)
