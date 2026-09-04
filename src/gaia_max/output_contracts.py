"""Deterministic parsing of exact-match output requirements from question text."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaia_max.domain import (
    AnswerType,
    NameScope,
    OutputContract,
    SortPolicy,
    UnitsPolicy,
)

T = TypeVar("T")


class ContractParseStatus(StrEnum):
    """Whether hard rules produced an actionable contract."""

    COMPLETE = "complete"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"


class ContractIssueCode(StrEnum):
    """Stable reason codes for non-complete contract parsing."""

    MISSING_ANSWER_TYPE = "missing_answer_type"
    CONFLICTING_REQUIREMENT = "conflicting_requirement"
    UNSUPPORTED_REQUIREMENT = "unsupported_requirement"
    CURRENCY_SYMBOL_UNSPECIFIED = "currency_symbol_unspecified"


class ContractParseIssue(BaseModel):
    """One ambiguity or contradiction requiring explicit reconciliation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ContractIssueCode
    field: str = Field(min_length=1)
    message: str = Field(min_length=1)


class OutputContractParseResult(BaseModel):
    """Question-derived contract plus an audit trail of deterministic rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ContractParseStatus
    contract: OutputContract | None = None
    matched_rules: tuple[str, ...] = ()
    issues: tuple[ContractParseIssue, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> OutputContractParseResult:
        if self.status is ContractParseStatus.COMPLETE:
            if self.contract is None or self.issues:
                raise ValueError("complete parse requires a contract without issues")
        elif not self.issues:
            raise ValueError("non-complete parse requires at least one issue")
        if self.status is ContractParseStatus.CONFLICT and self.contract is not None:
            raise ValueError("conflicting requirements cannot produce a contract")
        return self


class OutputContractParser:
    """Apply conservative, auditable rules; never infer a candidate answer."""

    _DECIMAL_WORDS: ClassVar[dict[str, int]] = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }

    def parse(self, question: str) -> OutputContractParseResult:
        """Parse only explicit or structurally safe output instructions."""

        if not question.strip():
            raise ValueError("question cannot be empty")
        text = self._normalize(question)
        reversed_text = self._normalize(question[::-1])
        matched_rules: list[str] = []
        issues: list[ContractParseIssue] = []

        answer_types: list[tuple[AnswerType, str]] = []
        sorts: list[tuple[SortPolicy, str]] = []
        delimiters: list[tuple[str, str]] = []
        decimal_places: list[tuple[int, str]] = []
        units: list[tuple[UnitsPolicy, str]] = []
        currency_symbols: list[tuple[bool, str]] = []
        name_scopes: list[tuple[NameScope, str]] = []
        notations: list[tuple[str, str]] = []
        item_counts: list[tuple[int, str]] = []
        trailing_punctuation: list[tuple[bool, str]] = []

        def record(candidates: list[tuple[T, str]], value: T, rule_id: str) -> None:
            candidates.append((value, rule_id))
            if rule_id not in matched_rules:
                matched_rules.append(rule_id)

        is_explicit_list = bool(
            re.search(r"\bcomma[ -](?:separated|delimited) list\b", text)
            or re.search(r"\b(?:semicolon|semi-colon)[ -]separated list\b", text)
            or "list in ascending order" in text
            or "list of ingredients" in text
            or "list of just the vegetables" in text
            or ("subset of" in text and "list" in text)
        )
        is_ordered_pair = "pitcher before, pitcher after" in text

        if "ioc country code" in text or "ioc code" in text:
            record(answer_types, AnswerType.IOC_CODE, "answer.ioc_code")
        if "algebraic notation" in text:
            record(answer_types, AnswerType.CHESS_SAN, "answer.chess_algebraic")
            record(notations, "algebraic", "notation.algebraic")
        if is_ordered_pair:
            record(answer_types, AnswerType.LIST, "answer.ordered_pair")
            record(sorts, SortPolicy.QUESTION_ORDER, "sort.before_after")
            record(item_counts, 2, "count.before_after_pair")
        if is_explicit_list:
            record(answer_types, AnswerType.LIST, "answer.explicit_list")
        if "who nominated" in text:
            record(answer_types, AnswerType.LIST, "answer.nominator_list")
            record(name_scopes, NameScope.FULL, "name.full_nominator")
        if "last names only" in text or "last name only" in text:
            record(name_scopes, NameScope.SURNAME, "name.last_only")
            if not is_ordered_pair and not is_explicit_list:
                record(answer_types, AnswerType.SURNAME, "answer.last_name")
        if "surname" in text:
            record(name_scopes, NameScope.SURNAME, "name.surname")
            if not is_ordered_pair and not is_explicit_list:
                record(answer_types, AnswerType.SURNAME, "answer.surname")
        if "first name" in text:
            record(name_scopes, NameScope.FIRST, "name.first")
            if not is_ordered_pair and not is_explicit_list:
                record(answer_types, AnswerType.FIRST_NAME, "answer.first_name")
        if "full name" in text:
            record(name_scopes, NameScope.FULL, "name.full")
            if not is_ordered_pair and not is_explicit_list:
                record(answer_types, AnswerType.NAME, "answer.full_name")
        if "city name" in text:
            record(answer_types, AnswerType.CITY, "answer.city")
        if "what does" in text and "say in response" in text:
            record(answer_types, AnswerType.EXACT_QUOTE, "answer.exact_spoken_reply")
        if re.search(r"\bhow many\b", text) or "highest number" in text:
            record(answer_types, AnswerType.INTEGER, "answer.integer_count")
        if "final numeric output" in text:
            record(answer_types, AnswerType.STRING, "answer.exact_program_output")
        if "award number" in text:
            record(answer_types, AnswerType.STRING, "answer.identifier")
        if "as the answer" in reversed_text and "write" in reversed_text:
            record(answer_types, AnswerType.STRING, "answer.reversed_instruction_string")
        if re.search(r"\b(?:in|as) usd\b", text):
            record(answer_types, AnswerType.CURRENCY, "answer.currency_usd")
            record(units, UnitsPolicy.QUESTION_SPECIFIC, "units.usd")

        if "alphabetize" in text or (
            "alphabetical order" in text and (is_explicit_list or is_ordered_pair)
        ):
            record(sorts, SortPolicy.ALPHABETICAL, "sort.alphabetical")
        if "ascending order" in text:
            record(sorts, SortPolicy.NUMERIC_ASCENDING, "sort.numeric_ascending")
        if "in the order given" in text or "in the order mentioned" in text:
            record(sorts, SortPolicy.QUESTION_ORDER, "sort.question_order")
        if "descending order" in text:
            matched_rules.append("sort.descending_unsupported")
            issues.append(
                ContractParseIssue(
                    code=ContractIssueCode.UNSUPPORTED_REQUIREMENT,
                    field="sort",
                    message="descending order is not represented by SortPolicy",
                )
            )

        if re.search(r"\bcomma[ -](?:separated|delimited) list\b", text):
            record(delimiters, ", ", "delimiter.comma_space")
        if re.search(r"\b(?:semicolon|semi-colon)[ -]separated list\b", text):
            record(delimiters, "; ", "delimiter.semicolon_space")
        if "one item per line" in text:
            record(delimiters, "\n", "delimiter.newline")

        precision_pattern = re.compile(
            r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
            r" decimal places?\b"
        )
        for match in precision_pattern.finditer(text):
            raw_precision = match.group(1)
            precision = (
                int(raw_precision)
                if raw_precision.isdigit()
                else self._DECIMAL_WORDS[raw_precision]
            )
            record(decimal_places, precision, f"precision.decimal_{precision}")

        if "without units" in text or "do not include units" in text:
            record(units, UnitsPolicy.OMIT, "units.omit_explicit")
        if "include the units" in text or "include units" in text:
            record(units, UnitsPolicy.LITERAL_REQUIRED, "units.literal_required")
        if "without a currency symbol" in text or "do not include a currency symbol" in text:
            record(currency_symbols, False, "currency_symbol.omit")
        if "include the dollar sign" in text or "include a dollar sign" in text:
            record(currency_symbols, True, "currency_symbol.dollar_required")
        if "use thousands separators" in text or "include thousands separators" in text:
            thousands_separator = True
            matched_rules.append("thousands_separator.required")
        else:
            thousands_separator = False

        count_pattern = re.compile(r"\b(?:exactly|provide|return|list) (\d+) (?:items?|names?)\b")
        for match in count_pattern.finditer(text):
            count = int(match.group(1))
            record(item_counts, count, f"count.explicit_{count}")

        if "end with a period" in text:
            record(trailing_punctuation, True, "punctuation.period_required")
        if "no trailing punctuation" in text:
            record(trailing_punctuation, False, "punctuation.trailing_forbidden")

        answer_type = self._resolve(answer_types, "answer_type", issues)
        sort = self._resolve(sorts, "sort", issues)
        delimiter = self._resolve(delimiters, "delimiter", issues)
        precision = self._resolve(decimal_places, "decimal_places", issues)
        units_policy = self._resolve(units, "units_policy", issues)
        include_currency_symbol = self._resolve(
            currency_symbols,
            "include_currency_symbol",
            issues,
        )
        name_scope = self._resolve(name_scopes, "name_scope", issues)
        notation = self._resolve(notations, "notation", issues)
        expected_item_count = self._resolve(item_counts, "expected_item_count", issues)
        punctuation = self._resolve(
            trailing_punctuation,
            "trailing_punctuation",
            issues,
        )

        if answer_type is None and not self._has_conflict(issues, "answer_type"):
            issues.append(
                ContractParseIssue(
                    code=ContractIssueCode.MISSING_ANSWER_TYPE,
                    field="answer_type",
                    message="no deterministic answer-type rule matched",
                )
            )
        if (
            answer_type is AnswerType.CURRENCY
            and include_currency_symbol is None
            and not self._has_conflict(issues, "include_currency_symbol")
        ):
            issues.append(
                ContractParseIssue(
                    code=ContractIssueCode.CURRENCY_SYMBOL_UNSPECIFIED,
                    field="include_currency_symbol",
                    message="currency requested without an explicit symbol convention",
                )
            )

        has_conflict = any(
            issue.code is ContractIssueCode.CONFLICTING_REQUIREMENT for issue in issues
        )
        if has_conflict:
            return OutputContractParseResult(
                status=ContractParseStatus.CONFLICT,
                matched_rules=tuple(matched_rules),
                issues=tuple(issues),
            )
        if answer_type is None:
            return OutputContractParseResult(
                status=ContractParseStatus.AMBIGUOUS,
                matched_rules=tuple(matched_rules),
                issues=tuple(issues),
            )

        contract = OutputContract(
            answer_type=answer_type,
            sort=sort or SortPolicy.NONE,
            delimiter=delimiter if delimiter is not None else ", ",
            decimal_places=precision,
            units_policy=units_policy or UnitsPolicy.OMIT,
            include_currency_symbol=include_currency_symbol,
            thousands_separator=thousands_separator,
            name_scope=name_scope or NameScope.UNCHANGED,
            notation=notation,
            trailing_punctuation=punctuation or False,
            expected_item_count=expected_item_count,
        )
        status = ContractParseStatus.AMBIGUOUS if issues else ContractParseStatus.COMPLETE
        return OutputContractParseResult(
            status=status,
            contract=contract,
            matched_rules=tuple(matched_rules),
            issues=tuple(issues),
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().replace("\u2019", "'").split())

    @staticmethod
    def _resolve(
        candidates: list[tuple[T, str]],
        field: str,
        issues: list[ContractParseIssue],
    ) -> T | None:
        distinct_values = list(dict.fromkeys(value for value, _rule in candidates))
        if len(distinct_values) > 1:
            rendered = ", ".join(str(value) for value in distinct_values)
            issues.append(
                ContractParseIssue(
                    code=ContractIssueCode.CONFLICTING_REQUIREMENT,
                    field=field,
                    message=f"question contains conflicting {field} requirements: {rendered}",
                )
            )
            return None
        return distinct_values[0] if distinct_values else None

    @staticmethod
    def _has_conflict(issues: list[ContractParseIssue], field: str) -> bool:
        return any(
            issue.code is ContractIssueCode.CONFLICTING_REQUIREMENT and issue.field == field
            for issue in issues
        )
