"""Pure, deterministic conversion from typed meaning to exact answer text."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from typing import ClassVar, Never

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


class AnswerSerializationError(ValueError):
    """Base error for a semantic answer that cannot satisfy its contract."""


class IncompatibleSemanticAnswerError(AnswerSerializationError):
    """The semantic answer type does not match the output contract."""


class IncompleteOutputContractError(AnswerSerializationError):
    """The contract omits a decision required for exact serialization."""


class InvalidListAnswerError(AnswerSerializationError):
    """List contents cannot satisfy their count, scope, or sorting contract."""


class AnswerSerializer:
    """Serialize typed answers without generative rewriting or global cleanup."""

    _ENTITY_TYPES: ClassVar[frozenset[AnswerType]] = frozenset(
        {
            AnswerType.NAME,
            AnswerType.FIRST_NAME,
            AnswerType.SURNAME,
            AnswerType.CITY,
            AnswerType.IOC_CODE,
        }
    )

    def serialize(self, answer: SemanticAnswer, contract: OutputContract) -> str:
        """Return only the exact answer text dictated by ``contract``."""

        answer_type = contract.answer_type
        if answer_type is AnswerType.INTEGER:
            rendered = self._integer(answer, contract)
        elif answer_type is AnswerType.DECIMAL:
            rendered = self._decimal(answer, contract)
        elif answer_type is AnswerType.CURRENCY:
            rendered = self._currency(answer, contract)
        elif answer_type in {AnswerType.STRING, AnswerType.EXACT_QUOTE}:
            rendered = self._string(answer)
        elif answer_type in self._ENTITY_TYPES:
            rendered = self._entity(answer, contract)
        elif answer_type is AnswerType.LIST:
            rendered = self._list(answer, contract)
        elif answer_type is AnswerType.CHESS_SAN:
            rendered = self._chess(answer, contract)
        else:  # pragma: no cover - exhaustive guard for future enum additions
            raise IncompleteOutputContractError(
                f"answer type {answer_type!s} has no deterministic serializer"
            )

        if contract.trailing_punctuation and not rendered.endswith("."):
            return f"{rendered}."
        return rendered

    def _integer(self, answer: SemanticAnswer, contract: OutputContract) -> str:
        if not isinstance(answer, IntegerAnswer):
            self._raise_incompatible(answer, contract)
        self._require_units_omitted(contract)
        return f"{answer.value:,}" if contract.thousands_separator else str(answer.value)

    def _decimal(self, answer: SemanticAnswer, contract: OutputContract) -> str:
        if not isinstance(answer, DecimalAnswer):
            self._raise_incompatible(answer, contract)
        self._require_units_omitted(contract)
        return self._format_decimal(
            answer.value,
            decimal_places=contract.decimal_places,
            thousands_separator=contract.thousands_separator,
        )

    def _currency(self, answer: SemanticAnswer, contract: OutputContract) -> str:
        if not isinstance(answer, DecimalAnswer):
            self._raise_incompatible(answer, contract)
        if contract.decimal_places is None:
            raise IncompleteOutputContractError(
                "currency serialization requires an explicit decimal_places value"
            )
        if contract.include_currency_symbol is None:
            raise IncompleteOutputContractError(
                "currency serialization requires an explicit currency-symbol decision"
            )
        if contract.units_policy not in {
            UnitsPolicy.OMIT,
            UnitsPolicy.QUESTION_SPECIFIC,
        }:
            raise IncompleteOutputContractError(
                "currency literal units are not represented by this output contract"
            )

        rendered = self._format_decimal(
            answer.value,
            decimal_places=contract.decimal_places,
            thousands_separator=contract.thousands_separator,
        )
        if not contract.include_currency_symbol:
            return rendered
        if rendered.startswith("-"):
            return f"-${rendered[1:]}"
        return f"${rendered}"

    @staticmethod
    def _string(answer: SemanticAnswer) -> str:
        if not isinstance(answer, StringAnswer):
            raise IncompatibleSemanticAnswerError(
                "string and exact-quote contracts require StringAnswer"
            )
        return answer.value

    def _entity(self, answer: SemanticAnswer, contract: OutputContract) -> str:
        if not isinstance(answer, EntityAnswer):
            self._raise_incompatible(answer, contract)
        field = self._entity_field(contract.answer_type, contract.name_scope)
        value = getattr(answer, field)
        if value is None:
            raise IncompatibleSemanticAnswerError(
                f"entity answer does not contain required field {field!r}"
            )
        return value

    def _list(self, answer: SemanticAnswer, contract: OutputContract) -> str:
        if not isinstance(answer, ListAnswer):
            self._raise_incompatible(answer, contract)
        if not contract.delimiter:
            raise IncompleteOutputContractError("list delimiter cannot be empty")
        if (
            contract.expected_item_count is not None
            and len(answer.items) != contract.expected_item_count
        ):
            raise InvalidListAnswerError(
                "list item count does not match output contract: "
                f"expected {contract.expected_item_count}, received {len(answer.items)}"
            )

        rendered_items = [self._list_item(item, contract.name_scope) for item in answer.items]
        if contract.sort is SortPolicy.ALPHABETICAL:
            text_items: list[str] = []
            for item in rendered_items:
                if not isinstance(item, str):
                    raise InvalidListAnswerError(
                        "alphabetical sorting requires text or entity list items"
                    )
                text_items.append(item)
            text_items.sort(key=lambda item: (item.casefold(), item))
            return contract.delimiter.join(text_items)
        elif contract.sort is SortPolicy.NUMERIC_ASCENDING:
            numeric_items: list[int] = []
            for item in rendered_items:
                if not isinstance(item, int) or isinstance(item, bool):
                    raise InvalidListAnswerError(
                        "numeric sorting requires integer list items"
                    )
                numeric_items.append(item)
            numeric_items.sort()
            return contract.delimiter.join(str(item) for item in numeric_items)
        elif contract.sort not in {SortPolicy.NONE, SortPolicy.QUESTION_ORDER}:
            raise IncompleteOutputContractError(
                f"sort policy {contract.sort!s} has no deterministic implementation"
            )
        return contract.delimiter.join(str(item) for item in rendered_items)

    @classmethod
    def _list_item(cls, item: str | int | EntityAnswer, scope: NameScope) -> str | int:
        if isinstance(item, EntityAnswer):
            field = cls._entity_field(AnswerType.NAME, scope)
            value = getattr(item, field)
            if value is None:
                raise InvalidListAnswerError(
                    f"entity list item does not contain required field {field!r}"
                )
            return value
        if scope is not NameScope.UNCHANGED:
            raise InvalidListAnswerError(
                "a scoped name list requires EntityAnswer items with separate name fields"
            )
        return item

    @staticmethod
    def _chess(answer: SemanticAnswer, contract: OutputContract) -> str:
        if not isinstance(answer, ChessMoveAnswer):
            raise IncompatibleSemanticAnswerError(
                "chess SAN contracts require ChessMoveAnswer"
            )
        if contract.notation is not None and contract.notation.casefold() not in {
            "algebraic",
            "san",
            "standard algebraic notation",
        }:
            raise IncompleteOutputContractError(
                f"unsupported chess notation {contract.notation!r}"
            )
        return answer.san

    @staticmethod
    def _entity_field(answer_type: AnswerType, scope: NameScope) -> str:
        if answer_type is AnswerType.FIRST_NAME:
            if scope not in {NameScope.UNCHANGED, NameScope.FIRST}:
                raise IncompleteOutputContractError(
                    "first-name answer type conflicts with contract name scope"
                )
            return "first_name"
        if answer_type is AnswerType.SURNAME:
            if scope not in {NameScope.UNCHANGED, NameScope.SURNAME}:
                raise IncompleteOutputContractError(
                    "surname answer type conflicts with contract name scope"
                )
            return "surname"
        if answer_type is AnswerType.CITY:
            if scope is not NameScope.UNCHANGED:
                raise IncompleteOutputContractError(
                    "city answer type conflicts with contract name scope"
                )
            return "city"
        if answer_type is AnswerType.IOC_CODE:
            if scope is not NameScope.UNCHANGED:
                raise IncompleteOutputContractError(
                    "IOC-code answer type conflicts with contract name scope"
                )
            return "ioc_code"
        if scope is NameScope.FIRST:
            return "first_name"
        if scope is NameScope.SURNAME:
            return "surname"
        return "full_name"

    @staticmethod
    def _format_decimal(
        value: Decimal,
        *,
        decimal_places: int | None,
        thousands_separator: bool,
    ) -> str:
        normalized = value
        if decimal_places is not None:
            quantum = Decimal(1).scaleb(-decimal_places)
            integer_digits = max(value.adjusted() + 1, 1)
            required_precision = max(50, integer_digits + decimal_places + 5)
            try:
                with localcontext() as context:
                    context.prec = required_precision
                    normalized = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
            except InvalidOperation as exc:  # pragma: no cover - defensive bound
                raise AnswerSerializationError("decimal value cannot be quantized") from exc
        if normalized.is_zero():
            normalized = abs(normalized)
        grouping = "," if thousands_separator else ""
        precision = f".{decimal_places}" if decimal_places is not None else ""
        return format(normalized, f"{grouping}{precision}f")

    @staticmethod
    def _require_units_omitted(contract: OutputContract) -> None:
        if contract.units_policy is not UnitsPolicy.OMIT:
            raise IncompleteOutputContractError(
                "non-currency numeric units require a literal not represented by the contract"
            )

    @staticmethod
    def _raise_incompatible(answer: SemanticAnswer, contract: OutputContract) -> Never:
        raise IncompatibleSemanticAnswerError(
            f"{contract.answer_type.value} contract is incompatible with {answer.kind} answer"
        )


def serialize_answer(answer: SemanticAnswer, contract: OutputContract) -> str:
    """Convenience entry point for graph nodes and specialist solvers."""

    return AnswerSerializer().serialize(answer, contract)
