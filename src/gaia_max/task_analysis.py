"""Structured, provider-neutral task analysis with deterministic reconciliation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Protocol, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from gaia_max.domain import (
    Modality,
    OutputContract,
    Question,
    TaskClass,
    TaskProfile,
    TemporalConstraint,
)
from gaia_max.output_contracts import (
    ContractParseStatus,
    OutputContractParser,
    OutputContractParseResult,
)
from gaia_max.snapshot import question_sha256

FORBIDDEN_ANALYSIS_KEYS = {
    "answer",
    "answer_value",
    "candidate_answer",
    "final answer",
    "final_answer",
    "gold_answer",
    "semantic_answer",
    "serialized_answer",
    "submitted_answer",
}


class TaskAnalysisError(RuntimeError):
    """Base error for structured task analysis."""


class StructuredAnalysisResponseError(TaskAnalysisError):
    """A model backend failed or returned invalid structured data."""


class TaskAnalysisUnavailableError(TaskAnalysisError):
    """No trusted profile or configured structured model can analyze the task."""


class HardContractConflictError(TaskAnalysisError):
    """Deterministic output requirements conflict and cannot be model-overridden."""


class ProfileAnalysisMismatchError(TaskAnalysisError):
    """A supplied profile is not bound to the exact runtime question."""


class LowConfidenceTaskAnalysisError(TaskAnalysisError):
    """An unknown task received only a low-confidence model proposal."""


class IncompleteTaskAnalysisError(TaskAnalysisError):
    """The reconciled analysis is missing a required field."""


class AnalysisAuthority(StrEnum):
    """Sources ordered from strongest deterministic authority to weakest proposal."""

    HARD_RULE = "hard_rule"
    PROFILE = "profile"
    MODEL = "model"
    POLICY = "policy"


class TaskAnalysisRequest(BaseModel):
    """Answer-free public context sent to a structured analysis backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    file_name: str | None = None
    deterministic_contract_status: ContractParseStatus
    deterministic_contract: OutputContract | None = None
    contract_issue_codes: tuple[str, ...] = ()


class TaskAnalysisProposal(BaseModel):
    """Bounded model proposal; it has no route, evidence, or candidate-answer field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    modality: Modality
    task_class: TaskClass
    requested_operation: str = Field(min_length=1, max_length=500)
    temporal_constraint: TemporalConstraint | None = None
    filters: dict[str, JsonValue] = Field(default_factory=dict, max_length=32)
    output_contract: OutputContract | None = None
    risk_flags: tuple[str, ...] = Field(default=(), max_length=32)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def reject_answer_bearing_keys(cls, value: object) -> object:
        _reject_forbidden_keys(value)
        return value

    @field_validator("risk_flags")
    @classmethod
    def validate_risk_flags(cls, flags: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(flags)) != len(flags):
            raise ValueError("risk flags must be unique")
        for flag in flags:
            if not flag or len(flag) > 64 or not all(
                character.islower() or character.isdigit() or character in "_-"
                for character in flag
            ):
                raise ValueError("risk flags must be lowercase identifier strings")
        return flags


class StructuredAnalysisBackend(Protocol):
    """Minimal interface implemented later by local or hosted structured models."""

    @property
    def name(self) -> str: ...

    async def generate(self, request: TaskAnalysisRequest) -> Mapping[str, object]: ...


class StructuredTaskAnalysisAdapter:
    """Validate a backend's untrusted JSON-like response before reconciliation."""

    def __init__(self, backend: StructuredAnalysisBackend) -> None:
        provider_name = backend.name.strip()
        if (
            not provider_name
            or len(provider_name) > 100
            or not all(
                character.isalnum() or character in "._-/"
                for character in provider_name
            )
        ):
            raise ValueError("structured analysis backend name is invalid")
        self._backend = backend
        self._provider_name = provider_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    async def propose(self, request: TaskAnalysisRequest) -> TaskAnalysisProposal:
        try:
            raw = await self._backend.generate(request)
        except Exception as exc:
            raise StructuredAnalysisResponseError(
                f"structured analysis backend {self.provider_name!r} failed"
            ) from exc
        if not isinstance(raw, Mapping):
            raise StructuredAnalysisResponseError(
                f"structured analysis backend {self.provider_name!r} returned a non-object"
            )
        try:
            return TaskAnalysisProposal.model_validate(dict(raw))
        except ValidationError as exc:
            raise StructuredAnalysisResponseError(
                f"structured analysis backend {self.provider_name!r} returned invalid schema"
            ) from exc


class AnalysisDisagreement(BaseModel):
    """A weaker source disagreed with the value retained by reconciliation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str = Field(min_length=1)
    authoritative_source: AnalysisAuthority
    conflicting_source: AnalysisAuthority
    authoritative_value: JsonValue
    conflicting_value: JsonValue
    blocks_submission: bool = True


class ReconciledTaskAnalysis(BaseModel):
    """Typed analysis used by later routing, planning, and verification nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    modality: Modality
    task_class: TaskClass
    requested_operation: str = Field(min_length=1)
    temporal_constraint: TemporalConstraint | None = None
    filters: dict[str, JsonValue] = Field(default_factory=dict)
    output_contract: OutputContract
    risk_flags: tuple[str, ...] = ()
    field_sources: dict[str, AnalysisAuthority]
    contract_parse_status: ContractParseStatus
    matched_contract_rules: tuple[str, ...] = ()
    disagreements: tuple[AnalysisDisagreement, ...] = ()
    profile_used: bool
    model_used: bool
    model_provider: str | None = None
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    requires_review: bool

    @model_validator(mode="after")
    def validate_model_metadata(self) -> ReconciledTaskAnalysis:
        if self.model_used != (self.model_provider is not None):
            raise ValueError("model_used must agree with model_provider presence")
        if self.model_used != (self.model_confidence is not None):
            raise ValueError("model_used must agree with model_confidence presence")
        if self.disagreements and not self.requires_review:
            raise ValueError("analysis disagreements require review")
        return self


class TaskAnalyzer:
    """Combine hard parsing, an exact profile, and an optional model proposal."""

    def __init__(
        self,
        *,
        contract_parser: OutputContractParser | None = None,
        model_adapter: StructuredTaskAnalysisAdapter | None = None,
        minimum_model_confidence: float = 0.7,
    ) -> None:
        if not 0 <= minimum_model_confidence <= 1:
            raise ValueError("minimum_model_confidence must be between zero and one")
        self._contract_parser = contract_parser or OutputContractParser()
        self._model_adapter = model_adapter
        self._minimum_model_confidence = minimum_model_confidence

    async def analyze(
        self,
        question: Question,
        *,
        profile: TaskProfile | None = None,
    ) -> ReconciledTaskAnalysis:
        """Return authoritative metadata while preserving every weaker disagreement."""

        contract_parse = self._contract_parser.parse(question.question)
        if contract_parse.status is ContractParseStatus.CONFLICT:
            raise HardContractConflictError(
                f"task {question.task_id} has conflicting deterministic output requirements"
            )
        if profile is not None:
            self._validate_profile(question, profile)
        if profile is None and self._model_adapter is None:
            raise TaskAnalysisUnavailableError(
                f"task {question.task_id} has neither a matching profile nor a model adapter"
            )

        proposal = await self._proposal(question, contract_parse)
        if profile is None:
            if proposal is None:
                raise TaskAnalysisUnavailableError("model proposal is unavailable")
            if proposal.confidence < self._minimum_model_confidence:
                raise LowConfidenceTaskAnalysisError(
                    f"task {question.task_id} model confidence is below policy threshold"
                )
            return self._reconcile_unknown(question, contract_parse, proposal)
        return self._reconcile_profiled(question, profile, contract_parse, proposal)

    async def _proposal(
        self,
        question: Question,
        contract_parse: OutputContractParseResult,
    ) -> TaskAnalysisProposal | None:
        if self._model_adapter is None:
            return None
        request = TaskAnalysisRequest(
            task_id=question.task_id,
            question=question.question,
            file_name=question.file_name,
            deterministic_contract_status=contract_parse.status,
            deterministic_contract=contract_parse.contract,
            contract_issue_codes=tuple(issue.code.value for issue in contract_parse.issues),
        )
        return await self._model_adapter.propose(request)

    def _reconcile_profiled(
        self,
        question: Question,
        profile: TaskProfile,
        contract_parse: OutputContractParseResult,
        proposal: TaskAnalysisProposal | None,
    ) -> ReconciledTaskAnalysis:
        disagreements: list[AnalysisDisagreement] = []
        output_contract = profile.output_contract
        output_source = AnalysisAuthority.PROFILE
        if contract_parse.contract is not None:
            output_contract = contract_parse.contract
            output_source = AnalysisAuthority.HARD_RULE
            self._record_disagreement(
                disagreements,
                "output_contract",
                output_source,
                AnalysisAuthority.PROFILE,
                output_contract,
                profile.output_contract,
            )

        if proposal is not None:
            self._record_disagreement(
                disagreements,
                "modality",
                AnalysisAuthority.PROFILE,
                AnalysisAuthority.MODEL,
                profile.modality,
                proposal.modality,
            )
            self._record_disagreement(
                disagreements,
                "task_class",
                AnalysisAuthority.PROFILE,
                AnalysisAuthority.MODEL,
                profile.task_class,
                proposal.task_class,
            )
            self._record_disagreement(
                disagreements,
                "requested_operation",
                AnalysisAuthority.PROFILE,
                AnalysisAuthority.MODEL,
                profile.requested_operation,
                proposal.requested_operation,
            )
            self._record_disagreement(
                disagreements,
                "temporal_constraint",
                AnalysisAuthority.PROFILE,
                AnalysisAuthority.MODEL,
                profile.temporal_constraint,
                proposal.temporal_constraint,
            )
            self._record_disagreement(
                disagreements,
                "filters",
                AnalysisAuthority.PROFILE,
                AnalysisAuthority.MODEL,
                profile.filters,
                proposal.filters,
            )
            if proposal.output_contract is not None:
                self._record_disagreement(
                    disagreements,
                    "output_contract",
                    output_source,
                    AnalysisAuthority.MODEL,
                    output_contract,
                    proposal.output_contract,
                )
            self._record_disagreement(
                disagreements,
                "risk_flags",
                AnalysisAuthority.PROFILE,
                AnalysisAuthority.MODEL,
                profile.risk_flags,
                proposal.risk_flags,
            )

        model_risks = proposal.risk_flags if proposal is not None else ()
        risk_flags = tuple(dict.fromkeys([*profile.risk_flags, *model_risks]))
        model_used = proposal is not None
        requires_review = (
            contract_parse.status is not ContractParseStatus.COMPLETE
            or bool(disagreements)
            or bool(
                proposal is not None
                and proposal.confidence < self._minimum_model_confidence
            )
        )
        return ReconciledTaskAnalysis(
            task_id=question.task_id,
            modality=profile.modality,
            task_class=profile.task_class,
            requested_operation=profile.requested_operation,
            temporal_constraint=profile.temporal_constraint,
            filters=cast(dict[str, JsonValue], profile.filters),
            output_contract=output_contract,
            risk_flags=risk_flags,
            field_sources={
                "modality": AnalysisAuthority.PROFILE,
                "task_class": AnalysisAuthority.PROFILE,
                "requested_operation": AnalysisAuthority.PROFILE,
                "temporal_constraint": AnalysisAuthority.PROFILE,
                "filters": AnalysisAuthority.PROFILE,
                "output_contract": output_source,
                "risk_flags": (
                    AnalysisAuthority.POLICY
                    if proposal is not None
                    else AnalysisAuthority.PROFILE
                ),
            },
            contract_parse_status=contract_parse.status,
            matched_contract_rules=contract_parse.matched_rules,
            disagreements=tuple(disagreements),
            profile_used=True,
            model_used=model_used,
            model_provider=(
                self._model_adapter.provider_name
                if proposal is not None and self._model_adapter is not None
                else None
            ),
            model_confidence=proposal.confidence if proposal is not None else None,
            requires_review=requires_review,
        )

    def _reconcile_unknown(
        self,
        question: Question,
        contract_parse: OutputContractParseResult,
        proposal: TaskAnalysisProposal,
    ) -> ReconciledTaskAnalysis:
        disagreements: list[AnalysisDisagreement] = []
        if contract_parse.contract is not None:
            output_contract = contract_parse.contract
            output_source = AnalysisAuthority.HARD_RULE
            if proposal.output_contract is not None:
                self._record_disagreement(
                    disagreements,
                    "output_contract",
                    output_source,
                    AnalysisAuthority.MODEL,
                    output_contract,
                    proposal.output_contract,
                )
        elif proposal.output_contract is not None:
            output_contract = proposal.output_contract
            output_source = AnalysisAuthority.MODEL
        else:
            raise IncompleteTaskAnalysisError(
                f"task {question.task_id} has no output contract from rules or model"
            )

        return ReconciledTaskAnalysis(
            task_id=question.task_id,
            modality=proposal.modality,
            task_class=proposal.task_class,
            requested_operation=proposal.requested_operation,
            temporal_constraint=proposal.temporal_constraint,
            filters=proposal.filters,
            output_contract=output_contract,
            risk_flags=proposal.risk_flags,
            field_sources={
                "modality": AnalysisAuthority.MODEL,
                "task_class": AnalysisAuthority.MODEL,
                "requested_operation": AnalysisAuthority.MODEL,
                "temporal_constraint": AnalysisAuthority.MODEL,
                "filters": AnalysisAuthority.MODEL,
                "output_contract": output_source,
                "risk_flags": AnalysisAuthority.MODEL,
            },
            contract_parse_status=contract_parse.status,
            matched_contract_rules=contract_parse.matched_rules,
            disagreements=tuple(disagreements),
            profile_used=False,
            model_used=True,
            model_provider=self._model_adapter.provider_name if self._model_adapter else None,
            model_confidence=proposal.confidence,
            requires_review=True,
        )

    @staticmethod
    def _validate_profile(question: Question, profile: TaskProfile) -> None:
        if (
            profile.task_id != question.task_id
            or profile.question != question.question
            or profile.file_name != question.file_name
            or profile.question_sha256 != question_sha256(question)
        ):
            raise ProfileAnalysisMismatchError(
                f"profile is not bound to runtime question {question.task_id}"
            )

    @staticmethod
    def _record_disagreement(
        disagreements: list[AnalysisDisagreement],
        field: str,
        authoritative_source: AnalysisAuthority,
        conflicting_source: AnalysisAuthority,
        authoritative_value: object,
        conflicting_value: object,
    ) -> None:
        authoritative_json = _to_json_value(authoritative_value)
        conflicting_json = _to_json_value(conflicting_value)
        if authoritative_json == conflicting_json:
            return
        disagreement = AnalysisDisagreement(
            field=field,
            authoritative_source=authoritative_source,
            conflicting_source=conflicting_source,
            authoritative_value=authoritative_json,
            conflicting_value=conflicting_json,
        )
        if disagreement not in disagreements:
            disagreements.append(disagreement)


def _to_json_value(value: object) -> JsonValue:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json_value(child) for child in value]
    raise TypeError(f"analysis metadata is not JSON-compatible: {type(value).__name__}")


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().casefold()
            if normalized in FORBIDDEN_ANALYSIS_KEYS:
                raise ValueError("model proposal contains a forbidden answer-bearing key")
            _reject_forbidden_keys(child)
    elif isinstance(value, list | tuple):
        for child in value:
            _reject_forbidden_keys(child)
