"""Core domain models shared by graph nodes and specialist solvers."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class Modality(StrEnum):
    TEXT = "text"
    WEB = "web"
    IMAGE = "image"
    AUDIO = "audio"
    PYTHON = "python"
    XLSX = "xlsx"
    YOUTUBE_AUDIO = "youtube_audio"
    YOUTUBE_VISUAL = "youtube_visual"


class TaskClass(StrEnum):
    UNKNOWN = "unknown"
    DETERMINISTIC_TEXT = "deterministic_text"
    OPERATION_TABLE = "operation_table"
    STRUCTURED_TABLE = "structured_table"
    CODE = "code"
    SPREADSHEET = "spreadsheet"
    AUDIO_INGREDIENT = "audio_ingredient"
    AUDIO_NUMERIC = "audio_numeric"
    YOUTUBE_SPEECH = "youtube_speech"
    YOUTUBE_VISUAL = "youtube_visual"
    CHESS = "chess"
    HISTORICAL_WIKIPEDIA = "historical_wikipedia"
    HISTORICAL_WEB = "historical_web"
    SCHOLARLY = "scholarly"
    CROSSLINGUAL = "crosslingual"
    DEEP_RESEARCH = "deep_research"


class AnswerType(StrEnum):
    INTEGER = "integer"
    DECIMAL = "decimal"
    STRING = "string"
    NAME = "name"
    FIRST_NAME = "first_name"
    SURNAME = "surname"
    CITY = "city"
    IOC_CODE = "ioc_code"
    LIST = "list"
    CURRENCY = "currency"
    EXACT_QUOTE = "exact_quote"
    CHESS_SAN = "chess_san"


class SortPolicy(StrEnum):
    NONE = "none"
    ALPHABETICAL = "alphabetical"
    NUMERIC_ASCENDING = "numeric_ascending"
    QUESTION_ORDER = "question_order"


class UnitsPolicy(StrEnum):
    OMIT = "omit"
    LITERAL_REQUIRED = "literal_required"
    QUESTION_SPECIFIC = "question_specific"


class NameScope(StrEnum):
    FULL = "full"
    FIRST = "first"
    SURNAME = "surname"
    UNCHANGED = "unchanged"


class EvidenceStatus(StrEnum):
    NOT_FOUND = "not_found"
    INDIRECT = "indirect"
    PRIMARY = "primary"
    INDEPENDENTLY_VERIFIED = "independently_verified"


class ArtifactSource(StrEnum):
    OFFICIAL_API = "official_api"
    OFFICIAL_GATED_DATASET = "official_gated_dataset"
    WEB_RETRIEVAL = "web_retrieval"
    WEB_ARCHIVE = "web_archive"
    DERIVED = "derived"
    FIXTURE = "fixture"


class Verdict(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


class TaskStatus(StrEnum):
    NEW = "new"
    ACQUIRING = "acquiring"
    ANALYZING = "analyzing"
    SOLVING = "solving"
    VERIFYING = "verifying"
    RETRYING = "retrying"
    ACCEPTED = "accepted"
    SERIALIZED = "serialized"
    READY = "ready"
    BLOCKED = "blocked"


class Question(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    level: str | None = Field(default=None, validation_alias=AliasChoices("level", "Level"))
    file_name: str | None = None

    @model_validator(mode="after")
    def normalize_empty_file_name(self) -> Question:
        if self.file_name == "":
            self.file_name = None
        return self


class QuestionSnapshot(BaseModel):
    retrieved_at: datetime
    source_url: HttpUrl
    count: int = Field(ge=1)
    task_ids: list[str] = Field(min_length=1)
    task_hashes: dict[str, str]
    snapshot_sha256: str = Field(pattern=SHA256_PATTERN)
    attachment_task_ids: list[str] = Field(default_factory=list)
    task_profiles_version: str = Field(min_length=1)

    @field_serializer("source_url")
    def serialize_source_url(self, source_url: HttpUrl) -> str:
        """Keep snapshots JSON/msgpack portable across checkpoint backends."""

        return str(source_url)

    @model_validator(mode="after")
    def validate_inventory(self) -> QuestionSnapshot:
        if self.count != len(self.task_ids):
            raise ValueError("snapshot count must equal the number of task IDs")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("snapshot task IDs must be unique")
        if set(self.task_hashes) != set(self.task_ids):
            raise ValueError("task hashes must exist for exactly the snapshot task IDs")
        if not set(self.attachment_task_ids).issubset(self.task_ids):
            raise ValueError("attachment task IDs must be present in the snapshot")
        for task_hash in self.task_hashes.values():
            if len(task_hash) != 64 or any(c not in "0123456789abcdef" for c in task_hash):
                raise ValueError("every task hash must be a lowercase SHA-256 value")
        return self


class ArtifactRef(BaseModel):
    artifact_id: str = Field(pattern=SHA256_PATTERN)
    sha256: str = Field(pattern=SHA256_PATTERN)
    original_name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    local_path: Path
    source: ArtifactSource
    retrieved_at: datetime


class TemporalConstraint(BaseModel):
    kind: Literal[
        "revision_cutoff",
        "as_of",
        "publication_date",
        "event_period",
        "compiled_date",
    ]
    start: datetime | None = None
    end: datetime | None = None
    timezone: str = "UTC"

    @model_validator(mode="after")
    def validate_interval(self) -> TemporalConstraint:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("temporal constraint start cannot be after end")
        return self


class OutputContract(BaseModel):
    answer_type: AnswerType
    sort: SortPolicy = SortPolicy.NONE
    delimiter: str = ", "
    decimal_places: int | None = Field(default=None, ge=0, le=12)
    units_policy: UnitsPolicy = UnitsPolicy.OMIT
    include_currency_symbol: bool | None = None
    thousands_separator: bool = False
    name_scope: NameScope = NameScope.UNCHANGED
    notation: str | None = None
    trailing_punctuation: bool = False
    expected_item_count: int | None = Field(default=None, ge=0)
    allow_duplicates: bool = False


class TaskProfile(BaseModel):
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    question_sha256: str = Field(pattern=SHA256_PATTERN)
    file_name: str | None = None
    modality: Modality
    task_class: TaskClass
    route: str = Field(min_length=1)
    risk_level: RiskLevel
    temporal_constraint: TemporalConstraint | None = None
    requested_operation: str = Field(min_length=1)
    filters: dict[str, object] = Field(default_factory=dict)
    required_sources: list[str] = Field(default_factory=list)
    output_contract: OutputContract
    verification_policy: str = Field(min_length=1)
    risk_flags: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    evidence_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    source_url: HttpUrl | None = None
    artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    source_type: str = Field(min_length=1)
    source_date: str | None = None
    snapshot_date: str | None = None
    retrieved_at: datetime
    is_primary_source: bool
    excerpt: str | None = None
    locator: str | None = None
    extraction_method: str = Field(min_length=1)
    status: EvidenceStatus

    @model_validator(mode="after")
    def validate_location(self) -> Evidence:
        if self.source_url is None and self.artifact_id is None:
            raise ValueError("evidence must reference a source URL or artifact")
        return self


class IntegerAnswer(BaseModel):
    kind: Literal["integer"] = "integer"
    value: StrictInt


class DecimalAnswer(BaseModel):
    kind: Literal["decimal"] = "decimal"
    value: Decimal

    @field_validator("value", mode="before")
    @classmethod
    def reject_binary_float_and_boolean(cls, value: object) -> object:
        if isinstance(value, float | bool):
            raise ValueError("decimal answers must not originate from binary floats or booleans")
        return value

    @field_validator("value")
    @classmethod
    def require_finite_decimal(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("decimal answers must be finite")
        return value


class StringAnswer(BaseModel):
    kind: Literal["string"] = "string"
    value: StrictStr = Field(min_length=1)


class EntityAnswer(BaseModel):
    kind: Literal["entity"] = "entity"
    full_name: StrictStr | None = Field(default=None, min_length=1)
    first_name: StrictStr | None = Field(default=None, min_length=1)
    surname: StrictStr | None = Field(default=None, min_length=1)
    city: StrictStr | None = Field(default=None, min_length=1)
    ioc_code: StrictStr | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_entity_field(self) -> EntityAnswer:
        if not any((self.full_name, self.first_name, self.surname, self.city, self.ioc_code)):
            raise ValueError("an entity answer requires at least one populated field")
        return self


class ListAnswer(BaseModel):
    kind: Literal["list"] = "list"
    items: list[StrictStr | StrictInt | EntityAnswer] = Field(min_length=1)


class ChessMoveAnswer(BaseModel):
    kind: Literal["chess_move"] = "chess_move"
    uci: StrictStr = Field(min_length=4, max_length=5)
    san: StrictStr = Field(min_length=2)


SemanticAnswer = Annotated[
    IntegerAnswer | DecimalAnswer | StringAnswer | EntityAnswer | ListAnswer | ChessMoveAnswer,
    Field(discriminator="kind"),
]


class CandidateAnswer(BaseModel):
    candidate_id: str = Field(min_length=1)
    semantic_value: SemanticAnswer
    solver: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    deterministic_checks: list[str] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)
    created_at: datetime


class VerificationResult(BaseModel):
    verifier: str = Field(min_length=1)
    tier: int = Field(ge=0, le=5)
    verdict: Verdict
    passed_checks: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    critique: str = ""
    missing_evidence: list[str] = Field(default_factory=list)
    evidence_conflicts: list[str] = Field(default_factory=list)
    requested_next_action: str | None = None


class RiskAssessment(BaseModel):
    level: RiskLevel
    positive_features: list[str] = Field(default_factory=list)
    negative_features: list[str] = Field(default_factory=list)
    hard_failures: list[str] = Field(default_factory=list)


class TaskResult(BaseModel):
    task_id: str = Field(min_length=1)
    status: TaskStatus
    selected_candidate_id: str | None = None
    semantic_answer: SemanticAnswer | None = None
    serialized_answer: str | None = None
    risk: RiskAssessment | None = None
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> TaskResult:
        if self.status is TaskStatus.READY:
            if self.semantic_answer is None or not self.serialized_answer:
                raise ValueError("a ready task requires semantic and serialized answers")
            if self.risk is not None and self.risk.hard_failures:
                raise ValueError("a ready task cannot contain hard risk failures")
        if self.status is TaskStatus.BLOCKED and not self.errors:
            raise ValueError("a blocked task requires at least one error reason")
        return self
