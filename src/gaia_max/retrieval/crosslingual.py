"""Cross-lingual query planning and stable-identity resolution utilities."""

from __future__ import annotations

from datetime import date

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StrictStr,
    field_serializer,
    model_validator,
)

from gaia_max.domain.models import SHA256_PATTERN
from gaia_max.retrieval.providers import SearchRequest
from gaia_max.retrieval.search import formulate_search_request


class CrossLingualResolutionError(ValueError):
    """Entity evidence is ambiguous, incomplete, or lacks source-backed naming."""


class EntityRole(BaseModel):
    """One source-backed entity-to-work/role relationship."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    work: StrictStr = Field(min_length=1, max_length=500)
    role: StrictStr = Field(min_length=1, max_length=500)
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> EntityRole:
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("entity role start cannot follow end")
        return self


class CrossLingualEntityRecord(BaseModel):
    """One authoritative source's stable entity facts and original spellings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stable_id: StrictStr = Field(min_length=1, max_length=300)
    source_url: HttpUrl | None = None
    artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    source_language: StrictStr = Field(min_length=2, max_length=35)
    native_label: StrictStr = Field(min_length=1, max_length=500)
    romanized_label: StrictStr | None = Field(default=None, min_length=1, max_length=500)
    given_name: StrictStr | None = Field(default=None, min_length=1, max_length=200)
    surname: StrictStr | None = Field(default=None, min_length=1, max_length=200)
    occupations: tuple[str, ...] = ()
    roles: tuple[EntityRole, ...] = ()
    nationality: StrictStr | None = Field(default=None, min_length=1, max_length=200)
    valid_from: date | None = None
    valid_to: date | None = None

    @model_validator(mode="after")
    def validate_provenance_and_dates(self) -> CrossLingualEntityRecord:
        if self.source_url is None and self.artifact_id is None:
            raise ValueError("cross-lingual entity record requires source provenance")
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("entity validity start cannot follow end")
        if self.romanized_label is not None and not (self.given_name or self.surname):
            raise ValueError(
                "source-backed Romanization requires explicit name components"
            )
        return self

    @field_serializer("source_url")
    def serialize_url(self, value: HttpUrl | None) -> str | None:
        return str(value) if value is not None else None


class EntityResolutionRequest(BaseModel):
    """Disambiguating facts that must be satisfied by one stable entity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: tuple[str, ...] = Field(min_length=1)
    occupation: str | None = None
    work: str | None = None
    role: str | None = None
    as_of: date | None = None
    nationality: str | None = None

    @model_validator(mode="after")
    def require_disambiguator(self) -> EntityResolutionRequest:
        if not any((self.occupation, self.work, self.role, self.as_of, self.nationality)):
            raise ValueError("entity resolution requires a disambiguating field")
        if (self.work is None) != (self.role is None):
            raise ValueError("entity work and role constraints must be supplied together")
        return self


class ResolvedEntity(BaseModel):
    """Unique stable identity plus every agreeing source record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stable_id: StrictStr = Field(min_length=1)
    records: tuple[CrossLingualEntityRecord, ...] = Field(min_length=1)

    def romanized_name(self) -> str:
        values = {
            record.romanized_label
            for record in self.records
            if record.romanized_label is not None
        }
        if len(values) != 1:
            raise CrossLingualResolutionError(
                "Romanized output requires one agreeing source-backed spelling"
            )
        return next(iter(values))

    def first_name(self) -> str:
        return self._unique_component("given_name")

    def family_name(self) -> str:
        return self._unique_component("surname")

    def _unique_component(self, field: str) -> str:
        values = {
            value
            for record in self.records
            if (value := getattr(record, field)) is not None
        }
        if len(values) != 1:
            raise CrossLingualResolutionError(
                f"entity output requires one agreeing source-backed {field}"
            )
        return next(iter(values))


class CrossLingualQueryPlan(BaseModel):
    """English and native-language queries kept distinct and auditable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requests: tuple[SearchRequest, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def require_language_diversity(self) -> CrossLingualQueryPlan:
        languages = {request.language for request in self.requests}
        if None in languages or len(languages) < 2:
            raise ValueError("cross-lingual query plan requires distinct language tags")
        return self


def build_crosslingual_query_plan(
    *,
    english_terms: str,
    native_terms: str,
    native_language: str,
    exact_phrases: tuple[str, ...] = (),
    site_domains: tuple[str, ...] = (),
) -> CrossLingualQueryPlan:
    """Build separate English/native searches without translating away original text."""

    if native_language.casefold().startswith("en"):
        raise ValueError("native language must differ from English")
    return CrossLingualQueryPlan(
        requests=(
            formulate_search_request(
                english_terms,
                exact_phrases=exact_phrases,
                language="en",
                site_domains=site_domains,
            ),
            formulate_search_request(
                native_terms,
                exact_phrases=exact_phrases,
                language=native_language,
                site_domains=site_domains,
            ),
        )
    )


def resolve_unique_entity(
    records: tuple[CrossLingualEntityRecord, ...],
    request: EntityResolutionRequest,
) -> ResolvedEntity:
    """Filter by every disambiguator and require one stable identifier."""

    if not records:
        raise CrossLingualResolutionError("entity resolution has no source records")
    labels = {label.casefold() for label in request.labels}
    matching: list[CrossLingualEntityRecord] = []
    for record in records:
        record_labels = {record.native_label.casefold()}
        if record.romanized_label:
            record_labels.add(record.romanized_label.casefold())
        if record_labels.isdisjoint(labels):
            continue
        if request.occupation and not _contains(record.occupations, request.occupation):
            continue
        if request.nationality and (
            record.nationality is None
            or record.nationality.casefold() != request.nationality.casefold()
        ):
            continue
        if request.as_of and not _date_valid(record, request.as_of):
            continue
        if request.work and request.role and not any(
            role.work.casefold() == request.work.casefold()
            and role.role.casefold() == request.role.casefold()
            and (request.as_of is None or _role_date_valid(role, request.as_of))
            for role in record.roles
        ):
            continue
        matching.append(record)
    identities = {record.stable_id for record in matching}
    if len(identities) != 1:
        raise CrossLingualResolutionError(
            "entity evidence must resolve to exactly one stable identifier"
        )
    stable_id = next(iter(identities))
    return ResolvedEntity(
        stable_id=stable_id,
        records=tuple(record for record in matching if record.stable_id == stable_id),
    )


def _contains(values: tuple[str, ...], expected: str) -> bool:
    return any(value.casefold() == expected.casefold() for value in values)


def _date_valid(record: CrossLingualEntityRecord, value: date) -> bool:
    return (record.valid_from is None or record.valid_from <= value) and (
        record.valid_to is None or value <= record.valid_to
    )


def _role_date_valid(role: EntityRole, value: date) -> bool:
    return (role.start_date is None or role.start_date <= value) and (
        role.end_date is None or value <= role.end_date
    )


__all__ = [
    "CrossLingualEntityRecord",
    "CrossLingualQueryPlan",
    "CrossLingualResolutionError",
    "EntityResolutionRequest",
    "EntityRole",
    "ResolvedEntity",
    "build_crosslingual_query_plan",
    "resolve_unique_entity",
]
