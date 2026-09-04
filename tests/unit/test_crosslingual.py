from datetime import date

import pytest
from pydantic import ValidationError

from gaia_max.retrieval import (
    CrossLingualEntityRecord,
    CrossLingualResolutionError,
    EntityResolutionRequest,
    EntityRole,
    build_crosslingual_query_plan,
    resolve_unique_entity,
)


def record(
    stable_id: str,
    *,
    native_label: str,
    romanized_label: str,
    given_name: str,
    surname: str,
    occupation: str = "actor",
    work: str = "Serial",
    role: str = "Character",
    language: str = "pl",
) -> CrossLingualEntityRecord:
    return CrossLingualEntityRecord.model_validate(
        {
            "stable_id": stable_id,
            "source_url": f"https://example.org/entity/{stable_id}",
            "source_language": language,
            "native_label": native_label,
            "romanized_label": romanized_label,
            "given_name": given_name,
            "surname": surname,
            "occupations": [occupation],
            "roles": [{"work": work, "role": role}],
        }
    )


def test_query_plan_preserves_separate_english_and_native_queries() -> None:
    plan = build_crosslingual_query_plan(
        english_terms="adaptation actor",
        native_terms="adaptacja aktor",
        native_language="pl-PL",
        exact_phrases=("named role",),
        site_domains=("example.pl",),
    )

    assert [request.language for request in plan.requests] == ["en", "pl-PL"]
    assert plan.requests[0].query.startswith("adaptation actor")
    assert plan.requests[1].query.startswith("adaptacja aktor")
    assert all(request.site_domains == ("example.pl",) for request in plan.requests)


def test_namesakes_remain_ambiguous_until_role_disambiguators_are_supplied() -> None:
    records = (
        record(
            "Q1",
            native_label="Jan Kowalski",
            romanized_label="Jan Kowalski",
            given_name="Jan",
            surname="Kowalski",
            work="First show",
            role="Doctor",
        ),
        record(
            "Q2",
            native_label="Jan Kowalski",
            romanized_label="Jan Kowalski",
            given_name="Jan",
            surname="Kowalski",
            work="Second show",
            role="Inspector",
        ),
    )
    with pytest.raises(ValidationError, match="disambiguating"):
        EntityResolutionRequest(labels=("Jan Kowalski",))

    resolved = resolve_unique_entity(
        records,
        EntityResolutionRequest(
            labels=("Jan Kowalski",),
            occupation="actor",
            work="Second show",
            role="Inspector",
        ),
    )

    assert resolved.stable_id == "Q2"
    assert resolved.first_name() == "Jan"


def test_dated_role_and_nationality_filter_historical_entity() -> None:
    old = CrossLingualEntityRecord.model_validate(
        {
            **record(
                "J1",
                native_label="山田 花子",
                romanized_label="Hanako Yamada",
                given_name="Hanako",
                surname="Yamada",
                work="National team",
                role="player",
                language="ja",
            ).model_dump(mode="json"),
            "nationality": "Japanese",
            "valid_from": "2018-01-01",
            "valid_to": "2020-12-31",
            "roles": [
                {
                    "work": "National team",
                    "role": "player",
                    "start_date": "2018-01-01",
                    "end_date": "2020-12-31",
                }
            ],
        }
    )
    current = record(
        "J2",
        native_label="山田 花子",
        romanized_label="Hanako Yamada",
        given_name="Hanako",
        surname="Yamada",
        work="National team",
        role="player",
        language="ja",
    )

    resolved = resolve_unique_entity(
        (old, current),
        EntityResolutionRequest(
            labels=("山田 花子", "Hanako Yamada"),
            work="National team",
            role="player",
            as_of=date(2019, 6, 1),
            nationality="Japanese",
        ),
    )

    assert resolved.stable_id == "J1"
    assert resolved.romanized_name() == "Hanako Yamada"
    assert resolved.family_name() == "Yamada"


def test_romanized_output_requires_source_backed_components_and_agreement() -> None:
    with pytest.raises(ValidationError, match="name components"):
        CrossLingualEntityRecord.model_validate(
            {
                "stable_id": "Q1",
                "source_url": "https://example.org/Q1",
                "source_language": "ja",
                "native_label": "山田 花子",
                "romanized_label": "Hanako Yamada",
                "occupations": ["player"],
            }
        )

    first = record(
        "Q1",
        native_label="山田 花子",
        romanized_label="Hanako Yamada",
        given_name="Hanako",
        surname="Yamada",
    )
    conflicting = record(
        "Q1",
        native_label="山田 花子",
        romanized_label="Yamada Hanako",
        given_name="Hanako",
        surname="Yamada",
    )
    resolved = resolve_unique_entity(
        (first, conflicting),
        EntityResolutionRequest(labels=("山田 花子",), occupation="actor"),
    )
    with pytest.raises(CrossLingualResolutionError, match="agreeing"):
        resolved.romanized_name()


def test_resolution_requires_exactly_one_stable_identity() -> None:
    records = (
        record(
            "Q1",
            native_label="Same Name",
            romanized_label="Same Name",
            given_name="Same",
            surname="Name",
        ),
        record(
            "Q2",
            native_label="Same Name",
            romanized_label="Same Name",
            given_name="Same",
            surname="Name",
        ),
    )
    request = EntityResolutionRequest(labels=("Same Name",), occupation="actor")
    with pytest.raises(CrossLingualResolutionError, match="exactly one"):
        resolve_unique_entity(records, request)


def test_role_dates_are_validated() -> None:
    with pytest.raises(ValidationError, match="start cannot follow"):
        EntityRole(
            work="Show",
            role="Role",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 1),
        )
