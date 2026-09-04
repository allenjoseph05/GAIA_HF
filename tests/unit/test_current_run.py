from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from gaia_max.artifacts import ArtifactStore
from gaia_max.attachment_resolution import OfficialAttachmentResolver
from gaia_max.attachment_validation import AttachmentValidator
from gaia_max.clients import GaiaApiClient, GaiaGatedDatasetClient
from gaia_max.config import Settings
from gaia_max.current_run import (
    PrivateCandidateRecord,
    PrivateCandidateStore,
    PrivateOutputContractResolution,
    preflight_current_candidate,
    resolve_current_attachment,
    resolve_private_output_contract,
)
from gaia_max.domain import (
    AnswerType,
    Modality,
    OutputContract,
    Question,
    RiskLevel,
    StringAnswer,
    TaskClass,
)
from gaia_max.gated_attachment_resolution import GatedAttachmentResolver
from gaia_max.profiles import TaskProfileRecord, TaskProfileRegistry
from gaia_max.snapshot import StoredQuestionSnapshot, build_question_snapshot


def test_private_candidate_store_round_trip(tmp_path: Path) -> None:
    record = PrivateCandidateRecord(
        task_id="task-1",
        snapshot_sha256="a" * 64,
        question_sha256="b" * 64,
        semantic_answer=StringAnswer(value="private value"),
        verification_checks=("round_trip",),
        verification_passed=True,
        recorded_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    store = PrivateCandidateStore(tmp_path)

    path = store.put(record)

    assert path.is_file()
    assert store.get("task-1") == record


def test_private_candidate_store_missing_record(tmp_path: Path) -> None:
    assert PrivateCandidateStore(tmp_path).get("missing") is None


def test_private_resolution_only_fills_ambiguous_currency_symbol_policy() -> None:
    contract = OutputContract(
        answer_type=AnswerType.CURRENCY,
        decimal_places=2,
        include_currency_symbol=None,
    )

    resolved = resolve_private_output_contract(
        contract,
        PrivateOutputContractResolution(
            include_currency_symbol=False,
            thousands_separator=True,
            rationale="The question does not request symbol punctuation.",
        ),
    )

    assert contract.include_currency_symbol is None
    assert resolved.include_currency_symbol is False
    assert contract.thousands_separator is False
    assert resolved.thousands_separator is True


def test_private_resolution_cannot_replace_explicit_public_contract() -> None:
    contract = OutputContract(
        answer_type=AnswerType.CURRENCY,
        decimal_places=2,
        include_currency_symbol=True,
    )

    with pytest.raises(ValueError, match="cannot replace"):
        resolve_private_output_contract(
            contract,
            PrivateOutputContractResolution(
                include_currency_symbol=False,
                rationale="Synthetic conflicting decision.",
            ),
        )


def test_current_preflight_stays_redacted_and_refuses_partial_candidate(
    tmp_path: Path,
) -> None:
    question = Question(task_id="task-1", question="Synthetic question")
    snapshot = build_question_snapshot(
        [question],
        source_url="https://course.test/questions",
        task_profiles_version="test-v1",
        retrieved_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True)
    (snapshots_dir / f"questions-{snapshot.snapshot_sha256}.json").write_text(
        StoredQuestionSnapshot(snapshot=snapshot, questions=[question]).model_dump_json(),
        encoding="utf-8",
    )
    profile = TaskProfileRecord(
        task_id=question.task_id,
        question_sha256=snapshot.task_hashes[question.task_id],
        modality=Modality.WEB,
        task_class=TaskClass.DEEP_RESEARCH,
        route="research",
        risk_level=RiskLevel.LOW,
        requested_operation="answer",
        output_contract=OutputContract(answer_type=AnswerType.STRING),
        verification_policy="synthetic",
    )
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        TaskProfileRegistry(
            schema_version=1,
            profiles_version="test-v1",
            question_snapshot_sha256=snapshot.snapshot_sha256,
            profile_count=1,
            profiles=[profile],
        ).model_dump_json(),
        encoding="utf-8",
    )

    report = preflight_current_candidate(
        Settings(
            runs_dir=tmp_path,
            artifacts_dir=tmp_path / "artifacts",
            task_profiles_path=profiles_path,
            task_profiles_version="test-v1",
            expected_question_count=1,
        )
    )
    encoded = report.model_dump_json()

    assert report.valid is False
    assert report.ready_count == 0
    assert report.blocked_count == 1
    assert report.private_manifest_path is None
    assert "submitted_answer" not in encoded
    assert not (tmp_path / "current" / "frozen-candidate.json").exists()


@pytest.mark.asyncio
async def test_current_attachment_handoff_invokes_gated_resolver(
    tmp_path: Path,
) -> None:
    task_id = "f918266a-b3e0-4914-865d-4faa564f1aef"
    question = Question(
        task_id=task_id,
        question="Inspect the attached Python file.",
        file_name=f"{task_id}.py",
    )
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        gaia_api_url="https://course.test",
        hf_token="hf_synthetic_token",
        attachment_max_attempts=1,
    )
    official_http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(503)),
        base_url="https://course.test",
    )
    gated_http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                content=b"print(42)\n",
                headers={"content-type": "text/x-python"},
            )
        ),
        base_url="https://huggingface.test",
    )
    store = ArtifactStore(tmp_path / "artifacts")
    validator = AttachmentValidator(max_bytes=settings.max_attachment_bytes)
    async with official_http, gated_http:
        item = await resolve_current_attachment(
            question,
            official_resolver=OfficialAttachmentResolver(
                client=GaiaApiClient(settings, http_client=official_http),
                store=store,
                validator=validator,
                settings=settings,
            ),
            gated_resolver=GatedAttachmentResolver(
                client=GaiaGatedDatasetClient(settings, http_client=gated_http),
                store=store,
                validator=validator,
                settings=settings,
            ),
        )

    assert item.ready is True
    assert item.artifact_id is not None
    source_locator = store.metadata(item.artifact_id).origins[0].source_locator
    assert source_locator is not None
    assert source_locator.startswith(
        "hf://datasets/gaia-benchmark/GAIA@"
    )
