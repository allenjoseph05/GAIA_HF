"""Answer-redacted preparation and dry-run audit for the current evaluation."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from gaia_max.answer_validation import FinalAnswerValidator
from gaia_max.artifacts import ArtifactStore
from gaia_max.attachment_resolution import (
    OfficialAttachmentResolver,
    OfficialAttachmentStatus,
)
from gaia_max.attachment_validation import AttachmentValidator
from gaia_max.clients import (
    GaiaApiClient,
    GaiaGatedDatasetClient,
    GatedDatasetAccessError,
    GatedDatasetAttachmentNotFoundError,
    GatedDatasetError,
)
from gaia_max.config import Settings
from gaia_max.domain import (
    AnswerType,
    OutputContract,
    Question,
    SemanticAnswer,
    TaskResult,
    TaskStatus,
)
from gaia_max.gated_attachment_resolution import (
    GatedAttachmentRejectedError,
    GatedAttachmentResolver,
    GatedAttachmentRetriesExhaustedError,
)
from gaia_max.preflight import CandidatePreflight
from gaia_max.profiles import load_profile_registry, reconcile_profile_registry
from gaia_max.serialization import AnswerSerializer
from gaia_max.snapshot import StoredQuestionSnapshot


class AttachmentPreparationItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    ready: bool
    artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reason: str | None = None


class CurrentRunPreparationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_attachments: int = Field(ge=0)
    ready_attachments: int = Field(ge=0)
    items: tuple[AttachmentPreparationItem, ...]
    submission_capability_present: bool = False


class PrivateEvidenceRef(BaseModel):
    """A compact source reference stored only below the ignored runs directory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    authority: str = Field(min_length=1)
    extraction_method: str = Field(min_length=1)
    primary: bool = False


class PrivateOutputContractResolution(BaseModel):
    """A narrow operator decision for a contract detail the public wording leaves open."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    include_currency_symbol: bool
    thousands_separator: bool | None = None
    rationale: str = Field(min_length=1)


class PrivateCandidateRecord(BaseModel):
    """One private candidate bound to an immutable question/profile snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    task_id: str = Field(min_length=1)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_answer: SemanticAnswer
    output_contract_resolution: PrivateOutputContractResolution | None = None
    evidence: tuple[PrivateEvidenceRef, ...] = ()
    verification_checks: tuple[str, ...] = Field(min_length=1)
    verification_passed: bool
    unresolved_risks: tuple[str, ...] = ()
    recorded_at: datetime


class CurrentTaskDryRunItem(BaseModel):
    """Answer-free task status safe to display in logs and reports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    status: str = Field(pattern=r"^(?:ready|blocked)$")
    serialization_valid: bool = False
    evidence_count: int = Field(ge=0)
    verification_check_count: int = Field(ge=0)
    errors: tuple[str, ...] = ()


class CurrentDryRunReport(BaseModel):
    """Full-set preflight report that never emits candidate answer text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    ready_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    items: tuple[CurrentTaskDryRunItem, ...]
    profile_reconciliation_safe: bool
    all_serializations_valid: bool
    submission_capability_present: bool = False


class CurrentFreezeReport(BaseModel):
    """Answer-redacted outcome of private full-candidate preflight."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    valid: bool
    expected_count: int = Field(ge=1)
    ready_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    issue_counts: dict[str, int]
    frozen_candidate_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    private_manifest_path: str | None = None
    submission_capability_present: bool = False


class PrivateCandidateStore:
    """Atomic private candidate persistence under an explicitly ignored path."""

    def __init__(self, runs_dir: Path) -> None:
        self.root = (runs_dir / "current" / "private-candidates").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, record: PrivateCandidateRecord) -> Path:
        target = (self.root / f"{record.task_id}.json").resolve()
        self._assert_below_root(target)
        payload = record.model_dump_json(indent=2).encode("utf-8")
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.root,
                prefix=f".{record.task_id}-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, target)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return target

    def get(self, task_id: str) -> PrivateCandidateRecord | None:
        target = (self.root / f"{task_id}.json").resolve()
        self._assert_below_root(target)
        if not target.is_file():
            return None
        return PrivateCandidateRecord.model_validate_json(target.read_text(encoding="utf-8"))

    def _assert_below_root(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("private candidate path escaped the run store") from exc


def latest_stored_snapshot(runs_dir: Path) -> StoredQuestionSnapshot:
    paths = list((runs_dir / "snapshots").glob("questions-*.json"))
    if not paths:
        raise FileNotFoundError("no local question snapshot; run `gaia snapshot` first")
    latest = max(paths, key=lambda path: path.stat().st_mtime_ns)
    return StoredQuestionSnapshot.model_validate_json(latest.read_text(encoding="utf-8"))


async def prepare_current_attachments(settings: Settings) -> CurrentRunPreparationReport:
    """Acquire every official attachment without reading or emitting candidate answers."""

    stored = latest_stored_snapshot(settings.runs_dir)
    questions = [question for question in stored.questions if question.file_name is not None]
    store = ArtifactStore(settings.artifacts_dir)
    validator = AttachmentValidator(max_bytes=settings.max_attachment_bytes)
    items: list[AttachmentPreparationItem] = []
    async with (
        GaiaApiClient(settings) as client,
        GaiaGatedDatasetClient(settings) as gated_client,
    ):
        resolver = OfficialAttachmentResolver(
            client=client,
            store=store,
            validator=validator,
            settings=settings,
        )
        gated_resolver = GatedAttachmentResolver(
            client=gated_client,
            store=store,
            validator=validator,
            settings=settings,
        )
        for question in questions:
            items.append(
                await resolve_current_attachment(
                    question,
                    official_resolver=resolver,
                    gated_resolver=gated_resolver,
                )
            )
    return CurrentRunPreparationReport(
        snapshot_sha256=stored.snapshot.snapshot_sha256,
        expected_attachments=len(questions),
        ready_attachments=sum(item.ready for item in items),
        items=tuple(items),
    )


async def resolve_current_attachment(
    question: Question,
    *,
    official_resolver: OfficialAttachmentResolver,
    gated_resolver: GatedAttachmentResolver,
) -> AttachmentPreparationItem:
    """Resolve one official attachment, invoking the gated handoff when required."""

    outcome = await official_resolver.resolve(question)
    if outcome.status is OfficialAttachmentStatus.RESOLVED:
        if outcome.artifact is None:
            raise RuntimeError("resolved attachment outcome lost its artifact")
        return AttachmentPreparationItem(
            task_id=question.task_id,
            file_name=outcome.expected_file_name,
            ready=True,
            artifact_id=outcome.artifact.artifact_id,
        )

    try:
        gated = await gated_resolver.resolve(question, outcome)
    except GatedDatasetAccessError:
        reason = "gated_dataset_access_denied"
    except GatedDatasetAttachmentNotFoundError:
        reason = "gated_attachment_not_found"
    except GatedAttachmentRetriesExhaustedError:
        reason = "gated_transient_retries_exhausted"
    except GatedAttachmentRejectedError:
        reason = "gated_attachment_rejected"
    except GatedDatasetError:
        reason = "gated_dataset_response_error"
    else:
        return AttachmentPreparationItem(
            task_id=question.task_id,
            file_name=gated.expected_file_name,
            ready=True,
            artifact_id=gated.artifact.artifact_id,
        )

    return AttachmentPreparationItem(
        task_id=question.task_id,
        file_name=outcome.expected_file_name,
        ready=False,
        reason=reason,
    )


def build_current_dry_run(settings: Settings) -> CurrentDryRunReport:
    """Validate all private candidates while returning only answer-redacted status."""

    stored = latest_stored_snapshot(settings.runs_dir)
    registry = load_profile_registry(settings.task_profiles_path)
    reconciliation = reconcile_profile_registry(registry, stored.snapshot)
    profiles = registry.by_task_id()
    candidates = PrivateCandidateStore(settings.runs_dir)
    serializer = AnswerSerializer()
    validator = FinalAnswerValidator()
    items: list[CurrentTaskDryRunItem] = []
    all_serializations_valid = True

    for task_id in stored.snapshot.task_ids:
        record = candidates.get(task_id)
        errors: list[str] = []
        if record is None:
            errors.append(
                "official_attachment_unavailable"
                if task_id in stored.snapshot.attachment_task_ids
                else "candidate_not_recorded"
            )
            items.append(
                CurrentTaskDryRunItem(
                    task_id=task_id,
                    status="blocked",
                    evidence_count=0,
                    verification_check_count=0,
                    errors=tuple(errors),
                )
            )
            continue

        profile = profiles.get(task_id)
        if profile is None:
            errors.append("profile_missing")
        else:
            if record.snapshot_sha256 != stored.snapshot.snapshot_sha256:
                errors.append("candidate_snapshot_mismatch")
            if record.question_sha256 != stored.snapshot.task_hashes[task_id]:
                errors.append("candidate_question_mismatch")
            if not record.verification_passed:
                errors.append("verification_not_passed")
            if record.unresolved_risks:
                errors.append("unresolved_candidate_risks")
            if profile.required_sources and not record.evidence:
                errors.append("required_evidence_missing")

        serialization_valid = False
        if profile is not None:
            try:
                output_contract = resolve_private_output_contract(
                    profile.output_contract,
                    record.output_contract_resolution,
                )
                serialized = serializer.serialize(record.semantic_answer, output_contract)
                validation = validator.validate(serialized, output_contract)
                if not validation.valid:
                    errors.append("serialized_answer_invalid")
                    all_serializations_valid = False
                else:
                    serialization_valid = True
            except (TypeError, ValueError):
                errors.append("serialization_failed")
                all_serializations_valid = False

        items.append(
            CurrentTaskDryRunItem(
                task_id=task_id,
                status="blocked" if errors else "ready",
                serialization_valid=serialization_valid,
                evidence_count=len(record.evidence),
                verification_check_count=len(record.verification_checks),
                errors=tuple(dict.fromkeys(errors)),
            )
        )

    ready_count = sum(item.status == "ready" for item in items)
    return CurrentDryRunReport(
        snapshot_sha256=stored.snapshot.snapshot_sha256,
        task_count=len(items),
        ready_count=ready_count,
        blocked_count=len(items) - ready_count,
        items=tuple(items),
        profile_reconciliation_safe=reconciliation.submission_safe,
        all_serializations_valid=all_serializations_valid,
    )


def preflight_current_candidate(settings: Settings) -> CurrentFreezeReport:
    """Freeze a private manifest only when every current task passes preflight."""

    stored = latest_stored_snapshot(settings.runs_dir)
    dry_run = build_current_dry_run(settings)
    dry_items = {item.task_id: item for item in dry_run.items}
    registry = load_profile_registry(settings.task_profiles_path)
    reconciliation = reconcile_profile_registry(registry, stored.snapshot)
    profiles = registry.by_task_id()
    candidates = PrivateCandidateStore(settings.runs_dir)
    serializer = AnswerSerializer()
    results: list[TaskResult] = []

    for task_id in stored.snapshot.task_ids:
        item = dry_items[task_id]
        record = candidates.get(task_id)
        profile = profiles.get(task_id)
        if item.status == "ready" and record is not None and profile is not None:
            results.append(
                TaskResult(
                    task_id=task_id,
                    status=TaskStatus.READY,
                    semantic_answer=record.semantic_answer,
                    serialized_answer=serializer.serialize(
                        record.semantic_answer,
                        resolve_private_output_contract(
                            profile.output_contract,
                            record.output_contract_resolution,
                        ),
                    ),
                )
            )
        else:
            results.append(
                TaskResult(
                    task_id=task_id,
                    status=TaskStatus.BLOCKED,
                    errors=list(item.errors) or ["candidate_not_ready"],
                )
            )

    preflight = CandidatePreflight().validate(
        run_id=f"current-{stored.snapshot.snapshot_sha256[:16]}",
        snapshot=stored.snapshot,
        results=results,
        snapshot_approved=reconciliation.submission_safe,
        run_complete=True,
    )
    issue_counts = dict(
        sorted(Counter(issue.code.value for issue in preflight.issues).items())
    )
    private_manifest_path: str | None = None
    candidate_sha256: str | None = None
    if preflight.manifest is not None:
        target = (settings.runs_dir / "current" / "frozen-candidate.json").resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, preflight.manifest.model_dump_json(indent=2).encode("utf-8"))
        private_manifest_path = str(target)
        candidate_sha256 = preflight.manifest.candidate_sha256

    return CurrentFreezeReport(
        snapshot_sha256=stored.snapshot.snapshot_sha256,
        valid=preflight.valid,
        expected_count=preflight.expected_count,
        ready_count=dry_run.ready_count,
        blocked_count=dry_run.blocked_count,
        issue_counts=issue_counts,
        frozen_candidate_sha256=candidate_sha256,
        private_manifest_path=private_manifest_path,
    )


def _atomic_write(target: Path, payload: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def resolve_private_output_contract(
    contract: OutputContract,
    resolution: PrivateOutputContractResolution | None,
) -> OutputContract:
    if resolution is None:
        return contract
    if contract.answer_type is not AnswerType.CURRENCY:
        raise ValueError("currency-symbol resolution requires a currency contract")
    if contract.include_currency_symbol is not None:
        raise ValueError("currency-symbol resolution cannot replace an explicit contract")
    updates: dict[str, object] = {
        "include_currency_symbol": resolution.include_currency_symbol
    }
    if resolution.thousands_separator is not None:
        updates["thousands_separator"] = resolution.thousands_separator
    return contract.model_copy(update=updates)


__all__ = [
    "AttachmentPreparationItem",
    "CurrentDryRunReport",
    "CurrentFreezeReport",
    "CurrentRunPreparationReport",
    "CurrentTaskDryRunItem",
    "PrivateCandidateRecord",
    "PrivateCandidateStore",
    "PrivateEvidenceRef",
    "PrivateOutputContractResolution",
    "build_current_dry_run",
    "latest_stored_snapshot",
    "preflight_current_candidate",
    "prepare_current_attachments",
    "resolve_current_attachment",
    "resolve_private_output_contract",
]
