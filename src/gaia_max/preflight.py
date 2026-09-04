"""Pure submission-candidate preflight and answer-redacted dry-run reporting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gaia_max.domain import QuestionSnapshot, TaskResult, TaskStatus
from gaia_max.run_graph import GaiaRunResult, RunStatus


class PreflightIssueCode(StrEnum):
    """Minimal structural failures that make a candidate unsafe to freeze."""

    SNAPSHOT_NOT_APPROVED = "snapshot_not_approved"
    RUN_NOT_COMPLETE = "run_not_complete"
    RESULT_COUNT_MISMATCH = "result_count_mismatch"
    DUPLICATE_TASK_ID = "duplicate_task_id"
    UNKNOWN_TASK_ID = "unknown_task_id"
    MISSING_TASK_ID = "missing_task_id"
    TASK_NOT_READY = "task_not_ready"
    EMPTY_ANSWER = "empty_answer"


class PreflightIssue(BaseModel):
    """One answer-free preflight failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: PreflightIssueCode
    task_id: str | None = None


class CandidateAnswerEntry(BaseModel):
    """One exact answer in a frozen candidate; not an HTTP submission model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    answer: str = Field(min_length=1)

    @field_validator("answer")
    @classmethod
    def reject_blank_answer(cls, answer: str) -> str:
        if not answer.strip():
            raise ValueError("candidate answer cannot be blank")
        return answer


class FrozenCandidateManifest(BaseModel):
    """Immutable, hashed output of a passing structural preflight."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_profiles_version: str = Field(min_length=1)
    answers: tuple[CandidateAnswerEntry, ...] = Field(min_length=1)
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hash_and_inventory(self) -> FrozenCandidateManifest:
        task_ids = [entry.task_id for entry in self.answers]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("frozen candidate task IDs must be unique")
        if self.candidate_sha256 != _candidate_sha256(
            run_id=self.run_id,
            snapshot_sha256=self.snapshot_sha256,
            task_profiles_version=self.task_profiles_version,
            answers=self.answers,
        ):
            raise ValueError("frozen candidate hash does not match its content")
        return self


class PreflightReport(BaseModel):
    """Structural decision plus a manifest only when every check passes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    valid: bool
    expected_count: int = Field(ge=1)
    received_count: int = Field(ge=0)
    issues: tuple[PreflightIssue, ...] = ()
    manifest: FrozenCandidateManifest | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> PreflightReport:
        if self.valid != (not self.issues):
            raise ValueError("preflight validity must agree with issue presence")
        if self.valid != (self.manifest is not None):
            raise ValueError("only a valid preflight may contain a frozen manifest")
        return self


class DryRunTaskSummary(BaseModel):
    """Answer-redacted status for one task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    status: TaskStatus
    has_serialized_answer: bool
    error_count: int = Field(ge=0)


class DryRunReport(BaseModel):
    """Small operator report that cannot expose candidate strings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    run_status: RunStatus
    expected_count: int = Field(ge=1)
    selected_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    tasks: tuple[DryRunTaskSummary, ...]
    preflight_valid: bool
    preflight_issue_codes: tuple[PreflightIssueCode, ...]
    submission_capability_present: bool = False


class CandidatePreflight:
    """Validate a result sequence without network, filesystem, or submission access."""

    def validate(
        self,
        *,
        run_id: str,
        snapshot: QuestionSnapshot,
        results: Sequence[TaskResult],
        snapshot_approved: bool,
        run_complete: bool = True,
    ) -> PreflightReport:
        issues: list[PreflightIssue] = []
        if not snapshot_approved:
            issues.append(PreflightIssue(code=PreflightIssueCode.SNAPSHOT_NOT_APPROVED))
        if not run_complete:
            issues.append(PreflightIssue(code=PreflightIssueCode.RUN_NOT_COMPLETE))
        if len(results) != snapshot.count:
            issues.append(PreflightIssue(code=PreflightIssueCode.RESULT_COUNT_MISMATCH))

        expected = set(snapshot.task_ids)
        counts = Counter(result.task_id for result in results)
        for task_id in sorted(task_id for task_id, count in counts.items() if count > 1):
            issues.append(
                PreflightIssue(
                    code=PreflightIssueCode.DUPLICATE_TASK_ID,
                    task_id=task_id,
                )
            )
        for task_id in sorted(set(counts) - expected):
            issues.append(
                PreflightIssue(
                    code=PreflightIssueCode.UNKNOWN_TASK_ID,
                    task_id=task_id,
                )
            )
        for task_id in sorted(expected - set(counts)):
            issues.append(
                PreflightIssue(
                    code=PreflightIssueCode.MISSING_TASK_ID,
                    task_id=task_id,
                )
            )

        by_id: dict[str, TaskResult] = {}
        for result in results:
            by_id.setdefault(result.task_id, result)
            if result.status is not TaskStatus.READY:
                issues.append(
                    PreflightIssue(
                        code=PreflightIssueCode.TASK_NOT_READY,
                        task_id=result.task_id,
                    )
                )
            if (
                not isinstance(result.serialized_answer, str)
                or not result.serialized_answer.strip()
            ):
                issues.append(
                    PreflightIssue(
                        code=PreflightIssueCode.EMPTY_ANSWER,
                        task_id=result.task_id,
                    )
                )

        issues = _deduplicate_issues(issues)
        if issues:
            return PreflightReport(
                valid=False,
                expected_count=snapshot.count,
                received_count=len(results),
                issues=tuple(issues),
            )

        answers = tuple(
            CandidateAnswerEntry(
                task_id=task_id,
                answer=by_id[task_id].serialized_answer or "",
            )
            for task_id in snapshot.task_ids
        )
        candidate_hash = _candidate_sha256(
            run_id=run_id,
            snapshot_sha256=snapshot.snapshot_sha256,
            task_profiles_version=snapshot.task_profiles_version,
            answers=answers,
        )
        manifest = FrozenCandidateManifest(
            run_id=run_id,
            snapshot_sha256=snapshot.snapshot_sha256,
            task_profiles_version=snapshot.task_profiles_version,
            answers=answers,
            candidate_sha256=candidate_hash,
        )
        return PreflightReport(
            valid=True,
            expected_count=snapshot.count,
            received_count=len(results),
            manifest=manifest,
        )


def preflight_run(
    run: GaiaRunResult,
    snapshot: QuestionSnapshot,
    *,
    snapshot_approved: bool,
) -> PreflightReport:
    """Run structural preflight over an A18 result in canonical selected order."""

    results = [
        run.task_results[task_id]
        for task_id in run.selected_task_ids
        if task_id in run.task_results
    ]
    return CandidatePreflight().validate(
        run_id=run.run_id,
        snapshot=snapshot,
        results=results,
        snapshot_approved=snapshot_approved,
        run_complete=run.status is RunStatus.COMPLETE,
    )


def build_dry_run_report(
    run: GaiaRunResult,
    snapshot: QuestionSnapshot,
    *,
    snapshot_approved: bool,
) -> DryRunReport:
    """Create an answer-redacted report; this function has no side effects."""

    preflight = preflight_run(run, snapshot, snapshot_approved=snapshot_approved)
    tasks = tuple(
        DryRunTaskSummary(
            task_id=task_id,
            status=result.status,
            has_serialized_answer=bool(
                isinstance(result.serialized_answer, str)
                and result.serialized_answer.strip()
            ),
            error_count=len(result.errors),
        )
        for task_id in run.selected_task_ids
        if (result := run.task_results.get(task_id)) is not None
    )
    return DryRunReport(
        run_id=run.run_id,
        run_status=run.status,
        expected_count=snapshot.count,
        selected_count=len(run.selected_task_ids),
        ready_count=sum(task.status is TaskStatus.READY for task in tasks),
        blocked_count=sum(task.status is TaskStatus.BLOCKED for task in tasks),
        tasks=tasks,
        preflight_valid=preflight.valid,
        preflight_issue_codes=tuple(issue.code for issue in preflight.issues),
    )


def _candidate_sha256(
    *,
    run_id: str,
    snapshot_sha256: str,
    task_profiles_version: str,
    answers: tuple[CandidateAnswerEntry, ...],
) -> str:
    payload = {
        "answers": [entry.model_dump(mode="json") for entry in answers],
        "run_id": run_id,
        "snapshot_sha256": snapshot_sha256,
        "task_profiles_version": task_profiles_version,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deduplicate_issues(issues: list[PreflightIssue]) -> list[PreflightIssue]:
    unique: list[PreflightIssue] = []
    for issue in issues:
        if issue not in unique:
            unique.append(issue)
    return unique
