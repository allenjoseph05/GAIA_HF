"""Versioned, answer-free task profile registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gaia_max.domain import OutputContract, Question, QuestionSnapshot, TaskProfile
from gaia_max.domain.models import Modality, RiskLevel, TaskClass, TemporalConstraint
from gaia_max.snapshot import question_sha256

FORBIDDEN_PROFILE_KEYS = {
    "answer",
    "candidate_answer",
    "final answer",
    "final_answer",
    "gold_answer",
    "semantic_answer",
    "serialized_answer",
    "submitted_answer",
}


class ProfileRegistryError(RuntimeError):
    """The answer-free task registry is missing, invalid, or stale."""


class TaskProfileRecord(BaseModel):
    """Answer-free routing and contract metadata for one public task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_file_name: str | None = None
    modality: Modality
    task_class: TaskClass
    route: str = Field(min_length=1)
    requested_operation: str = Field(min_length=1)
    temporal_constraint: TemporalConstraint | None = None
    required_sources: list[str] = Field(default_factory=list)
    output_contract: OutputContract
    verification_policy: str = Field(min_length=1)
    risk_level: RiskLevel
    risk_flags: list[str] = Field(default_factory=list)

    def materialize(self, question: Question) -> TaskProfile:
        """Combine stable metadata with the current public question text."""

        current_hash = question_sha256(question)
        if current_hash != self.question_sha256:
            raise ProfileRegistryError(
                f"task {question.task_id} question hash does not match its profile"
            )
        if question.file_name != self.expected_file_name:
            raise ProfileRegistryError(
                f"task {question.task_id} filename does not match its profile"
            )
        return TaskProfile(
            task_id=question.task_id,
            question=question.question,
            question_sha256=current_hash,
            file_name=question.file_name,
            modality=self.modality,
            task_class=self.task_class,
            route=self.route,
            risk_level=self.risk_level,
            temporal_constraint=self.temporal_constraint,
            requested_operation=self.requested_operation,
            required_sources=self.required_sources,
            output_contract=self.output_contract,
            verification_policy=self.verification_policy,
            risk_flags=self.risk_flags,
        )


class TaskProfileRegistry(BaseModel):
    """Validated registry document bound to one public question snapshot."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    profiles_version: str = Field(min_length=1)
    question_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_count: int = Field(ge=1)
    profiles: list[TaskProfileRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> TaskProfileRegistry:
        if self.profile_count != len(self.profiles):
            raise ValueError("profile_count must equal the number of profiles")
        task_ids = [profile.task_id for profile in self.profiles]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("profile task IDs must be unique")
        return self

    def by_task_id(self) -> dict[str, TaskProfileRecord]:
        return {profile.task_id: profile for profile in self.profiles}


class ProfileReconciliation(BaseModel):
    """Difference between a runtime question snapshot and the profile registry."""

    model_config = ConfigDict(extra="forbid")

    snapshot_hash_matches: bool
    profile_version_matches: bool
    matched_task_ids: list[str]
    missing_profile_task_ids: list[str]
    unknown_profile_task_ids: list[str]
    stale_profile_task_ids: list[str]

    @property
    def submission_safe(self) -> bool:
        return (
            self.snapshot_hash_matches
            and self.profile_version_matches
            and not self.missing_profile_task_ids
            and not self.unknown_profile_task_ids
            and not self.stale_profile_task_ids
        )


def _reject_forbidden_keys(value: Any, *, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_PROFILE_KEYS:
                raise ProfileRegistryError(
                    f"forbidden answer-bearing key {key!r} found at {location}"
                )
            _reject_forbidden_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, location=f"{location}[{index}]")


def load_profile_registry(path: Path) -> TaskProfileRegistry:
    """Load and validate an answer-free JSON registry."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfileRegistryError(f"task profile registry not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileRegistryError(f"task profile registry is invalid JSON: {path}") from exc

    _reject_forbidden_keys(raw)
    try:
        return TaskProfileRegistry.model_validate(raw)
    except ValidationError as exc:
        raise ProfileRegistryError(f"task profile registry schema is invalid: {path}") from exc


def reconcile_profile_registry(
    registry: TaskProfileRegistry,
    snapshot: QuestionSnapshot,
) -> ProfileReconciliation:
    """Compare registry and runtime inventories without using question answers."""

    profiles = registry.by_task_id()
    snapshot_ids = set(snapshot.task_ids)
    profile_ids = set(profiles)
    common_ids = snapshot_ids & profile_ids
    stale = sorted(
        task_id
        for task_id in common_ids
        if profiles[task_id].question_sha256 != snapshot.task_hashes[task_id]
    )
    stale_set = set(stale)
    matched = sorted(task_id for task_id in common_ids if task_id not in stale_set)

    return ProfileReconciliation(
        snapshot_hash_matches=(
            registry.question_snapshot_sha256 == snapshot.snapshot_sha256
        ),
        profile_version_matches=(registry.profiles_version == snapshot.task_profiles_version),
        matched_task_ids=matched,
        missing_profile_task_ids=sorted(snapshot_ids - profile_ids),
        unknown_profile_task_ids=sorted(profile_ids - snapshot_ids),
        stale_profile_task_ids=stale,
    )
