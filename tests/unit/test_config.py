from __future__ import annotations

import pytest
from pydantic import ValidationError

from gaia_max.config import Settings


def test_safe_defaults_disable_submission_and_paid_services() -> None:
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]

    assert settings.dry_run is True
    assert settings.allow_submit is False
    assert settings.submission_enabled is False
    assert settings.zero_cost_mode is True
    assert settings.allow_paid_services is False
    assert settings.max_attachment_bytes == 100 * 1024 * 1024
    assert settings.attachment_max_attempts == 3
    assert settings.attachment_retry_base_seconds == 0.5
    assert settings.search_max_attempts == 3
    assert settings.search_retry_base_seconds == 0.5
    assert settings.search_retry_max_seconds == 8
    assert settings.search_retry_jitter_ratio == 0.2
    assert settings.search_cache_ttl_seconds == 86_400
    assert settings.max_page_bytes == 5 * 1024 * 1024
    assert settings.page_max_redirects == 5
    assert settings.page_min_extracted_chars == 80
    assert settings.page_max_attempts == 2
    assert settings.page_retry_base_seconds == 0.5
    assert settings.page_retry_max_seconds == 4
    assert settings.source_max_flagged_line_ratio == 0.4
    assert settings.source_max_injection_findings == 20
    assert settings.source_min_remaining_chars == 40
    assert settings.max_model_source_chars == 100_000
    assert settings.max_task_concurrency == 4
    assert settings.telemetry_backend == "none"
    assert settings.research_planning_interval == 4
    assert settings.asr_model == "small.en"
    assert settings.video_coarse_fps == 1
    assert settings.video_dense_fps == 6


def test_submission_cannot_be_enabled_while_dry_run_is_active() -> None:
    with pytest.raises(ValidationError, match="ALLOW_SUBMIT cannot be true"):
        Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            allow_submit=True,
            dry_run=True,
        )


def test_zero_cost_mode_rejects_paid_services() -> None:
    with pytest.raises(ValidationError, match="ZERO_COST_MODE cannot be combined"):
        Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            zero_cost_mode=True,
            allow_paid_services=True,
        )


def test_public_summary_never_contains_token_value() -> None:
    token = "hf_secret_value_that_must_not_leak"
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        hf_token=token,
    )

    summary = settings.public_summary()

    assert summary["hf_token_configured"] is True
    assert token not in repr(summary)


def test_submission_can_only_be_enabled_by_explicit_consistent_flags() -> None:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        allow_submit=True,
        dry_run=False,
    )

    assert settings.submission_enabled is True


def test_search_retry_window_is_validated_at_configuration_boundary() -> None:
    with pytest.raises(ValidationError, match="SEARCH_RETRY_MAX_SECONDS"):
        Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            search_retry_base_seconds=2,
            search_retry_max_seconds=1,
        )


def test_page_retry_window_is_validated_at_configuration_boundary() -> None:
    with pytest.raises(ValidationError, match="PAGE_RETRY_MAX_SECONDS"):
        Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            page_retry_base_seconds=2,
            page_retry_max_seconds=1,
        )


def test_dense_video_sampling_must_exceed_coarse_sampling() -> None:
    with pytest.raises(ValidationError, match="VIDEO_DENSE_FPS"):
        Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            video_coarse_fps=4,
            video_dense_fps=4,
        )


def test_research_planning_interval_must_fit_step_budget() -> None:
    with pytest.raises(ValidationError, match="RESEARCH_PLANNING_INTERVAL"):
        Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            max_research_steps=3,
            research_planning_interval=4,
        )


def test_langfuse_requires_both_keys_and_redacts_them_from_summary() -> None:
    with pytest.raises(ValidationError, match="requires both Langfuse keys"):
        Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            telemetry_backend="langfuse",
            langfuse_public_key="pk-test",
        )

    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        telemetry_backend="langfuse",
        langfuse_public_key="pk-secret-value",
        langfuse_secret_key="sk-secret-value",
    )
    summary = settings.public_summary()
    assert summary["langfuse_credentials_configured"] is True
    assert "secret-value" not in repr(summary)
