"""Validated application configuration with no-cost and submission safety gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment or a local ``.env`` file.

    Submission and paid services are intentionally opt-in. A free Hugging Face
    access token does not itself imply paid usage.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    gaia_api_url: AnyHttpUrl = AnyHttpUrl(
        "https://agents-course-unit4-scoring.hf.space"
    )
    hf_token: SecretStr | None = None

    dry_run: bool = True
    allow_submit: bool = False
    zero_cost_mode: bool = True
    allow_paid_services: bool = False

    runs_dir: Path = Path("runs")
    artifacts_dir: Path = Path("runs/artifacts")
    stockfish_path: Path | None = None
    chess_engine_depth: int = Field(default=18, ge=8, le=40)
    chess_engine_multipv: int = Field(default=3, ge=2, le=10)
    media_cache_dir: Path = Path("runs/media")
    ffmpeg_executable: str = Field(default="ffmpeg", min_length=1)
    ffprobe_executable: str = Field(default="ffprobe", min_length=1)
    ytdlp_executable: str = Field(default="yt-dlp", min_length=1)
    asr_model: str = Field(default="small.en", min_length=1)
    asr_device: str = Field(default="cpu", pattern=r"^(cpu|cuda)$")
    asr_compute_type: str = Field(default="int8", min_length=1)
    max_media_duration_seconds: float = Field(default=14_400, gt=0, le=86_400)
    video_coarse_fps: float = Field(default=1, gt=0, le=10)
    video_dense_fps: float = Field(default=6, gt=0, le=30)

    expected_question_count: int = Field(default=20, ge=1, le=1000)
    task_profiles_version: str = Field(default="2026-08-26-v1", min_length=1)
    task_profiles_path: Path = Path("config/task_profiles.json")

    http_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    download_timeout_seconds: float = Field(default=120.0, gt=0, le=1800)
    max_attachment_bytes: int = Field(default=100 * 1024 * 1024, ge=1, le=2**31)
    attachment_max_attempts: int = Field(default=3, ge=1, le=10)
    attachment_retry_base_seconds: float = Field(default=0.5, ge=0, le=30)
    search_max_attempts: int = Field(default=3, ge=1, le=10)
    search_retry_base_seconds: float = Field(default=0.5, ge=0, le=60)
    search_retry_max_seconds: float = Field(default=8.0, ge=0, le=300)
    search_retry_jitter_ratio: float = Field(default=0.2, ge=0, le=1)
    search_cache_ttl_seconds: float = Field(default=86_400, ge=0, le=31_536_000)
    search_primary_min_interval_seconds: float = Field(default=0, ge=0, le=60)
    search_secondary_min_interval_seconds: float = Field(default=0, ge=0, le=60)
    max_page_bytes: int = Field(default=5 * 1024 * 1024, ge=1, le=100 * 1024 * 1024)
    page_max_redirects: int = Field(default=5, ge=0, le=20)
    page_min_extracted_chars: int = Field(default=80, ge=1, le=10_000)
    page_max_attempts: int = Field(default=2, ge=1, le=10)
    page_retry_base_seconds: float = Field(default=0.5, ge=0, le=60)
    page_retry_max_seconds: float = Field(default=4.0, ge=0, le=300)
    page_retry_jitter_ratio: float = Field(default=0.2, ge=0, le=1)
    mediawiki_max_history_pages: int = Field(default=10, ge=1, le=100)
    mediawiki_revision_batch_size: int = Field(default=50, ge=1, le=500)
    source_max_flagged_line_ratio: float = Field(default=0.4, ge=0, le=1)
    source_max_injection_findings: int = Field(default=20, ge=1, le=1000)
    source_min_remaining_chars: int = Field(default=40, ge=0, le=100_000)
    max_model_source_chars: int = Field(default=100_000, ge=1, le=1_000_000)
    max_research_steps: int = Field(default=16, ge=1, le=100)
    max_replan: int = Field(default=3, ge=0, le=20)
    research_planning_interval: int = Field(default=4, ge=1, le=100)
    research_max_tool_calls: int = Field(default=16, ge=1, le=1000)
    research_tool_timeout_seconds: float = Field(default=30, gt=0, le=1800)
    research_require_independent_retrieval: bool = False
    max_verifier_retries: int = Field(default=3, ge=0, le=20)
    max_task_concurrency: int = Field(default=4, ge=1, le=64)
    telemetry_backend: Literal["none", "otel", "langfuse"] = "none"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: AnyHttpUrl = AnyHttpUrl("https://cloud.langfuse.com")

    @model_validator(mode="after")
    def validate_safety_gates(self) -> Settings:
        if self.allow_submit and self.dry_run:
            raise ValueError("ALLOW_SUBMIT cannot be true while DRY_RUN is true")
        if self.zero_cost_mode and self.allow_paid_services:
            raise ValueError("ZERO_COST_MODE cannot be combined with ALLOW_PAID_SERVICES")
        if self.search_retry_max_seconds < self.search_retry_base_seconds:
            raise ValueError(
                "SEARCH_RETRY_MAX_SECONDS cannot be smaller than "
                "SEARCH_RETRY_BASE_SECONDS"
            )
        if self.page_retry_max_seconds < self.page_retry_base_seconds:
            raise ValueError(
                "PAGE_RETRY_MAX_SECONDS cannot be smaller than "
                "PAGE_RETRY_BASE_SECONDS"
            )
        if self.video_dense_fps <= self.video_coarse_fps:
            raise ValueError("VIDEO_DENSE_FPS must be greater than VIDEO_COARSE_FPS")
        if self.research_planning_interval > self.max_research_steps:
            raise ValueError(
                "RESEARCH_PLANNING_INTERVAL cannot exceed MAX_RESEARCH_STEPS"
            )
        if self.telemetry_backend == "langfuse" and (
            self.langfuse_public_key is None or self.langfuse_secret_key is None
        ):
            raise ValueError("Langfuse telemetry requires both Langfuse keys")
        return self

    @property
    def submission_enabled(self) -> bool:
        """Whether configuration permits reaching a future submission node."""

        return self.allow_submit and not self.dry_run

    def public_summary(self) -> dict[str, Any]:
        """Return safe diagnostic information without exposing secret values."""

        return {
            "gaia_api_url": str(self.gaia_api_url),
            "hf_token_configured": self.hf_token is not None,
            "dry_run": self.dry_run,
            "allow_submit": self.allow_submit,
            "asr_model": self.asr_model,
            "asr_device": self.asr_device,
            "video_coarse_fps": self.video_coarse_fps,
            "video_dense_fps": self.video_dense_fps,
            "submission_enabled": self.submission_enabled,
            "zero_cost_mode": self.zero_cost_mode,
            "allow_paid_services": self.allow_paid_services,
            "runs_dir": str(self.runs_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "stockfish_configured": self.stockfish_path is not None,
            "chess_engine_depth": self.chess_engine_depth,
            "chess_engine_multipv": self.chess_engine_multipv,
            "expected_question_count": self.expected_question_count,
            "task_profiles_version": self.task_profiles_version,
            "task_profiles_path": str(self.task_profiles_path),
            "http_timeout_seconds": self.http_timeout_seconds,
            "download_timeout_seconds": self.download_timeout_seconds,
            "max_attachment_bytes": self.max_attachment_bytes,
            "attachment_max_attempts": self.attachment_max_attempts,
            "attachment_retry_base_seconds": self.attachment_retry_base_seconds,
            "search_max_attempts": self.search_max_attempts,
            "search_retry_base_seconds": self.search_retry_base_seconds,
            "search_retry_max_seconds": self.search_retry_max_seconds,
            "search_retry_jitter_ratio": self.search_retry_jitter_ratio,
            "search_cache_ttl_seconds": self.search_cache_ttl_seconds,
            "search_primary_min_interval_seconds": (
                self.search_primary_min_interval_seconds
            ),
            "search_secondary_min_interval_seconds": (
                self.search_secondary_min_interval_seconds
            ),
            "max_page_bytes": self.max_page_bytes,
            "page_max_redirects": self.page_max_redirects,
            "page_min_extracted_chars": self.page_min_extracted_chars,
            "page_max_attempts": self.page_max_attempts,
            "page_retry_base_seconds": self.page_retry_base_seconds,
            "page_retry_max_seconds": self.page_retry_max_seconds,
            "page_retry_jitter_ratio": self.page_retry_jitter_ratio,
            "mediawiki_max_history_pages": self.mediawiki_max_history_pages,
            "mediawiki_revision_batch_size": self.mediawiki_revision_batch_size,
            "source_max_flagged_line_ratio": self.source_max_flagged_line_ratio,
            "source_max_injection_findings": self.source_max_injection_findings,
            "source_min_remaining_chars": self.source_min_remaining_chars,
            "max_model_source_chars": self.max_model_source_chars,
            "max_research_steps": self.max_research_steps,
            "max_replan": self.max_replan,
            "research_planning_interval": self.research_planning_interval,
            "research_max_tool_calls": self.research_max_tool_calls,
            "research_tool_timeout_seconds": self.research_tool_timeout_seconds,
            "research_require_independent_retrieval": (
                self.research_require_independent_retrieval
            ),
            "max_verifier_retries": self.max_verifier_retries,
            "max_task_concurrency": self.max_task_concurrency,
            "telemetry_backend": self.telemetry_backend,
            "langfuse_credentials_configured": (
                self.langfuse_public_key is not None
                and self.langfuse_secret_key is not None
            ),
            "langfuse_base_url": str(self.langfuse_base_url),
        }
