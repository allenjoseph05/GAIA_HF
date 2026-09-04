"""Timestamped ASR contracts and evidence-first disagreement reconciliation."""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaia_max.media.core import DerivedMedia


class ASRError(ValueError):
    """An ASR result is incomplete, temporally invalid, or unresolved."""


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    raw_text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_interval(self) -> TranscriptSegment:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("transcript segment end must follow its start")
        return self


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    engine: str = Field(min_length=1)
    language: str | None = None
    duration_seconds: float = Field(gt=0)
    segments: tuple[TranscriptSegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_timeline(self) -> Transcript:
        previous_end = 0.0
        for segment in self.segments:
            if segment.start_seconds < previous_end - 0.05:
                raise ValueError("transcript segments overlap or are out of order")
            if segment.end_seconds > self.duration_seconds + 0.1:
                raise ValueError("transcript segment exceeds media duration")
            previous_end = segment.end_seconds
        return self


class ASREngine(Protocol):
    name: str

    def transcribe(
        self,
        media: DerivedMedia,
        *,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> Transcript: ...


class FasterWhisperSegmentPort(Protocol):
    start: float
    end: float
    text: str
    avg_logprob: float


class FasterWhisperInfoPort(Protocol):
    language: str


class FasterWhisperModelPort(Protocol):
    def transcribe(
        self,
        audio: str,
        *,
        beam_size: int,
        word_timestamps: bool,
        clip_timestamps: str | None,
    ) -> tuple[Iterable[FasterWhisperSegmentPort], FasterWhisperInfoPort]: ...


class FasterWhisperFactory(Protocol):
    def __call__(
        self,
        model_size_or_path: str,
        *,
        device: str,
        compute_type: str,
        download_root: str | None,
    ) -> FasterWhisperModelPort: ...


class FasterWhisperEngine:
    """Optional local CPU/GPU ASR adapter with timestamped focused decoding."""

    def __init__(self, model: FasterWhisperModelPort, *, identity: str) -> None:
        self._model = model
        self.name = identity

    @classmethod
    def from_installed_package(
        cls,
        model_size_or_path: str = "small.en",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Path | None = None,
    ) -> FasterWhisperEngine:
        try:
            module = importlib.import_module("faster_whisper")
            factory = cast(FasterWhisperFactory, module.WhisperModel)
        except (ImportError, AttributeError) as exc:
            raise ASRError(
                "faster-whisper is unavailable; install the asr extra under a supported Python"
            ) from exc
        model = factory(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root.resolve()) if download_root else None,
        )
        return cls(model, identity=f"faster-whisper:{model_size_or_path}:{device}")

    def transcribe(
        self,
        media: DerivedMedia,
        *,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> Transcript:
        if (start_seconds is None) != (end_seconds is None):
            raise ASRError("focused decode requires both start and end timestamps")
        clip: str | None = None
        if start_seconds is not None and end_seconds is not None:
            if start_seconds < 0 or end_seconds <= start_seconds:
                raise ASRError("focused decode interval is invalid")
            if end_seconds > media.probe.duration_seconds + 0.1:
                raise ASRError("focused decode exceeds media duration")
            clip = f"{start_seconds:.3f},{end_seconds:.3f}"
        raw_segments, info = self._model.transcribe(
            str(media.artifact.local_path.resolve()),
            beam_size=5,
            word_timestamps=True,
            clip_timestamps=clip,
        )
        segments = tuple(
            TranscriptSegment(
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                raw_text=segment.text.strip(),
                normalized_text=normalize_transcript_text(segment.text),
                confidence=max(0, min(1, math.exp(float(segment.avg_logprob)))),
            )
            for segment in raw_segments
            if segment.text.strip()
        )
        if not segments:
            raise ASRError("faster-whisper returned no speech segments")
        return Transcript(
            engine=self.name,
            language=info.language,
            duration_seconds=media.probe.duration_seconds,
            segments=segments,
        )


class ASRPass(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    primary: Transcript
    secondary: Transcript | None = None


class DisputedSegment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    primary_text: str = Field(min_length=1)
    secondary_text: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    resolved_text: str | None = None


def normalize_transcript_text(text: str) -> str:
    return " ".join(text.casefold().split())


def transcribe_timestamped(engine: ASREngine, media: DerivedMedia) -> Transcript:
    transcript = engine.transcribe(media)
    if transcript.engine != engine.name:
        raise ASRError("ASR transcript engine identity does not match the adapter")
    if abs(transcript.duration_seconds - media.probe.duration_seconds) > 0.1:
        raise ASRError("ASR transcript duration does not match normalized media")
    for segment in transcript.segments:
        if segment.normalized_text != normalize_transcript_text(segment.raw_text):
            raise ASRError("normalized transcript must preserve raw text separately")
    return transcript


def reconcile_transcripts(
    primary: Transcript,
    secondary: Transcript,
    *,
    adjudicated: dict[tuple[float, float], str] | None = None,
) -> tuple[DisputedSegment, ...]:
    """Identify exact/numeric disagreements; require explicit evidence adjudication."""

    disputes: list[DisputedSegment] = []
    resolutions = adjudicated or {}
    for first in primary.segments:
        candidates = [
            second
            for second in secondary.segments
            if min(first.end_seconds, second.end_seconds)
            > max(first.start_seconds, second.start_seconds)
        ]
        for second in candidates:
            if first.normalized_text == second.normalized_text:
                continue
            exact_sensitive = _contains_number(first.raw_text) or _contains_number(second.raw_text)
            reason = "numeric disagreement" if exact_sensitive else "exact-word disagreement"
            key = (first.start_seconds, first.end_seconds)
            resolved = resolutions.get(key)
            disputes.append(
                DisputedSegment(
                    start_seconds=first.start_seconds,
                    end_seconds=first.end_seconds,
                    primary_text=first.raw_text,
                    secondary_text=second.raw_text,
                    reason=reason,
                    resolved_text=resolved,
                )
            )
    unknown = set(resolutions) - {
        (item.start_seconds, item.end_seconds) for item in disputes
    }
    if unknown:
        raise ASRError("adjudication includes a segment that is not disputed")
    return tuple(disputes)


def _contains_number(text: str) -> bool:
    number_words = (
        r"zero|one|two|three|four|five|six|seven|eight|nine|ten|teen|twenty|"
        r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred"
    )
    return bool(re.search(rf"\d|\b(?:{number_words})\b", text, re.I))


__all__ = [
    "ASREngine",
    "ASRError",
    "ASRPass",
    "DisputedSegment",
    "FasterWhisperEngine",
    "FasterWhisperModelPort",
    "Transcript",
    "TranscriptSegment",
    "normalize_transcript_text",
    "reconcile_transcripts",
    "transcribe_timestamped",
]
