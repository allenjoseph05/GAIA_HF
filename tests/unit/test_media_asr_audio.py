from __future__ import annotations

import math
from pathlib import Path

import pytest

from gaia_max.artifacts import ArtifactStore
from gaia_max.domain import ArtifactSource
from gaia_max.media.asr import (
    ASRError,
    FasterWhisperEngine,
    Transcript,
    TranscriptSegment,
    reconcile_transcripts,
    transcribe_timestamped,
)
from gaia_max.media.audio_tasks import extract_ingredients, extract_numeric_pages
from gaia_max.media.core import DerivedMedia, MediaProbe, MediaStream


def _segment(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        start_seconds=start,
        end_seconds=end,
        raw_text=text,
        normalized_text=" ".join(text.casefold().split()),
        confidence=0.9,
    )


def _transcript(*segments: TranscriptSegment, engine: str = "sensor-a") -> Transcript:
    return Transcript(engine=engine, duration_seconds=10, segments=segments)


def _media(tmp_path: Path) -> DerivedMedia:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.put_bytes(
        b"not-decoded-in-contract-test",
        original_name="derived.wav",
        media_type="audio/wav",
        source=ArtifactSource.DERIVED,
    )
    return DerivedMedia(
        artifact=artifact,
        source_artifact_id="a" * 64,
        probe=MediaProbe(
            duration_seconds=10,
            format_name="wav",
            streams=(
                MediaStream(
                    index=0,
                    codec_type="audio",
                    codec_name="pcm_s16le",
                    sample_rate=16_000,
                    channels=1,
                ),
            ),
        ),
    )


class _FakeEngine:
    name = "sensor-a"

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript

    def transcribe(
        self,
        media: DerivedMedia,
        *,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> Transcript:
        return self.transcript


class _WhisperSegment:
    start = 1.0
    end = 2.0
    text = " Page twelve "
    avg_logprob = -0.1


class _WhisperInfo:
    language = "en"


class _WhisperModel:
    def transcribe(
        self,
        audio: str,
        *,
        beam_size: int,
        word_timestamps: bool,
        clip_timestamps: str | None,
    ) -> tuple[tuple[_WhisperSegment, ...], _WhisperInfo]:
        assert beam_size == 5 and word_timestamps
        return (_WhisperSegment(),), _WhisperInfo()


def test_timestamped_asr_preserves_raw_and_normalized_text(tmp_path: Path) -> None:
    transcript = _transcript(_segment(0, 2, "  Page Twenty  "))
    assert transcribe_timestamped(_FakeEngine(transcript), _media(tmp_path)) == transcript


def test_timestamped_asr_rejects_lossy_normalization(tmp_path: Path) -> None:
    transcript = Transcript(
        engine="sensor-a",
        duration_seconds=10,
        segments=(
            TranscriptSegment(
                start_seconds=0,
                end_seconds=1,
                raw_text="Page Twenty",
                normalized_text="page 20",
            ),
        ),
    )
    with pytest.raises(ASRError, match="preserve raw"):
        transcribe_timestamped(_FakeEngine(transcript), _media(tmp_path))


def test_faster_whisper_adapter_returns_timestamped_confident_segments(tmp_path: Path) -> None:
    engine = FasterWhisperEngine(_WhisperModel(), identity="faster-whisper:test")

    transcript = transcribe_timestamped(engine, _media(tmp_path))

    assert transcript.language == "en"
    assert transcript.segments[0].raw_text == "Page twelve"
    assert transcript.segments[0].normalized_text == "page twelve"
    confidence = transcript.segments[0].confidence
    assert confidence is not None and math.isclose(confidence, 0.9048, abs_tol=0.001)


def test_second_decode_marks_numeric_disagreement_and_adjudication() -> None:
    primary = _transcript(_segment(1, 2, "page fifteen"), engine="sensor-a")
    secondary = _transcript(_segment(1, 2, "page fifty"), engine="sensor-b")

    unresolved = reconcile_transcripts(primary, secondary)
    resolved = reconcile_transcripts(primary, secondary, adjudicated={(1, 2): "fifteen"})

    assert unresolved[0].reason == "numeric disagreement"
    assert unresolved[0].resolved_text is None
    assert resolved[0].resolved_text == "fifteen"


def test_ingredient_mode_honors_section_and_exclusions() -> None:
    transcript = _transcript(
        _segment(0, 1, "Now the filling ingredients"),
        _segment(1, 2, "two cups chopped Apples, one teaspoon cinnamon"),
        _segment(2, 3, "crust ingredients"),
        _segment(3, 4, "butter and flour"),
        _segment(4, 5, "Bake the pie"),
    )

    result = extract_ingredients(
        transcript,
        section_start="filling ingredients",
        section_end="bake the pie",
    )

    assert result.ingredients == ("apples", "cinnamon")
    assert result.evidence_timestamps == (1, 1)


def test_numeric_page_mode_filters_context_deduplicates_and_sorts() -> None:
    transcript = _transcript(
        _segment(0, 1, "There were fifty birds"),
        _segment(1, 2, "See page forty two"),
        _segment(2, 3, "Pages 7 and page forty two"),
    )

    result = extract_numeric_pages(transcript)

    assert result.pages == (7, 42)
    assert result.evidence_timestamps == (2, 1)


def test_transcript_rejects_out_of_order_timestamps() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _transcript(_segment(2, 3, "later"), _segment(1, 2.5, "earlier"))
