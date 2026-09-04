from __future__ import annotations

from collections.abc import Sequence

import pytest

from gaia_max.media.asr import Transcript, TranscriptSegment
from gaia_max.media.youtube import (
    CaptionCue,
    CaptionTrack,
    YouTubeEvidenceError,
    acquire_youtube_evidence,
    parse_webvtt,
    parse_youtube_video_id,
    solve_exact_youtube_speech,
)

VIDEO_ID = "abcdefghijk"


def _track(kind: str = "manual") -> CaptionTrack:
    return CaptionTrack(
        video_id=VIDEO_ID,
        language="en",
        kind=kind,
        cues=(
            CaptionCue(start_seconds=10, end_seconds=12, text="What color was it?"),
            CaptionCue(start_seconds=12, end_seconds=14, text="It was bright blue."),
        ),
    )


class _Provider:
    def __init__(
        self,
        manual: CaptionTrack | None = None,
        automatic: CaptionTrack | None = None,
    ) -> None:
        self.manual = manual
        self.automatic = automatic
        self.calls: list[str] = []

    def manual_captions(
        self, video_id: str, languages: Sequence[str]
    ) -> CaptionTrack | None:
        self.calls.append("manual")
        return self.manual

    def automatic_captions(
        self, video_id: str, languages: Sequence[str]
    ) -> CaptionTrack | None:
        self.calls.append("automatic")
        return self.automatic


def _transcript(text: str) -> Transcript:
    return Transcript(
        engine="audio-sensor",
        duration_seconds=20,
        segments=(
            TranscriptSegment(
                start_seconds=12,
                end_seconds=14,
                raw_text=text,
                normalized_text=" ".join(text.casefold().split()),
            ),
        ),
    )


@pytest.mark.parametrize(
    "value",
    [
        VIDEO_ID,
        f"https://youtu.be/{VIDEO_ID}?t=10",
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com/shorts/{VIDEO_ID}",
        f"https://youtube.com/embed/{VIDEO_ID}",
    ],
)
def test_parse_youtube_video_id_variants(value: str) -> None:
    assert parse_youtube_video_id(value) == VIDEO_ID


def test_caption_ladder_prefers_manual_then_automatic_then_audio() -> None:
    manual = _Provider(manual=_track())
    automatic = _Provider(automatic=_track("automatic"))
    absent = _Provider()

    assert acquire_youtube_evidence(VIDEO_ID, manual).captions is not None
    assert manual.calls == ["manual"]
    assert acquire_youtube_evidence(VIDEO_ID, automatic).captions is not None
    assert automatic.calls == ["manual", "automatic"]
    assert acquire_youtube_evidence(VIDEO_ID, absent).audio_fallback_required


def test_exact_speech_requires_caption_asr_agreement() -> None:
    answer = solve_exact_youtube_speech(
        _track(),
        _transcript("It was bright blue!"),
        anchor="what color was it",
    )
    assert answer.text == "It was bright blue."
    assert answer.reply_start_seconds == 12
    assert not answer.adjudicated


def test_exact_speech_disagreement_requires_supported_adjudication() -> None:
    with pytest.raises(YouTubeEvidenceError, match="require adjudication"):
        solve_exact_youtube_speech(
            _track(),
            _transcript("It was light blue."),
            anchor="what color was it",
        )
    result = solve_exact_youtube_speech(
        _track(),
        _transcript("It was light blue."),
        anchor="what color was it",
        adjudicated_text="It was light blue.",
    )
    assert result.adjudicated


def test_exact_speech_rejects_unsupported_adjudication() -> None:
    with pytest.raises(YouTubeEvidenceError, match="unsupported"):
        solve_exact_youtube_speech(
            _track(),
            _transcript("It was light blue."),
            anchor="what color was it",
            adjudicated_text="It was green.",
        )


def test_parse_webvtt_preserves_timestamps_and_deduplicates_rolling_cues() -> None:
    cues = parse_webvtt(
        """WEBVTT

00:01.000 --> 00:02.500
<c>Hello there</c>

00:02.500 --> 00:03.000
Hello there

00:03.000 --> 00:04.000
General Kenobi
"""
    )
    assert [(cue.start_seconds, cue.end_seconds, cue.text) for cue in cues] == [
        (1, 2.5, "Hello there"),
        (3, 4, "General Kenobi"),
    ]
