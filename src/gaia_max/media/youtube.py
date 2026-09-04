"""YouTube identity, caption-first acquisition, and exact-speech reconciliation."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaia_max.media.asr import Transcript, TranscriptSegment

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


class YouTubeEvidenceError(ValueError):
    """YouTube evidence is missing, ambiguous, or not independently reconciled."""


class CaptionCue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_interval(self) -> CaptionCue:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("caption cue end must follow start")
        return self


class CaptionTrack(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")
    language: str = Field(min_length=2)
    kind: str = Field(pattern=r"^(manual|automatic)$")
    cues: tuple[CaptionCue, ...] = Field(min_length=1)


class YouTubeEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")
    captions: CaptionTrack | None = None
    audio_fallback_required: bool


class ExactSpeechAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    anchor_timestamp_seconds: float = Field(ge=0)
    reply_start_seconds: float = Field(ge=0)
    caption_text: str = Field(min_length=1)
    asr_text: str = Field(min_length=1)
    adjudicated: bool = False


class CaptionProvider(Protocol):
    def manual_captions(self, video_id: str, languages: Sequence[str]) -> CaptionTrack | None: ...

    def automatic_captions(
        self, video_id: str, languages: Sequence[str]
    ) -> CaptionTrack | None: ...


class YtDlpClient:
    """Shell-free yt-dlp adapter with bounded paths and reusable downloaded evidence."""

    def __init__(self, cache_root: Path, *, executable: str = "yt-dlp") -> None:
        self.cache_root = cache_root.resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.executable = executable

    def manual_captions(
        self, video_id: str, languages: Sequence[str]
    ) -> CaptionTrack | None:
        return self._captions(video_id, languages, automatic=False)

    def automatic_captions(
        self, video_id: str, languages: Sequence[str]
    ) -> CaptionTrack | None:
        return self._captions(video_id, languages, automatic=True)

    def download_media(self, video_id: str) -> Path:
        checked = parse_youtube_video_id(video_id)
        video_root = (self.cache_root / checked / "media").resolve()
        self._assert_in_cache(video_root)
        video_root.mkdir(parents=True, exist_ok=True)
        existing = tuple(path for path in video_root.glob("media.*") if path.is_file())
        if len(existing) == 1:
            return existing[0]
        command = [
            self.executable,
            "--no-playlist",
            "--no-progress",
            "--restrict-filenames",
            "--format",
            "bestvideo*+bestaudio/best",
            "--merge-output-format",
            "mkv",
            "--output",
            str(video_root / "media.%(ext)s"),
            "--print",
            "after_move:filepath",
            f"https://www.youtube.com/watch?v={checked}",
        ]
        completed = self._run(command, timeout=1800)
        output_lines = completed.stdout.decode("utf-8", errors="replace").splitlines()
        if not output_lines:
            raise YouTubeEvidenceError("yt-dlp did not report a downloaded media path")
        output = Path(output_lines[-1].strip()).resolve()
        self._assert_in_cache(output)
        if not output.is_file():
            raise YouTubeEvidenceError("yt-dlp media output is missing")
        return output

    def _captions(
        self,
        video_id: str,
        languages: Sequence[str],
        *,
        automatic: bool,
    ) -> CaptionTrack | None:
        checked = parse_youtube_video_id(video_id)
        if not languages:
            raise ValueError("caption acquisition requires at least one language")
        kind = "automatic" if automatic else "manual"
        track_root = (self.cache_root / checked / kind).resolve()
        self._assert_in_cache(track_root)
        track_root.mkdir(parents=True, exist_ok=True)
        existing = tuple(sorted(track_root.glob("track.*.vtt")))
        if not existing:
            switch = "--write-auto-subs" if automatic else "--write-subs"
            command = [
                self.executable,
                "--no-playlist",
                "--no-progress",
                "--skip-download",
                switch,
                "--sub-format",
                "vtt",
                "--sub-langs",
                ",".join(languages),
                "--output",
                str(track_root / "track.%(ext)s"),
                f"https://www.youtube.com/watch?v={checked}",
            ]
            completed = self._run(command, timeout=300, allow_unavailable=True)
            if completed.returncode:
                return None
            existing = tuple(sorted(track_root.glob("track.*.vtt")))
        if not existing:
            return None
        track_path = existing[0].resolve()
        self._assert_in_cache(track_path)
        language = track_path.stem.split(".")[-1]
        return CaptionTrack(
            video_id=checked,
            language=language,
            kind=kind,
            cues=parse_webvtt(track_path.read_text(encoding="utf-8-sig")),
        )

    def _run(
        self,
        command: Sequence[str],
        *,
        timeout: float,
        allow_unavailable: bool = False,
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise YouTubeEvidenceError("yt-dlp could not run") from exc
        if completed.returncode and not allow_unavailable:
            raise YouTubeEvidenceError("yt-dlp could not download the requested media")
        return completed

    def _assert_in_cache(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.cache_root)
        except ValueError as exc:
            raise YouTubeEvidenceError("yt-dlp output escaped the media cache") from exc


def parse_webvtt(text: str) -> tuple[CaptionCue, ...]:
    """Parse timestamped VTT cues and remove adjacent rolling-caption duplicates."""

    timestamp = re.compile(
        r"(?P<start>\d{2}:\d{2}(?::\d{2})?\.\d{3})\s+-->\s+"
        r"(?P<end>\d{2}:\d{2}(?::\d{2})?\.\d{3})"
    )
    lines = text.replace("\r\n", "\n").split("\n")
    cues: list[CaptionCue] = []
    index = 0
    while index < len(lines):
        match = timestamp.search(lines[index])
        if match is None:
            index += 1
            continue
        start = _vtt_seconds(match.group("start"))
        end = _vtt_seconds(match.group("end"))
        index += 1
        content: list[str] = []
        while index < len(lines) and lines[index].strip():
            cleaned = re.sub(r"<[^>]+>", "", lines[index]).strip()
            if cleaned:
                content.append(cleaned)
            index += 1
        cue_text = " ".join(content)
        if cue_text and (not cues or cue_text != cues[-1].text):
            cues.append(CaptionCue(start_seconds=start, end_seconds=end, text=cue_text))
    if not cues:
        raise YouTubeEvidenceError("caption file contains no timestamped cues")
    return tuple(cues)


def _vtt_seconds(value: str) -> float:
    parts = value.split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2])
    hours = int(parts[-3]) if len(parts) == 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def parse_youtube_video_id(value: str) -> str:
    """Parse canonical, short, shorts, embed, or bare YouTube video IDs."""

    candidate = value.strip()
    if _VIDEO_ID.fullmatch(candidate):
        return candidate
    parsed = urlparse(candidate)
    host = (parsed.hostname or "").casefold()
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/", maxsplit=1)[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
            candidate = parts[1]
        else:
            candidate = ""
    else:
        candidate = ""
    if not _VIDEO_ID.fullmatch(candidate):
        raise YouTubeEvidenceError("value does not contain a valid YouTube video ID")
    return candidate


def acquire_youtube_evidence(
    value: str,
    provider: CaptionProvider,
    *,
    languages: Sequence[str] = ("en",),
) -> YouTubeEvidence:
    video_id = parse_youtube_video_id(value)
    manual = provider.manual_captions(video_id, languages)
    if manual is not None:
        _validate_track(manual, video_id, expected_kind="manual")
        return YouTubeEvidence(
            video_id=video_id,
            captions=manual,
            audio_fallback_required=False,
        )
    automatic = provider.automatic_captions(video_id, languages)
    if automatic is not None:
        _validate_track(automatic, video_id, expected_kind="automatic")
        return YouTubeEvidence(
            video_id=video_id,
            captions=automatic,
            audio_fallback_required=False,
        )
    return YouTubeEvidence(video_id=video_id, audio_fallback_required=True)


def solve_exact_youtube_speech(
    captions: CaptionTrack,
    transcript: Transcript,
    *,
    anchor: str,
    adjudicated_text: str | None = None,
    context_seconds: float = 15,
) -> ExactSpeechAnswer:
    """Return the cue after an anchor only when caption and ASR evidence reconcile."""

    anchor_tokens = _comparison_text(anchor)
    anchor_indexes = [
        index
        for index, cue in enumerate(captions.cues)
        if anchor_tokens in _comparison_text(cue.text)
    ]
    if len(anchor_indexes) != 1:
        raise YouTubeEvidenceError("speech anchor must match exactly one caption cue")
    anchor_index = anchor_indexes[0]
    if anchor_index + 1 >= len(captions.cues):
        raise YouTubeEvidenceError("anchor has no following caption reply")
    reply = captions.cues[anchor_index + 1]
    if reply.start_seconds - captions.cues[anchor_index].end_seconds > context_seconds:
        raise YouTubeEvidenceError("following caption is outside the anchor context window")
    matching_segments = [
        segment
        for segment in transcript.segments
        if _overlaps(segment, reply.start_seconds, reply.end_seconds)
    ]
    if not matching_segments:
        raise YouTubeEvidenceError("ASR has no timestamp evidence for the caption reply")
    asr_text = " ".join(segment.raw_text.strip() for segment in matching_segments)
    caption_comparison = _comparison_text(reply.text)
    asr_comparison = _comparison_text(asr_text)
    if caption_comparison == asr_comparison:
        result = reply.text.strip()
        adjudicated = False
    else:
        if adjudicated_text is None:
            raise YouTubeEvidenceError("caption and ASR reply disagree and require adjudication")
        adjudicated_comparison = _comparison_text(adjudicated_text)
        if adjudicated_comparison not in {caption_comparison, asr_comparison}:
            raise YouTubeEvidenceError("adjudication is unsupported by either evidence path")
        result = adjudicated_text.strip()
        adjudicated = True
    return ExactSpeechAnswer(
        text=result,
        anchor_timestamp_seconds=captions.cues[anchor_index].start_seconds,
        reply_start_seconds=reply.start_seconds,
        caption_text=reply.text,
        asr_text=asr_text,
        adjudicated=adjudicated,
    )


def _validate_track(track: CaptionTrack, video_id: str, *, expected_kind: str) -> None:
    if track.video_id != video_id or track.kind != expected_kind:
        raise YouTubeEvidenceError("caption track identity or kind does not match acquisition")


def _comparison_text(text: str) -> str:
    return " ".join(re.findall(r"[\w']+", text.casefold()))


def _overlaps(segment: TranscriptSegment, start: float, end: float) -> bool:
    return min(segment.end_seconds, end) > max(segment.start_seconds, start)


__all__ = [
    "CaptionCue",
    "CaptionProvider",
    "CaptionTrack",
    "ExactSpeechAnswer",
    "YouTubeEvidence",
    "YouTubeEvidenceError",
    "YtDlpClient",
    "acquire_youtube_evidence",
    "parse_webvtt",
    "parse_youtube_video_id",
    "solve_exact_youtube_speech",
]
