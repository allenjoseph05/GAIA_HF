"""Safe ffprobe/ffmpeg media inspection and reproducible audio normalization."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaia_max.artifacts import ArtifactStore
from gaia_max.domain import ArtifactRef, ArtifactSource
from gaia_max.domain.models import SHA256_PATTERN


class MediaProcessingError(RuntimeError):
    """Media is unsafe, malformed, unsupported, or could not be processed."""


class MediaStream(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    codec_type: str = Field(pattern=r"^(audio|video)$")
    codec_name: str = Field(min_length=1)
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    frame_rate: float | None = Field(default=None, gt=0)


class MediaProbe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    duration_seconds: float = Field(gt=0)
    format_name: str = Field(min_length=1)
    streams: tuple[MediaStream, ...] = Field(min_length=1)

    @property
    def has_audio(self) -> bool:
        return any(stream.codec_type == "audio" for stream in self.streams)

    @property
    def has_video(self) -> bool:
        return any(stream.codec_type == "video" for stream in self.streams)


class DerivedMedia(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    artifact: ArtifactRef
    source_artifact_id: str = Field(pattern=SHA256_PATTERN)
    timestamp_origin_seconds: float = Field(default=0, ge=0)
    probe: MediaProbe

    @model_validator(mode="after")
    def require_derived_source(self) -> DerivedMedia:
        if self.artifact.source is not ArtifactSource.DERIVED:
            raise ValueError("normalized media must be stored as a derived artifact")
        return self


def probe_media(path: Path, *, ffprobe: str = "ffprobe") -> MediaProbe:
    """Probe one local file without a shell and validate the returned structure."""

    source = path.resolve()
    if not source.is_file():
        raise MediaProcessingError(f"media file does not exist: {source}")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name:stream=index,codec_type,codec_name,sample_rate,channels,width,height,r_frame_rate",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaProcessingError("ffprobe could not run") from exc
    if completed.returncode:
        raise MediaProcessingError("ffprobe rejected the media file")
    try:
        decoded: object = json.loads(completed.stdout)
        payload = _object_dict(decoded)
        format_payload = _object_dict(payload["format"])
        duration = float(str(format_payload["duration"]))
        format_name = str(format_payload["format_name"])
        raw_streams_value = payload["streams"]
        if not isinstance(raw_streams_value, list):
            raise TypeError("streams must be a list")
        raw_streams = cast(list[object], raw_streams_value)
        streams = tuple(
            _parse_stream(stream)
            for item in raw_streams
            if (stream := _object_dict(item)).get("codec_type") in {"audio", "video"}
        )
        return MediaProbe(duration_seconds=duration, format_name=format_name, streams=streams)
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaProcessingError("ffprobe returned incomplete metadata") from exc


def normalize_audio(
    store: ArtifactStore,
    source: ArtifactRef,
    *,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = 16_000,
    max_duration_seconds: float = 14_400,
) -> DerivedMedia:
    """Create deterministic mono signed-16-bit PCM while retaining time origin zero."""

    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    input_probe = probe_media(source.local_path)
    if not input_probe.has_audio:
        raise MediaProcessingError("media has no audio stream")
    if input_probe.duration_seconds > max_duration_seconds:
        raise MediaProcessingError("media exceeds the configured duration limit")
    with tempfile.TemporaryDirectory(prefix="gaia-media-") as temporary:
        output = Path(temporary) / "normalized.wav"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source.local_path.resolve()),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            "-map_metadata",
            "-1",
            str(output),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, check=False, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MediaProcessingError("ffmpeg normalization could not run") from exc
        if completed.returncode or not output.is_file():
            raise MediaProcessingError("ffmpeg could not normalize the audio stream")
        artifact = store.put_bytes(
            output.read_bytes(),
            original_name=f"normalized-{source.artifact_id[:12]}.wav",
            media_type="audio/wav",
            source=ArtifactSource.DERIVED,
            source_locator=f"artifact:{source.artifact_id}#audio=0&origin=0",
        )
    normalized_probe = probe_media(artifact.local_path)
    audio = next(stream for stream in normalized_probe.streams if stream.codec_type == "audio")
    if audio.sample_rate != sample_rate or audio.channels != 1 or audio.codec_name != "pcm_s16le":
        raise MediaProcessingError("normalized audio failed codec invariants")
    if abs(normalized_probe.duration_seconds - input_probe.duration_seconds) > 0.1:
        raise MediaProcessingError("normalization did not preserve the media timeline")
    return DerivedMedia(
        artifact=artifact,
        source_artifact_id=source.artifact_id,
        probe=normalized_probe,
    )


def _parse_stream(item: dict[str, object]) -> MediaStream:
    codec_type = item.get("codec_type")
    if codec_type not in {"audio", "video"}:
        raise ValueError("ignore non-audio/video stream")
    if not isinstance(codec_type, str):
        raise TypeError("codec type must be text")
    return MediaStream(
        index=int(str(item["index"])),
        codec_type=codec_type,
        codec_name=str(item["codec_name"]),
        sample_rate=_optional_int(item.get("sample_rate")),
        channels=_optional_int(item.get("channels")),
        width=_optional_int(item.get("width")),
        height=_optional_int(item.get("height")),
        frame_rate=_frame_rate(item.get("r_frame_rate")),
    )


def _optional_int(value: object) -> int | None:
    return None if value in {None, ""} else int(str(value))


def _frame_rate(value: object) -> float | None:
    if value in {None, "", "0/0"}:
        return None
    text = str(value)
    if "/" not in text:
        return float(text)
    numerator, denominator = text.split("/", maxsplit=1)
    denominator_value = float(denominator)
    return None if denominator_value == 0 else float(numerator) / denominator_value


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("JSON value must be an object with string keys")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError("JSON value must be an object with string keys")
    return {cast(str, key): item for key, item in mapping.items()}


__all__ = [
    "DerivedMedia",
    "MediaProbe",
    "MediaProcessingError",
    "MediaStream",
    "normalize_audio",
    "probe_media",
]
