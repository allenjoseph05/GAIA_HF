"""Timestamped frame sampling and independent structured species analysis."""

from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaia_max.artifacts import ArtifactStore
from gaia_max.domain import ArtifactRef, ArtifactSource
from gaia_max.domain.models import SHA256_PATTERN
from gaia_max.media.core import MediaProcessingError, probe_media


class VideoAnalysisError(ValueError):
    """Visual evidence is incomplete, unsupported, or unresolved."""


class FrameArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp_seconds: float = Field(ge=0)
    artifact_id: str = Field(pattern=SHA256_PATTERN)


class BoundingRegion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    left: float = Field(ge=0, le=1)
    top: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_area(self) -> BoundingRegion:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("bounding region must have positive area")
        return self


class SpeciesDetection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    species: str = Field(min_length=2, pattern=r"^[^\d]+$")
    region: BoundingRegion
    confidence: float = Field(ge=0, le=1)


class FrameSpeciesObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sensor: str = Field(min_length=1)
    timestamp_seconds: float = Field(ge=0)
    frame_artifact_id: str = Field(pattern=SHA256_PATTERN)
    detections: tuple[SpeciesDetection, ...]

    @property
    def species(self) -> frozenset[str]:
        return frozenset(item.species.casefold().strip() for item in self.detections)


class VisualSensor(Protocol):
    name: str

    def inspect(self, frame: FrameArtifact) -> FrameSpeciesObservation: ...


class SpeciesMaximum(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    maximum_distinct_species: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    species: tuple[str, ...]
    sensors: tuple[str, str]
    dense_resampling_required: bool = False
    disputed_timestamps: tuple[float, ...] = ()


def frame_sample_times(
    duration_seconds: float,
    *,
    frames_per_second: float = 1,
    window: tuple[float, float] | None = None,
) -> tuple[float, ...]:
    if duration_seconds <= 0 or frames_per_second <= 0:
        raise ValueError("duration and sample rate must be positive")
    start, end = window or (0.0, duration_seconds)
    if start < 0 or end <= start or end > duration_seconds + 1e-6:
        raise ValueError("frame sampling window is invalid")
    count = math.ceil((end - start) * frames_per_second)
    return tuple(round(start + index / frames_per_second, 6) for index in range(count))


def extract_frames(
    store: ArtifactStore,
    video: ArtifactRef,
    timestamps: Sequence[float],
    *,
    ffmpeg: str = "ffmpeg",
) -> tuple[FrameArtifact, ...]:
    probe = probe_media(video.local_path)
    if not probe.has_video:
        raise MediaProcessingError("media has no video stream")
    if not timestamps or any(time < 0 or time >= probe.duration_seconds for time in timestamps):
        raise VideoAnalysisError("frame timestamps must lie within the video duration")
    frames: list[FrameArtifact] = []
    with tempfile.TemporaryDirectory(prefix="gaia-frames-") as temporary:
        for index, timestamp in enumerate(timestamps):
            output = Path(temporary) / f"frame-{index:06d}.png"
            command = [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(video.local_path.resolve()),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-f",
                "image2",
                str(output),
            ]
            completed = subprocess.run(command, capture_output=True, check=False, timeout=120)
            if completed.returncode or not output.is_file():
                raise MediaProcessingError(f"could not extract frame at {timestamp:.3f}s")
            artifact = store.put_bytes(
                output.read_bytes(),
                original_name=f"frame-{timestamp:.3f}.png",
                media_type="image/png",
                source=ArtifactSource.DERIVED,
                source_locator=f"artifact:{video.artifact_id}#t={timestamp:.6f}",
            )
            frames.append(
                FrameArtifact(timestamp_seconds=timestamp, artifact_id=artifact.artifact_id)
            )
    return tuple(frames)


def build_contact_sheet(
    store: ArtifactStore,
    frames: Sequence[FrameArtifact],
    *,
    columns: int = 4,
) -> ArtifactRef:
    if not frames or columns <= 0:
        raise ValueError("contact sheet requires frames and positive columns")
    loaded: list[Image.Image] = []
    try:
        for frame in frames:
            artifact = store.get(frame.artifact_id)
            image = Image.open(BytesIO(store.read_bytes(artifact))).convert("RGB")
            image.thumbnail((320, 180))
            loaded.append(image)
        width = max(image.width for image in loaded)
        height = max(image.height for image in loaded) + 24
        rows = math.ceil(len(loaded) / columns)
        sheet = Image.new("RGB", (width * columns, height * rows), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (frame, image) in enumerate(zip(frames, loaded, strict=True)):
            x = (index % columns) * width
            y = (index // columns) * height
            sheet.paste(image, (x, y))
            draw.text(
                (x + 3, y + image.height + 3),
                f"{frame.timestamp_seconds:.3f}s",
                fill="black",
            )
        buffer = BytesIO()
        sheet.save(buffer, format="PNG")
        return store.put_bytes(
            buffer.getvalue(),
            original_name="contact-sheet.png",
            media_type="image/png",
            source=ArtifactSource.DERIVED,
            source_locator="frames:" + ",".join(frame.artifact_id for frame in frames),
        )
    finally:
        for image in loaded:
            image.close()


def reconcile_species_maximum(
    primary: Sequence[FrameSpeciesObservation],
    secondary: Sequence[FrameSpeciesObservation],
    *,
    adjudicated: Mapping[float, Sequence[str]] | None = None,
) -> SpeciesMaximum:
    """Require two sensor identities; disagreements trigger targeted dense review."""

    if not primary or not secondary:
        raise VideoAnalysisError("species analysis requires two complete vision passes")
    primary_sensor = _single_sensor(primary)
    secondary_sensor = _single_sensor(secondary)
    if primary_sensor == secondary_sensor:
        raise VideoAnalysisError("vision passes must have independent sensor identities")
    first = {item.timestamp_seconds: item.species for item in primary}
    second = {item.timestamp_seconds: item.species for item in secondary}
    if set(first) != set(second):
        raise VideoAnalysisError("vision passes must inspect identical timestamps")
    resolutions = adjudicated or {}
    disputes = tuple(sorted(time for time in first if first[time] != second[time]))
    if set(resolutions) - set(disputes):
        raise VideoAnalysisError("adjudication contains a non-disputed timestamp")
    resolved: dict[float, frozenset[str]] = {}
    for timestamp in first:
        if timestamp not in disputes:
            resolved[timestamp] = first[timestamp]
        elif timestamp in resolutions:
            supported = frozenset(name.casefold().strip() for name in resolutions[timestamp])
            if not supported.issubset(first[timestamp] | second[timestamp]):
                raise VideoAnalysisError("adjudication introduced an unsupported species")
            resolved[timestamp] = supported
    if not resolved:
        return SpeciesMaximum(
            maximum_distinct_species=0,
            timestamp_seconds=min(first),
            species=(),
            sensors=(primary_sensor, secondary_sensor),
            dense_resampling_required=True,
            disputed_timestamps=disputes,
        )
    best_time = max(resolved, key=lambda time: (len(resolved[time]), -time))
    return SpeciesMaximum(
        maximum_distinct_species=len(resolved[best_time]),
        timestamp_seconds=best_time,
        species=tuple(sorted(resolved[best_time])),
        sensors=(primary_sensor, secondary_sensor),
        dense_resampling_required=bool(set(disputes) - set(resolutions)),
        disputed_timestamps=disputes,
    )


def _single_sensor(observations: Sequence[FrameSpeciesObservation]) -> str:
    sensors = {item.sensor for item in observations}
    if len(sensors) != 1:
        raise VideoAnalysisError("one vision pass must use one stable sensor identity")
    return next(iter(sensors))


__all__ = [
    "BoundingRegion",
    "FrameArtifact",
    "FrameSpeciesObservation",
    "SpeciesDetection",
    "SpeciesMaximum",
    "VideoAnalysisError",
    "VisualSensor",
    "build_contact_sheet",
    "extract_frames",
    "frame_sample_times",
    "reconcile_species_maximum",
]
