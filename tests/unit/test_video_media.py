from __future__ import annotations

import subprocess
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from gaia_max.artifacts import ArtifactStore
from gaia_max.domain import ArtifactSource
from gaia_max.media.video import (
    BoundingRegion,
    FrameSpeciesObservation,
    SpeciesDetection,
    VideoAnalysisError,
    build_contact_sheet,
    extract_frames,
    frame_sample_times,
    reconcile_species_maximum,
)


def _observation(
    sensor: str,
    timestamp: float,
    artifact_id: str,
    species: tuple[str, ...],
) -> FrameSpeciesObservation:
    return FrameSpeciesObservation(
        sensor=sensor,
        timestamp_seconds=timestamp,
        frame_artifact_id=artifact_id,
        detections=tuple(
            SpeciesDetection(
                species=name,
                region=BoundingRegion(left=0, top=0, right=0.5, bottom=0.5),
                confidence=0.9,
            )
            for name in species
        ),
    )


def test_coarse_and_dense_frame_schedules_cover_requested_window() -> None:
    assert frame_sample_times(3, frames_per_second=1) == (0, 1, 2)
    assert frame_sample_times(3, frames_per_second=4, window=(1, 2)) == (
        1,
        1.25,
        1.5,
        1.75,
    )


def test_extract_frames_and_contact_sheet_from_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic.mp4"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=4:d=2",
            "-c:v",
            "mpeg4",
            str(video_path),
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
    store = ArtifactStore(tmp_path / "artifacts")
    source = store.put_bytes(
        video_path.read_bytes(),
        original_name="synthetic.mp4",
        media_type="video/mp4",
        source=ArtifactSource.FIXTURE,
    )

    frames = extract_frames(store, source, (0, 0.5, 1.0, 1.5))
    sheet = build_contact_sheet(store, frames, columns=2)

    assert len(frames) == 4
    with Image.open(BytesIO(store.read_bytes(sheet))) as image:
        assert image.width > 64 and image.height > 64


def test_species_maximum_deduplicates_repeated_individuals() -> None:
    artifact = "a" * 64
    primary = (
        _observation("sensor-a", 0, artifact, ("lion",)),
        _observation("sensor-a", 1, artifact, ("lion", "lion", "zebra")),
    )
    secondary = (
        _observation("sensor-b", 0, artifact, ("lion",)),
        _observation("sensor-b", 1, artifact, ("zebra", "lion")),
    )

    result = reconcile_species_maximum(primary, secondary)

    assert result.maximum_distinct_species == 2
    assert result.species == ("lion", "zebra")
    assert result.timestamp_seconds == 1
    assert not result.dense_resampling_required


def test_species_disagreement_triggers_dense_sampling_until_adjudicated() -> None:
    artifact = "b" * 64
    primary = (_observation("sensor-a", 1, artifact, ("lion", "zebra")),)
    secondary = (_observation("sensor-b", 1, artifact, ("lion", "gazelle")),)

    unresolved = reconcile_species_maximum(primary, secondary)
    resolved = reconcile_species_maximum(
        primary,
        secondary,
        adjudicated={1: ("lion", "zebra")},
    )

    assert unresolved.dense_resampling_required
    assert unresolved.disputed_timestamps == (1,)
    assert not resolved.dense_resampling_required
    assert resolved.maximum_distinct_species == 2


def test_species_adjudication_cannot_invent_species() -> None:
    artifact = "c" * 64
    first = (_observation("sensor-a", 1, artifact, ("lion",)),)
    second = (_observation("sensor-b", 1, artifact, ("zebra",)),)
    with pytest.raises(VideoAnalysisError, match="unsupported"):
        reconcile_species_maximum(first, second, adjudicated={1: ("tiger",)})


def test_structured_detection_rejects_bare_numeric_species() -> None:
    with pytest.raises(ValueError):
        SpeciesDetection(
            species="3",
            region=BoundingRegion(left=0, top=0, right=1, bottom=1),
            confidence=0.5,
        )
