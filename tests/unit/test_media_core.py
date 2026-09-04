from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path

import pytest

from gaia_max.artifacts import ArtifactStore
from gaia_max.domain import ArtifactSource
from gaia_max.media.core import MediaProcessingError, normalize_audio, probe_media


def _stereo_wave(*, seconds: float = 0.25, rate: int = 8_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(rate)
        frames = bytearray()
        for index in range(round(seconds * rate)):
            sample = round(10_000 * math.sin(2 * math.pi * 440 * index / rate))
            frames.extend(struct.pack("<hh", sample, sample))
        output.writeframes(bytes(frames))
    return buffer.getvalue()


def test_probe_and_normalize_audio_reproducibly(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    source = store.put_bytes(
        _stereo_wave(),
        original_name="input.wav",
        media_type="audio/wav",
        source=ArtifactSource.FIXTURE,
    )

    original = probe_media(source.local_path)
    first = normalize_audio(store, source)
    second = normalize_audio(store, source)

    assert original.has_audio and not original.has_video
    assert first.artifact.artifact_id == second.artifact.artifact_id
    assert first.source_artifact_id == source.artifact_id
    stream = first.probe.streams[0]
    assert (stream.codec_name, stream.sample_rate, stream.channels) == ("pcm_s16le", 16_000, 1)
    assert first.timestamp_origin_seconds == 0


def test_probe_rejects_missing_media(tmp_path: Path) -> None:
    with pytest.raises(MediaProcessingError, match="does not exist"):
        probe_media(tmp_path / "missing.wav")


def test_normalize_rejects_duration_limit(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    source = store.put_bytes(
        _stereo_wave(),
        original_name="input.wav",
        media_type="audio/wav",
        source=ArtifactSource.FIXTURE,
    )
    with pytest.raises(MediaProcessingError, match="duration"):
        normalize_audio(store, source, max_duration_seconds=0.1)
