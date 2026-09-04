from __future__ import annotations

from contextlib import contextmanager

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gaia_max.observability import (
    LangfuseTelemetry,
    NoOpTelemetry,
    OpenTelemetry,
    UnsafeTelemetryAttribute,
)


def test_noop_sink_still_blocks_answer_and_prompt_content() -> None:
    telemetry = NoOpTelemetry()
    with (
        pytest.raises(UnsafeTelemetryAttribute, match="forbidden"),
        telemetry.span("research.run", {"candidate_answer": "secret"}),
    ):
        pass

    with (
        telemetry.span("research.run", {"research.attempt": 1}) as span,
        pytest.raises(UnsafeTelemetryAttribute),
    ):
        span.set_attribute("model.prompt", "do not trace me")


def test_opentelemetry_exports_only_validated_metadata() -> None:
    exporter = InMemorySpanExporter()
    telemetry = OpenTelemetry.from_exporter(exporter, service_name="gaia-test")

    with telemetry.span(
        "research.retrieve",
        {"retrieval.top_k": 3, "retrieval.dense_enabled": False},
    ) as span:
        span.set_attribute("retrieval.result_count", 2)

    telemetry.shutdown()
    exported = exporter.get_finished_spans()
    assert len(exported) == 1
    assert exported[0].name == "research.retrieve"
    assert exported[0].attributes == {
        "retrieval.top_k": 3,
        "retrieval.dense_enabled": False,
        "retrieval.result_count": 2,
    }


class FakeObservation:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []

    def update(self, **kwargs: object) -> None:
        self.updates.append(kwargs)


class FakeLangfuse:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.observation = FakeObservation()
        self.flushed = False

    @contextmanager
    def start_as_current_observation(self, **kwargs: object):
        self.calls.append(kwargs)
        yield self.observation

    def flush(self) -> None:
        self.flushed = True


def test_langfuse_adapter_uses_content_free_otel_metadata() -> None:
    client = FakeLangfuse()
    telemetry = LangfuseTelemetry(client)

    with telemetry.span("research.agent.run", {"research.max_steps": 8}) as span:
        span.set_attribute("research.agent_success", True)
    telemetry.shutdown()

    assert client.calls == [
        {
            "name": "research.agent.run",
            "as_type": "span",
            "metadata": {"research.max_steps": 8},
        }
    ]
    assert client.observation.updates == [
        {"metadata": {"research.agent_success": True}}
    ]
    assert client.flushed
