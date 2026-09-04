"""Framework-neutral, answer-safe tracing with optional OTel and Langfuse sinks."""

from __future__ import annotations

import re
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from gaia_max.config import Settings

TelemetryValue = str | bool | int | float

_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_FORBIDDEN_KEY = re.compile(
    r"(?:answer|candidate|completion|content|credential|input|output|password|prompt|"
    r"question|secret|token)",
    re.IGNORECASE,
)


class UnsafeTelemetryAttribute(ValueError):
    """Raised before sensitive or high-cardinality content can enter a trace."""


class TraceSpan(Protocol):
    """Small span surface shared by no-op, OTel, and Langfuse implementations."""

    def set_attribute(self, key: str, value: TelemetryValue) -> None: ...


class Telemetry(Protocol):
    """Tracing port used by core code without importing an observability SDK."""

    def span(
        self,
        name: str,
        attributes: Mapping[str, TelemetryValue] | None = None,
    ) -> AbstractContextManager[TraceSpan]: ...


def safe_attributes(
    attributes: Mapping[str, TelemetryValue] | None,
) -> dict[str, TelemetryValue]:
    """Validate low-cardinality metadata; content fields are forbidden, not masked."""

    clean: dict[str, TelemetryValue] = {}
    for key, value in (attributes or {}).items():
        if _SAFE_NAME.fullmatch(key) is None or _FORBIDDEN_KEY.search(key):
            raise UnsafeTelemetryAttribute(f"telemetry attribute is forbidden: {key!r}")
        if isinstance(value, str) and len(value) > 200:
            raise UnsafeTelemetryAttribute(f"telemetry string is too long: {key!r}")
        clean[key] = value
    return clean


class _NoOpSpan:
    def set_attribute(self, key: str, value: TelemetryValue) -> None:
        safe_attributes({key: value})


class NoOpTelemetry:
    """Default sink: retains validation while producing no external side effects."""

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, TelemetryValue] | None = None,
    ) -> Generator[TraceSpan]:
        _validate_span_name(name)
        safe_attributes(attributes)
        yield _NoOpSpan()


class _OpenTelemetrySpan:
    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: TelemetryValue) -> None:
        clean = safe_attributes({key: value})
        self._span.set_attribute(key, clean[key])


class OpenTelemetry:
    """OpenTelemetry adapter with an isolated provider and injectable exporter."""

    def __init__(self, tracer: Any, *, provider: Any | None = None) -> None:
        self._tracer = tracer
        self._provider = provider

    @classmethod
    def from_exporter(
        cls,
        exporter: Any,
        *,
        service_name: str = "gaia-max",
    ) -> OpenTelemetry:
        """Create an isolated SDK provider; never mutate OTel's global provider."""

        Resource = import_module("opentelemetry.sdk.resources").Resource
        trace_sdk = import_module("opentelemetry.sdk.trace")
        SimpleSpanProcessor = import_module(
            "opentelemetry.sdk.trace.export"
        ).SimpleSpanProcessor

        provider = trace_sdk.TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return cls(provider.get_tracer("gaia_max", "1.0.0"), provider=provider)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, TelemetryValue] | None = None,
    ) -> Generator[TraceSpan]:
        _validate_span_name(name)
        clean = safe_attributes(attributes)
        with self._tracer.start_as_current_span(name, attributes=clean) as raw_span:
            yield _OpenTelemetrySpan(raw_span)

    def shutdown(self) -> None:
        if self._provider is not None:
            self._provider.shutdown()


class _LangfuseSpan:
    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def set_attribute(self, key: str, value: TelemetryValue) -> None:
        self._observation.update(metadata=safe_attributes({key: value}))


class LangfuseTelemetry:
    """Optional Langfuse v4 sink; v4 emits its observations as OTel spans."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def create(
        cls,
        *,
        public_key: str,
        secret_key: str,
        base_url: str | None = None,
        environment: str = "production",
    ) -> LangfuseTelemetry:
        Langfuse = import_module("langfuse").Langfuse

        def redact(_value: Any) -> str:
            return "[REDACTED]"

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            environment=environment,
            mask=redact,
        )
        return cls(client)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, TelemetryValue] | None = None,
    ) -> Generator[TraceSpan]:
        _validate_span_name(name)
        clean = safe_attributes(attributes)
        with self._client.start_as_current_observation(
            name=name,
            as_type="span",
            metadata=clean,
        ) as observation:
            yield _LangfuseSpan(observation)

    def shutdown(self) -> None:
        self._client.flush()


def _validate_span_name(name: str) -> None:
    if _SAFE_NAME.fullmatch(name) is None:
        raise UnsafeTelemetryAttribute(f"invalid telemetry span name: {name!r}")


def telemetry_from_settings(
    settings: Settings,
    *,
    otel_exporter: Any | None = None,
) -> Telemetry:
    """Build the configured sink without ever placing credentials in span metadata."""

    if settings.telemetry_backend == "none":
        return NoOpTelemetry()
    if settings.telemetry_backend == "otel":
        if otel_exporter is None:
            raise ValueError("OTel telemetry requires an explicit span exporter")
        return OpenTelemetry.from_exporter(otel_exporter)
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        raise ValueError("Langfuse telemetry credentials are unavailable")
    return LangfuseTelemetry.create(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=str(settings.langfuse_base_url),
    )


__all__ = [
    "LangfuseTelemetry",
    "NoOpTelemetry",
    "OpenTelemetry",
    "Telemetry",
    "TelemetryValue",
    "TraceSpan",
    "UnsafeTelemetryAttribute",
    "safe_attributes",
    "telemetry_from_settings",
]
