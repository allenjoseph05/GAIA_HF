"""Runtime enforcement for code-owned, per-route tool capabilities and budgets."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from enum import StrEnum
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from gaia_max.observability import NoOpTelemetry, Telemetry
from gaia_max.retrieval.source_safety import ToolAuthorizationError, ToolPolicy
from gaia_max.solver_routing import SolverRoute

_JSON_ADAPTER = TypeAdapter(JsonValue)
ToolHandler = Callable[[Mapping[str, JsonValue]], Awaitable[JsonValue]]


class ToolDecision(StrEnum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    FAILED = "failed"


class ToolFailureCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    NOT_ALLOWED = "not_allowed"
    NOT_REGISTERED = "not_registered"
    CALL_BUDGET_EXHAUSTED = "call_budget_exhausted"
    INPUT_BUDGET_EXCEEDED = "input_budget_exceeded"
    OUTPUT_BUDGET_EXCEEDED = "output_budget_exceeded"
    TIMEOUT = "timeout"
    INVALID_OUTPUT = "invalid_output"
    TOOL_FAILED = "tool_failed"


class ToolBudget(BaseModel):
    """Non-negotiable limits owned by application code, never source text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_calls: int = Field(default=8, ge=1, le=1000)
    max_total_input_bytes: int = Field(default=64 * 1024, ge=1, le=100 * 1024 * 1024)
    max_output_bytes_per_call: int = Field(default=256 * 1024, ge=1, le=100 * 1024 * 1024)
    timeout_seconds: float = Field(default=30, gt=0, le=1800)
    max_concurrency: int = Field(default=2, ge=1, le=32)


class RouteToolPolicy(BaseModel):
    """Minimal capabilities and resource limits for one trusted solver route."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route: SolverRoute
    tools: ToolPolicy
    budget: ToolBudget = Field(default_factory=ToolBudget)


class ToolAuditEvent(BaseModel):
    """Content-free audit record safe for logs and public demonstrations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=1)
    route: SolverRoute
    tool_name: str = Field(min_length=1)
    decision: ToolDecision
    failure_code: ToolFailureCode | None = None
    input_bytes: int = Field(ge=0)
    output_bytes: int = Field(ge=0)
    elapsed_ms: int = Field(ge=0)
    remaining_calls: int = Field(ge=0)


class ToolExecutionError(RuntimeError):
    """Sanitized tool failure; underlying content and credentials are never included."""

    def __init__(self, code: ToolFailureCode) -> None:
        self.code = code
        super().__init__(f"tool execution blocked: {code.value}")


class EnforcedToolGateway:
    """Authorize, meter, bound, validate, and audit every tool invocation."""

    def __init__(
        self,
        *,
        policy: RouteToolPolicy,
        handlers: Mapping[str, ToolHandler],
        telemetry: Telemetry | None = None,
    ) -> None:
        unknown_handlers = set(handlers) - set(policy.tools.allowed_tools)
        if unknown_handlers:
            raise ValueError("handlers cannot exceed the code-owned tool allowlist")
        self.policy = policy
        self._telemetry = telemetry or NoOpTelemetry()
        self._handlers = dict(handlers)
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.BoundedSemaphore(policy.budget.max_concurrency)
        self._calls = 0
        self._total_input_bytes = 0
        self._audit_sequence = 0
        self._events: list[ToolAuditEvent] = []

    @property
    def audit_events(self) -> tuple[ToolAuditEvent, ...]:
        return tuple(sorted(self._events, key=lambda event: event.sequence))

    async def invoke(
        self,
        tool_name: str,
        payload: Mapping[str, JsonValue],
    ) -> JsonValue:
        """Invoke one registered tool after all authorization and budget checks."""

        with self._telemetry.span(
            "tool.invoke",
            {
                "tool.name": tool_name,
                "tool.route": self.policy.route.value,
            },
        ) as span:
            try:
                output = await self._invoke(tool_name, payload)
            except ToolExecutionError as exc:
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.failure_code", exc.code.value)
                raise
            span.set_attribute("tool.success", True)
            return output

    async def _invoke(
        self,
        tool_name: str,
        payload: Mapping[str, JsonValue],
    ) -> JsonValue:
        """Internal enforcement ladder kept separate from trace lifecycle."""

        started = time.monotonic()
        async with self._lock:
            self._audit_sequence += 1
            sequence = self._audit_sequence
        try:
            input_bytes = _json_size(cast(JsonValue, dict(payload)))
        except (TypeError, ValueError) as exc:
            self._record(
                sequence,
                tool_name,
                ToolDecision.BLOCKED,
                ToolFailureCode.INVALID_INPUT,
                0,
                0,
                started,
            )
            raise ToolExecutionError(ToolFailureCode.INVALID_INPUT) from exc
        async with self._lock:
            try:
                self.policy.tools.require_allowed(tool_name)
            except ToolAuthorizationError as exc:
                self._record(
                    sequence,
                    tool_name,
                    ToolDecision.BLOCKED,
                    ToolFailureCode.NOT_ALLOWED,
                    input_bytes,
                    0,
                    started,
                )
                raise ToolExecutionError(ToolFailureCode.NOT_ALLOWED) from exc
            handler = self._handlers.get(tool_name)
            if handler is None:
                self._record(
                    sequence,
                    tool_name,
                    ToolDecision.BLOCKED,
                    ToolFailureCode.NOT_REGISTERED,
                    input_bytes,
                    0,
                    started,
                )
                raise ToolExecutionError(ToolFailureCode.NOT_REGISTERED)
            if self._calls >= self.policy.budget.max_calls:
                self._record(
                    sequence,
                    tool_name,
                    ToolDecision.BLOCKED,
                    ToolFailureCode.CALL_BUDGET_EXHAUSTED,
                    input_bytes,
                    0,
                    started,
                )
                raise ToolExecutionError(ToolFailureCode.CALL_BUDGET_EXHAUSTED)
            if self._total_input_bytes + input_bytes > self.policy.budget.max_total_input_bytes:
                self._record(
                    sequence,
                    tool_name,
                    ToolDecision.BLOCKED,
                    ToolFailureCode.INPUT_BUDGET_EXCEEDED,
                    input_bytes,
                    0,
                    started,
                )
                raise ToolExecutionError(ToolFailureCode.INPUT_BUDGET_EXCEEDED)
            self._calls += 1
            self._total_input_bytes += input_bytes

        try:
            async with self._semaphore:
                raw_output = await asyncio.wait_for(
                    handler(payload),
                    timeout=self.policy.budget.timeout_seconds,
                )
        except TimeoutError as exc:
            self._record(
                sequence,
                tool_name,
                ToolDecision.FAILED,
                ToolFailureCode.TIMEOUT,
                input_bytes,
                0,
                started,
            )
            raise ToolExecutionError(ToolFailureCode.TIMEOUT) from exc
        except Exception as exc:
            self._record(
                sequence,
                tool_name,
                ToolDecision.FAILED,
                ToolFailureCode.TOOL_FAILED,
                input_bytes,
                0,
                started,
            )
            raise ToolExecutionError(ToolFailureCode.TOOL_FAILED) from exc

        try:
            output = cast(JsonValue, _JSON_ADAPTER.validate_python(raw_output))
            output_bytes = _json_size(output)
        except (TypeError, ValueError) as exc:
            self._record(
                sequence,
                tool_name,
                ToolDecision.FAILED,
                ToolFailureCode.INVALID_OUTPUT,
                input_bytes,
                0,
                started,
            )
            raise ToolExecutionError(ToolFailureCode.INVALID_OUTPUT) from exc
        if output_bytes > self.policy.budget.max_output_bytes_per_call:
            self._record(
                sequence,
                tool_name,
                ToolDecision.BLOCKED,
                ToolFailureCode.OUTPUT_BUDGET_EXCEEDED,
                input_bytes,
                output_bytes,
                started,
            )
            raise ToolExecutionError(ToolFailureCode.OUTPUT_BUDGET_EXCEEDED)
        self._record(
            sequence,
            tool_name,
            ToolDecision.ALLOWED,
            None,
            input_bytes,
            output_bytes,
            started,
        )
        return output

    def _record(
        self,
        sequence: int,
        tool_name: str,
        decision: ToolDecision,
        failure_code: ToolFailureCode | None,
        input_bytes: int,
        output_bytes: int,
        started: float,
    ) -> None:
        self._events.append(
            ToolAuditEvent(
                sequence=sequence,
                route=self.policy.route,
                tool_name=tool_name,
                decision=decision,
                failure_code=failure_code,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                elapsed_ms=max(0, int((time.monotonic() - started) * 1000)),
                remaining_calls=max(0, self.policy.budget.max_calls - self._calls),
            )
        )


def _json_size(value: JsonValue) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    )


__all__ = [
    "EnforcedToolGateway",
    "RouteToolPolicy",
    "ToolAuditEvent",
    "ToolBudget",
    "ToolDecision",
    "ToolExecutionError",
    "ToolFailureCode",
    "ToolHandler",
]
