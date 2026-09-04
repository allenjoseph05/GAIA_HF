"""Shared strict LangGraph checkpoint serialization helpers."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def build_safe_checkpoint_serializer(
    allowed_types: tuple[type[Any], ...],
) -> JsonPlusSerializer:
    """Build an allowlisted serializer with pickle fallback permanently disabled."""

    unique_types = tuple(dict.fromkeys(allowed_types))
    allowed_keys = tuple(
        (allowed_type.__module__, allowed_type.__name__)
        for allowed_type in unique_types
    )
    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=allowed_keys,
        allowed_msgpack_modules=unique_types,
    )
