"""Production composition root for the optional deep-research capability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gaia_max.observability import NoOpTelemetry, Telemetry
from gaia_max.research.llamaindex_retrieval import (
    LlamaIndexDocumentRetriever,
    LlamaIndexRetrievalToolHandler,
    PassageCatalog,
)
from gaia_max.research.models import IndexableDocument
from gaia_max.research.smolagents_backend import SmolagentsResearchBackend
from gaia_max.research.solver import DeepResearchSolver, ResearchLedger
from gaia_max.retrieval.source_safety import ToolPolicy
from gaia_max.solver_routing import SolverRoute
from gaia_max.tool_enforcement import (
    EnforcedToolGateway,
    RouteToolPolicy,
    ToolBudget,
    ToolHandler,
)

DOCUMENT_RETRIEVAL_TOOL = "retrieve_guarded_documents"


@dataclass(frozen=True, slots=True)
class DeepResearchComponents:
    """Owned components returned together so their run-local trust state stays aligned."""

    solver: DeepResearchSolver
    backend: SmolagentsResearchBackend
    retriever: LlamaIndexDocumentRetriever
    catalog: PassageCatalog
    gateway: EnforcedToolGateway
    ledger: ResearchLedger


def build_deep_research_components(
    *,
    model: Any,
    documents: Sequence[IndexableDocument],
    embedding_model: Any | None = None,
    additional_handlers: Mapping[str, ToolHandler] | None = None,
    additional_tool_descriptions: Mapping[str, str] | None = None,
    max_agent_steps: int = 12,
    planning_interval: int = 4,
    max_replans: int = 2,
    max_tool_calls: int = 16,
    tool_timeout_seconds: float = 30,
    require_independent_retrieval: bool = False,
    telemetry: Telemetry | None = None,
) -> DeepResearchComponents:
    """Wire LlamaIndex -> enforced tools -> smolagents -> evidence-gated solver."""

    trace = telemetry or NoOpTelemetry()
    catalog = PassageCatalog()
    retriever = LlamaIndexDocumentRetriever(
        documents,
        embedding_model=embedding_model,
        telemetry=trace,
    )
    handlers = dict(additional_handlers or {})
    descriptions = dict(additional_tool_descriptions or {})
    if set(handlers) != set(descriptions):
        raise ValueError("each additional research tool needs one description")
    if DOCUMENT_RETRIEVAL_TOOL in handlers:
        raise ValueError("the guarded-document tool name is reserved")
    handlers[DOCUMENT_RETRIEVAL_TOOL] = LlamaIndexRetrievalToolHandler(
        retriever,
        catalog,
    )
    descriptions[DOCUMENT_RETRIEVAL_TOOL] = (
        "Retrieve safety-approved passages from the indexed long documents. "
        "payload_json fields: query (string) and top_k (integer, 1-10). "
        "Returns passage IDs, exact excerpts, source URLs, pages, and retrieval scores."
    )
    gateway = EnforcedToolGateway(
        policy=RouteToolPolicy(
            route=SolverRoute.DEEP_RESEARCH_FALLBACK,
            tools=ToolPolicy(allowed_tools=tuple(sorted(handlers))),
            budget=ToolBudget(
                max_calls=max_tool_calls,
                timeout_seconds=tool_timeout_seconds,
            ),
        ),
        handlers=handlers,
        telemetry=trace,
    )
    backend = SmolagentsResearchBackend(
        model=model,
        gateway=gateway,
        tool_descriptions=descriptions,
        max_steps=max_agent_steps,
        planning_interval=planning_interval,
        telemetry=trace,
    )
    ledger = ResearchLedger()
    solver = DeepResearchSolver(
        backend=backend,
        passage_catalog=catalog,
        ledger=ledger,
        max_replans=max_replans,
        require_independent_retrieval=require_independent_retrieval,
        telemetry=trace,
    )
    return DeepResearchComponents(
        solver=solver,
        backend=backend,
        retriever=retriever,
        catalog=catalog,
        gateway=gateway,
        ledger=ledger,
    )


__all__ = [
    "DOCUMENT_RETRIEVAL_TOOL",
    "DeepResearchComponents",
    "build_deep_research_components",
]
