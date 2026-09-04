"""Guarded hybrid passage retrieval implemented on LlamaIndex primitives."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any

from pydantic import JsonValue

from gaia_max.artifacts import ArtifactStore
from gaia_max.observability import NoOpTelemetry, Telemetry
from gaia_max.research.models import IndexableDocument, RetrievedPassage
from gaia_max.retrieval.source_safety import (
    GuardedSourceDocument,
    SourceDisposition,
    UnsafeSourceBlockedError,
)


class OptionalResearchDependencyError(RuntimeError):
    """An explicitly requested optional research backend is not installed."""


class PassageCatalog:
    """Run-local authority: agents may cite only passages actually returned by tools."""

    def __init__(self) -> None:
        self._passages: dict[str, RetrievedPassage] = {}
        self._lock = threading.Lock()

    def add(self, passages: Sequence[RetrievedPassage]) -> None:
        with self._lock:
            for passage in passages:
                prior = self._passages.get(passage.passage_id)
                if prior is not None and prior != passage:
                    raise ValueError("passage identity collision")
                self._passages[passage.passage_id] = passage

    def resolve(self, passage_ids: Sequence[str]) -> tuple[RetrievedPassage, ...]:
        with self._lock:
            missing = [item for item in passage_ids if item not in self._passages]
            if missing:
                raise KeyError("one or more citations were not returned by an approved tool")
            return tuple(self._passages[item] for item in passage_ids)


def indexable_document_from_guarded(
    source: GuardedSourceDocument,
    store: ArtifactStore,
    *,
    document_id: str | None = None,
    page_number: int | None = None,
    is_primary_source: bool = False,
) -> IndexableDocument:
    """Read only the model-safe artifact selected by ``SourceBoundary``."""

    if source.safety.disposition is SourceDisposition.BLOCKED:
        raise UnsafeSourceBlockedError("blocked source cannot enter a document index")
    artifact = store.get(source.safety.model_text_artifact_id)
    try:
        text = store.read_bytes(artifact).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("guarded model-text artifact is not UTF-8") from exc
    return IndexableDocument(
        document_id=document_id or source.safety.model_text_artifact_id,
        text=text,
        model_text_artifact_id=source.safety.model_text_artifact_id,
        source_url=source.document.final_url,
        page_number=page_number,
        source_date=(
            source.document.published_at.isoformat()
            if source.document.published_at is not None
            else None
        ),
        retrieved_at=source.document.retrieved_at,
        is_primary_source=is_primary_source,
        safety_disposition=source.safety.disposition.value,
    )


class LlamaIndexDocumentRetriever:
    """BM25 plus optional dense retrieval over safety-approved document text."""

    def __init__(
        self,
        documents: Sequence[IndexableDocument],
        *,
        embedding_model: Any | None = None,
        chunk_size: int = 768,
        chunk_overlap: int = 96,
        candidate_pool_size: int = 20,
        lexical_weight: float = 0.55,
        telemetry: Telemetry | None = None,
    ) -> None:
        if not documents:
            raise ValueError("at least one guarded document is required")
        if chunk_size < 128 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("invalid document chunking configuration")
        if not 1 <= candidate_pool_size <= 100:
            raise ValueError("candidate pool size must be between 1 and 100")
        if not 0 <= lexical_weight <= 1:
            raise ValueError("lexical weight must be between zero and one")
        if len({item.document_id for item in documents}) != len(documents):
            raise ValueError("indexable document IDs must be unique")

        self._telemetry = telemetry or NoOpTelemetry()
        self._candidate_pool_size = candidate_pool_size
        self._lexical_weight = lexical_weight
        self._has_dense = embedding_model is not None
        self._nodes_by_id: dict[str, Any] = {}
        self._lexical: Any
        self._dense: Any | None = None
        self._build(documents, embedding_model, chunk_size, chunk_overlap)

    def _build(
        self,
        documents: Sequence[IndexableDocument],
        embedding_model: Any | None,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        try:
            core = import_module("llama_index.core")
            IngestionPipeline = import_module(
                "llama_index.core.ingestion"
            ).IngestionPipeline
            SentenceSplitter = import_module(
                "llama_index.core.node_parser"
            ).SentenceSplitter
            BM25Retriever = import_module(
                "llama_index.retrievers.bm25"
            ).BM25Retriever
        except ImportError as exc:
            raise OptionalResearchDependencyError(
                "install gaia-max[research] to enable LlamaIndex retrieval"
            ) from exc

        metadata_keys = (
            "document_id",
            "model_text_artifact_id",
            "source_url",
            "page_number",
            "source_date",
            "retrieved_at",
            "is_primary_source",
        )
        llama_documents = []
        for source in documents:
            metadata = {
                "document_id": source.document_id,
                "model_text_artifact_id": source.model_text_artifact_id,
                "source_url": str(source.source_url) if source.source_url else "",
                "page_number": source.page_number,
                "source_date": source.source_date or "",
                "retrieved_at": source.retrieved_at.isoformat(),
                "is_primary_source": source.is_primary_source,
            }
            llama_documents.append(
                core.Document(
                    id_=source.document_id,
                    text=source.text,
                    metadata=metadata,
                    excluded_embed_metadata_keys=list(metadata_keys),
                    excluded_llm_metadata_keys=list(metadata_keys),
                )
            )
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    include_metadata=True,
                    include_prev_next_rel=True,
                )
            ],
            disable_cache=True,
        )
        with self._telemetry.span(
            "research.index.build",
            {
                "document.count": len(documents),
                "retrieval.dense_enabled": self._has_dense,
            },
        ) as span:
            nodes = pipeline.run(documents=llama_documents, show_progress=False)
            if not nodes:
                raise ValueError("guarded documents produced no retrievable nodes")
            self._nodes_by_id = {node.node_id: node for node in nodes}
            self._lexical = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=min(self._candidate_pool_size, len(nodes)),
            )
            if embedding_model is not None:
                index = core.VectorStoreIndex(
                    nodes=nodes,
                    embed_model=embedding_model,
                    show_progress=False,
                )
                self._dense = index.as_retriever(
                    similarity_top_k=min(self._candidate_pool_size, len(nodes))
                )
            span.set_attribute("node.count", len(nodes))

    async def retrieve(self, query: str, *, top_k: int = 6) -> tuple[RetrievedPassage, ...]:
        if query != query.strip() or not query or len(query) > 4000:
            raise ValueError("retrieval query must be non-empty, trimmed, and bounded")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        with self._telemetry.span(
            "research.retrieve",
            {
                "retrieval.top_k": top_k,
                "retrieval.dense_enabled": self._has_dense,
            },
        ) as span:
            passages = await asyncio.to_thread(self._retrieve_sync, query, top_k)
            span.set_attribute("retrieval.result_count", len(passages))
            return passages

    def _retrieve_sync(self, query: str, top_k: int) -> tuple[RetrievedPassage, ...]:
        QueryBundle = import_module("llama_index.core").QueryBundle

        lexical_hits = self._lexical.retrieve(QueryBundle(query_str=query))
        dense_hits = (
            self._dense.retrieve(QueryBundle(query_str=query))
            if self._dense is not None
            else []
        )
        fused: dict[str, float] = {}
        scores: dict[str, dict[str, float | None]] = {}
        for label, weight, hits in (
            ("lexical", self._lexical_weight, lexical_hits),
            ("vector", 1 - self._lexical_weight, dense_hits),
        ):
            if weight == 0:
                continue
            for position, hit in enumerate(hits, start=1):
                node_id = hit.node.node_id
                fused[node_id] = fused.get(node_id, 0) + weight / (60 + position)
                scores.setdefault(node_id, {"lexical": None, "vector": None})[label] = (
                    float(hit.score) if hit.score is not None else None
                )
        ranked = sorted(fused, key=lambda item: (-fused[item], item))[:top_k]
        return tuple(
            self._passage(
                self._nodes_by_id[node_id],
                rank=rank,
                fused_score=fused[node_id],
                lexical_score=scores[node_id]["lexical"],
                vector_score=scores[node_id]["vector"],
            )
            for rank, node_id in enumerate(ranked, start=1)
        )

    @staticmethod
    def _passage(
        node: Any,
        *,
        rank: int,
        fused_score: float,
        lexical_score: float | None,
        vector_score: float | None,
    ) -> RetrievedPassage:
        metadata = node.metadata
        excerpt = node.get_content(metadata_mode="none").strip()
        identity = "\x00".join(
            (metadata["model_text_artifact_id"], metadata["document_id"], excerpt)
        )
        passage_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        page_number = metadata.get("page_number")
        locator = f"page {page_number}, LlamaIndex node {node.node_id}" if page_number else (
            f"LlamaIndex node {node.node_id}"
        )
        return RetrievedPassage(
            passage_id=passage_id,
            node_id=node.node_id,
            document_id=metadata["document_id"],
            model_text_artifact_id=metadata["model_text_artifact_id"],
            source_url=metadata.get("source_url") or None,
            page_number=page_number,
            source_date=metadata.get("source_date") or None,
            retrieved_at=metadata["retrieved_at"],
            is_primary_source=metadata["is_primary_source"],
            excerpt=excerpt,
            locator=locator,
            lexical_score=lexical_score,
            vector_score=vector_score,
            fused_score=fused_score,
            rank=rank,
        )


class LlamaIndexRetrievalToolHandler:
    """Validated JSON tool handler that records exactly what an agent was shown."""

    def __init__(
        self,
        retriever: LlamaIndexDocumentRetriever,
        catalog: PassageCatalog,
        *,
        max_top_k: int = 10,
    ) -> None:
        self._retriever = retriever
        self._catalog = catalog
        self._max_top_k = max_top_k

    async def __call__(self, payload: Mapping[str, JsonValue]) -> JsonValue:
        if set(payload) - {"query", "top_k"}:
            raise ValueError("retrieval tool received unexpected fields")
        query = payload.get("query")
        top_k = payload.get("top_k", 6)
        if not isinstance(query, str) or isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("retrieval tool requires a string query and integer top_k")
        if not 1 <= top_k <= self._max_top_k:
            raise ValueError("retrieval tool top_k exceeds its code-owned limit")
        passages = await self._retriever.retrieve(query, top_k=top_k)
        self._catalog.add(passages)
        return {
            "passages": [passage.model_dump(mode="json") for passage in passages],
        }


__all__ = [
    "LlamaIndexDocumentRetriever",
    "LlamaIndexRetrievalToolHandler",
    "OptionalResearchDependencyError",
    "PassageCatalog",
    "indexable_document_from_guarded",
]
