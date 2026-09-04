from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

from gaia_max.research.llamaindex_retrieval import (
    LlamaIndexDocumentRetriever,
    LlamaIndexRetrievalToolHandler,
    PassageCatalog,
)
from gaia_max.research.models import IndexableDocument


class MeaningfulTestEmbedding(BaseEmbedding):
    """Tiny semantic feature model used to exercise LlamaIndex's dense path offline."""

    def _vector(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [
            float(lowered.count("chlorophyll") + lowered.count("photosynthesis")),
            float(lowered.count("currency") + lowered.count("banknote")),
            float(lowered.count("planet") + lowered.count("orbit")),
        ]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)


def document(document_id: str, text: str, marker: str) -> IndexableDocument:
    return IndexableDocument(
        document_id=document_id,
        text=text,
        model_text_artifact_id=hashlib.sha256(marker.encode()).hexdigest(),
        source_url=f"https://{document_id}.example/source",
        page_number=2,
        retrieved_at=datetime.now(UTC),
        is_primary_source=True,
        safety_disposition="sanitized",
    )


@pytest.mark.asyncio
async def test_real_hybrid_retrieval_preserves_provenance_and_catalog_authority() -> None:
    retriever = LlamaIndexDocumentRetriever(
        [
            document(
                "biology",
                "Chlorophyll captures light energy during photosynthesis in green plants. " * 8,
                "biology",
            ),
            document(
                "finance",
                "A central bank issues currency and designs each banknote denomination. " * 8,
                "finance",
            ),
        ],
        embedding_model=MeaningfulTestEmbedding(model_name="offline-feature-model"),
        chunk_size=128,
        chunk_overlap=16,
    )
    catalog = PassageCatalog()
    handler = LlamaIndexRetrievalToolHandler(retriever, catalog)

    output = await handler({"query": "How does chlorophyll support photosynthesis?", "top_k": 2})

    passages = output["passages"]
    assert passages[0]["document_id"] == "biology"
    assert passages[0]["vector_score"] is not None
    assert passages[0]["page_number"] == 2
    assert catalog.resolve([passages[0]["passage_id"]])[0].excerpt
    with pytest.raises(KeyError, match="citations"):
        catalog.resolve(["f" * 64])


@pytest.mark.asyncio
async def test_bm25_mode_is_real_and_does_not_require_dense_dependencies() -> None:
    retriever = LlamaIndexDocumentRetriever(
        [document("lexical", "The rareword zephyrlattice identifies this passage. " * 6, "x")],
        chunk_size=128,
        chunk_overlap=16,
    )
    hits = await retriever.retrieve("zephyrlattice", top_k=1)

    assert hits[0].document_id == "lexical"
    assert hits[0].lexical_score is not None
    assert hits[0].vector_score is None
