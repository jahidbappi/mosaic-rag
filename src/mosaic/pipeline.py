from __future__ import annotations

import time
from typing import Literal

from mosaic.chunking.chunkers import BaseChunker, FixedSizeChunker, SemanticChunker
from mosaic.embeddings.base import BaseEmbedder
from mosaic.embeddings.providers import MockMultimodalEmbedder, MockTextEmbedder
from mosaic.rerank.rerankers import BaseReranker, CrossEncoderReranker, ScoreFusionReranker
from mosaic.retrieval.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from mosaic.types import Answer, Citation, Document, Query, RetrievalResult


class MosaicPipeline:
    """End-to-end multimodal RAG pipeline."""

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        retriever_strategy: Literal["dense", "hybrid", "bm25"] = "hybrid",
        reranker: BaseReranker | None = None,
        chunker: BaseChunker | None = None,
    ) -> None:
        self.embedder = embedder or MockMultimodalEmbedder()
        self.chunker = chunker or SemanticChunker()
        self.reranker = reranker or ScoreFusionReranker()
        self.retriever_strategy = retriever_strategy
        self._chunks: list = []

        dense = DenseRetriever(self.embedder)
        sparse = BM25Retriever()
        self._retrievers = {
            "dense": dense,
            "bm25": sparse,
            "hybrid": HybridRetriever(dense, sparse),
        }

    def index(self, documents: list[Document], *, chunk: bool = True) -> int:
        if chunk:
            all_chunks = []
            for doc in documents:
                all_chunks.extend(self.chunker.chunk(doc))
        else:
            all_chunks = [c for doc in documents for c in doc.chunks]
        self._chunks = all_chunks

        for retriever in self._retrievers.values():
            if hasattr(retriever, "index"):
                retriever.index(all_chunks)
        return len(all_chunks)

    def retrieve(self, query: Query) -> RetrievalResult:
        retriever = self._retrievers[self.retriever_strategy]
        result = retriever.retrieve(query, self._chunks, query.top_k)
        reranked = self.reranker.rerank(query, result.chunks, query.top_k)
        result.chunks = reranked
        return result

    def answer(self, query: Query) -> Answer:
        start = time.perf_counter()
        retrieval = self.retrieve(query)

        if not retrieval.chunks:
            return Answer(text="No relevant context found.", citations=[])

        context = "\n\n".join(c.chunk.content for c in retrieval.chunks[:3])
        citations = [
            Citation(
                chunk_id=c.chunk.id,
                document_id=c.chunk.document_id,
                excerpt=c.chunk.content[:180],
                score=c.score,
            )
            for c in retrieval.chunks[:3]
        ]

        answer_text = (
            f"Based on retrieved context, here is a grounded response to '{query.text}': "
            f"{context[:400]}{'...' if len(context) > 400 else ''}"
        )

        _ = start  # reserved for latency tracking in eval harness
        return Answer(text=answer_text, citations=citations)


def create_default_pipeline(strategy: str = "hybrid") -> MosaicPipeline:
    return MosaicPipeline(
        embedder=MockTextEmbedder(),
        retriever_strategy=strategy,  # type: ignore[arg-type]
        reranker=CrossEncoderReranker(),
        chunker=FixedSizeChunker(),
    )
