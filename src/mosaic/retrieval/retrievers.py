from __future__ import annotations

import math
import re
import time
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

from mosaic.embeddings.base import BaseEmbedder
from mosaic.types import DocumentChunk, Query, RetrievalResult, ScoredChunk


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class BaseRetriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, query: Query, chunks: list[DocumentChunk], top_k: int) -> RetrievalResult:
        raise NotImplementedError


class DenseRetriever(BaseRetriever):
    name = "dense"

    def __init__(self, embedder: BaseEmbedder) -> None:
        self.embedder = embedder
        self._vectors: dict[str, np.ndarray] = {}

    def index(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            self._vectors[chunk.id] = self.embedder.embed_chunk(chunk)

    def retrieve(self, query: Query, chunks: list[DocumentChunk], top_k: int) -> RetrievalResult:
        start = time.perf_counter()
        if not self._vectors:
            self.index(chunks)

        q_vec = (
            self.embedder.embed_image(query.image_path)
            if query.image_path
            else self.embedder.embed_text(query.text)
        )

        scored: list[tuple[DocumentChunk, float]] = []
        chunk_map = {c.id: c for c in chunks}
        for chunk_id, vec in self._vectors.items():
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue
            scored.append((chunk, cosine_similarity(q_vec, vec)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        result_chunks = [
            ScoredChunk(chunk=chunk, score=score, rank=i + 1) for i, (chunk, score) in enumerate(top)
        ]

        return RetrievalResult(
            query=query,
            chunks=result_chunks,
            latency_ms=(time.perf_counter() - start) * 1000,
            strategy=self.name,
        )


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[tuple[DocumentChunk, Counter[str]]] = []
        self._avgdl = 0.0
        self._df: Counter[str] = Counter()
        self._N = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def index(self, chunks: list[DocumentChunk]) -> None:
        self._docs = []
        self._df = Counter()
        lengths: list[int] = []
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            tf = Counter(tokens)
            self._docs.append((chunk, tf))
            lengths.append(len(tokens))
            for term in set(tokens):
                self._df[term] += 1
        self._N = len(self._docs)
        self._avgdl = sum(lengths) / max(len(lengths), 1)

    def _score(self, query_terms: list[str], tf: Counter[str], dl: int) -> float:
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            df = self._df.get(term, 0)
            idf = math.log(1 + (self._N - df + 0.5) / (df + 0.5))
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score += idf * (freq * (self.k1 + 1)) / denom
        return score

    def retrieve(self, query: Query, chunks: list[DocumentChunk], top_k: int) -> RetrievalResult:
        start = time.perf_counter()
        if not self._docs:
            self.index(chunks)

        query_terms = self._tokenize(query.text)
        scored: list[tuple[DocumentChunk, float]] = []
        for chunk, tf in self._docs:
            dl = sum(tf.values())
            scored.append((chunk, self._score(query_terms, tf, dl)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        result_chunks = [
            ScoredChunk(chunk=chunk, score=score, rank=i + 1) for i, (chunk, score) in enumerate(top)
        ]

        return RetrievalResult(
            query=query,
            chunks=result_chunks,
            latency_ms=(time.perf_counter() - start) * 1000,
            strategy=self.name,
        )


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever, alpha: float = 0.65) -> None:
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    def index(self, chunks: list[DocumentChunk]) -> None:
        self.dense.index(chunks)
        self.sparse.index(chunks)

    def retrieve(self, query: Query, chunks: list[DocumentChunk], top_k: int) -> RetrievalResult:
        start = time.perf_counter()
        self.index(chunks)

        dense_result = self.dense.retrieve(query, chunks, top_k=max(top_k * 3, 10))
        sparse_result = self.sparse.retrieve(query, chunks, top_k=max(top_k * 3, 10))

        combined: dict[str, float] = {}
        chunk_map: dict[str, DocumentChunk] = {}

        if dense_result.chunks:
            max_dense = max(c.score for c in dense_result.chunks) or 1.0
            for c in dense_result.chunks:
                combined[c.chunk.id] = combined.get(c.chunk.id, 0.0) + self.alpha * (c.score / max_dense)
                chunk_map[c.chunk.id] = c.chunk

        if sparse_result.chunks:
            max_sparse = max(c.score for c in sparse_result.chunks) or 1.0
            for c in sparse_result.chunks:
                combined[c.chunk.id] = combined.get(c.chunk.id, 0.0) + (1 - self.alpha) * (
                    c.score / max_sparse
                )
                chunk_map[c.chunk.id] = c.chunk

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        result_chunks = [
            ScoredChunk(chunk=chunk_map[cid], score=score, rank=i + 1)
            for i, (cid, score) in enumerate(ranked)
        ]

        return RetrievalResult(
            query=query,
            chunks=result_chunks,
            latency_ms=(time.perf_counter() - start) * 1000,
            strategy=self.name,
        )
