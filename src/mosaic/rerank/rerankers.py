from __future__ import annotations

from abc import ABC, abstractmethod

from mosaic.types import Query, ScoredChunk


class BaseReranker(ABC):
    name: str

    @abstractmethod
    def rerank(self, query: Query, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        raise NotImplementedError


class ScoreFusionReranker(BaseReranker):
    name = "score-fusion"

    def rerank(self, query: Query, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        query_terms = set(query.text.lower().split())
        reranked: list[tuple[ScoredChunk, float]] = []

        for item in chunks:
            overlap = sum(1 for t in query_terms if t in item.chunk.content.lower())
            bonus = overlap * 0.05
            reranked.append((item, item.score + bonus))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return [
            ScoredChunk(chunk=item.chunk, score=score, rank=i + 1)
            for i, (item, score) in enumerate(reranked[:top_k])
        ]


class CrossEncoderReranker(BaseReranker):
    name = "cross-encoder"

    def __init__(self) -> None:
        self._model = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            self._model = None

    def rerank(self, query: Query, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if self._model is None or not chunks:
            return ScoreFusionReranker().rerank(query, chunks, top_k)

        pairs = [(query.text, c.chunk.content) for c in chunks]
        scores = self._model.predict(pairs)
        reranked = sorted(zip(chunks, scores, strict=False), key=lambda x: float(x[1]), reverse=True)
        return [
            ScoredChunk(chunk=item.chunk, score=float(score), rank=i + 1)
            for i, (item, score) in enumerate(reranked[:top_k])
        ]
