from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class DocumentChunk(BaseModel):
    id: str
    document_id: str
    content: str
    modality: Modality
    image_path: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Document(BaseModel):
    id: str
    title: str
    chunks: list[DocumentChunk] = Field(default_factory=list)


class Query(BaseModel):
    text: str
    image_path: str | None = None
    top_k: int = 5


class ScoredChunk(BaseModel):
    chunk: DocumentChunk
    score: float
    rank: int


class RetrievalResult(BaseModel):
    query: Query
    chunks: list[ScoredChunk]
    latency_ms: float
    strategy: str


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    excerpt: str
    score: float


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    faithfulness: float | None = None
    correctness: float | None = None


class BenchmarkSample(BaseModel):
    id: str
    query: str
    expected_answer: str
    relevant_chunk_ids: list[str]
    image_path: str | None = None


class BenchmarkMetrics(BaseModel):
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    faithfulness: float
    correctness: float
    p50_latency_ms: float
    p95_latency_ms: float
    cost_per_query_usd: float


StrategyName = Literal["dense", "hybrid", "bm25"]
