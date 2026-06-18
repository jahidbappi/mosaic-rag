from __future__ import annotations

import json
from pathlib import Path

from mosaic.types import BenchmarkSample, Document, DocumentChunk, Modality


def load_sample_corpus() -> list[Document]:
    """Built-in multimodal document corpus for reproducible benchmarks."""
    return [
        Document(
            id="doc-architecture",
            title="System Architecture Overview",
            chunks=[
                DocumentChunk(
                    id="arch-1",
                    document_id="doc-architecture",
                    content="The ingestion service normalizes PDFs and slides into text-image pairs.",
                    modality=Modality.TEXT,
                ),
                DocumentChunk(
                    id="arch-2",
                    document_id="doc-architecture",
                    content="Hybrid retrieval combines dense embeddings with BM25 for keyword grounding.",
                    modality=Modality.TEXT,
                ),
                DocumentChunk(
                    id="arch-3",
                    document_id="doc-architecture",
                    content="A cross-encoder reranker improves precision at the final top-k stage.",
                    modality=Modality.TEXT,
                ),
            ],
        ),
        Document(
            id="doc-eval",
            title="Evaluation Methodology",
            chunks=[
                DocumentChunk(
                    id="eval-1",
                    document_id="doc-eval",
                    content="Recall@k measures whether any relevant chunk appears in the top k results.",
                    modality=Modality.TEXT,
                ),
                DocumentChunk(
                    id="eval-2",
                    document_id="doc-eval",
                    content="Faithfulness scores whether the generated answer stays grounded in citations.",
                    modality=Modality.TEXT,
                ),
                DocumentChunk(
                    id="eval-3",
                    document_id="doc-eval",
                    content="Latency p95 and cost per query are tracked for every ablation run.",
                    modality=Modality.TEXT,
                ),
            ],
        ),
        Document(
            id="doc-product",
            title="Product Requirements",
            chunks=[
                DocumentChunk(
                    id="prod-1",
                    document_id="doc-product",
                    content="Users upload mixed documents containing screenshots, charts, and paragraphs.",
                    modality=Modality.TEXT,
                ),
                DocumentChunk(
                    id="prod-2",
                    document_id="doc-product",
                    content="Answers must include citations pointing to the exact retrieved chunk.",
                    modality=Modality.TEXT,
                ),
                DocumentChunk(
                    id="prod-3",
                    document_id="doc-product",
                    content="Semantic chunking outperformed fixed-size chunking on slide decks in early tests.",
                    modality=Modality.TEXT,
                ),
            ],
        ),
    ]


def load_benchmark_samples() -> list[BenchmarkSample]:
    return [
        BenchmarkSample(
            id="q1",
            query="How does hybrid retrieval work?",
            expected_answer="Hybrid retrieval combines dense embeddings with BM25.",
            relevant_chunk_ids=["arch-2"],
        ),
        BenchmarkSample(
            id="q2",
            query="What is recall at k?",
            expected_answer="Recall@k checks if a relevant chunk is in the top k results.",
            relevant_chunk_ids=["eval-1"],
        ),
        BenchmarkSample(
            id="q3",
            query="Do answers need citations?",
            expected_answer="Yes, answers must include citations to retrieved chunks.",
            relevant_chunk_ids=["prod-2"],
        ),
        BenchmarkSample(
            id="q4",
            query="Which chunking strategy worked better on slide decks?",
            expected_answer="Semantic chunking outperformed fixed-size chunking on slide decks.",
            relevant_chunk_ids=["prod-3"],
        ),
        BenchmarkSample(
            id="q5",
            query="What reranker improves final precision?",
            expected_answer="A cross-encoder reranker improves precision at top-k.",
            relevant_chunk_ids=["arch-3"],
        ),
        BenchmarkSample(
            id="q6",
            query="What metrics track operational cost?",
            expected_answer="Latency p95 and cost per query are tracked.",
            relevant_chunk_ids=["eval-3"],
        ),
    ]


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
