from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from mosaic.eval.datasets import load_benchmark_samples, load_sample_corpus
from mosaic.pipeline import MosaicPipeline
from mosaic.types import BenchmarkMetrics, BenchmarkSample, Query


def token_overlap(a: str, b: str) -> float:
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top = set(retrieved_ids[:k])
    return 1.0 if any(rid in top for rid in relevant_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


@dataclass
class AblationResult:
    name: str
    strategy: str
    chunker: str
    embedder: str
    reranker: str
    metrics: BenchmarkMetrics


def content_relevant(chunk_content: str, sample: BenchmarkSample) -> bool:
    return token_overlap(chunk_content, sample.expected_answer) >= 0.2


def recall_at_k_content(retrieved_contents: list[str], sample: BenchmarkSample, k: int) -> float:
    top = retrieved_contents[:k]
    return 1.0 if any(content_relevant(c, sample) for c in top) else 0.0


def reciprocal_rank_content(retrieved_contents: list[str], sample: BenchmarkSample) -> float:
    for i, content in enumerate(retrieved_contents, start=1):
        if content_relevant(content, sample):
            return 1.0 / i
    return 0.0


class BenchmarkHarness:
    def __init__(self, pipeline: MosaicPipeline, samples: list[BenchmarkSample] | None = None) -> None:
        self.pipeline = pipeline
        self.samples = samples or load_benchmark_samples()

    def run(self, *, chunk: bool = False) -> BenchmarkMetrics:
        self.pipeline.index(load_sample_corpus(), chunk=chunk)

        recalls_1: list[float] = []
        recalls_5: list[float] = []
        recalls_10: list[float] = []
        mrr_scores: list[float] = []
        faithfulness_scores: list[float] = []
        correctness_scores: list[float] = []
        latencies: list[float] = []

        for sample in self.samples:
            query = Query(text=sample.query, top_k=10)
            start = time.perf_counter()
            retrieval = self.pipeline.retrieve(query)
            answer = self.pipeline.answer(query)
            latencies.append((time.perf_counter() - start) * 1000)

            retrieved_ids = [c.chunk.id for c in retrieval.chunks]
            retrieved_contents = [c.chunk.content for c in retrieval.chunks]

            if chunk:
                recalls_1.append(recall_at_k_content(retrieved_contents, sample, 1))
                recalls_5.append(recall_at_k_content(retrieved_contents, sample, 5))
                recalls_10.append(recall_at_k_content(retrieved_contents, sample, 10))
                mrr_scores.append(reciprocal_rank_content(retrieved_contents, sample))
            else:
                recalls_1.append(recall_at_k(retrieved_ids, sample.relevant_chunk_ids, 1))
                recalls_5.append(recall_at_k(retrieved_ids, sample.relevant_chunk_ids, 5))
                recalls_10.append(recall_at_k(retrieved_ids, sample.relevant_chunk_ids, 10))
                mrr_scores.append(reciprocal_rank(retrieved_ids, sample.relevant_chunk_ids))

            cited_text = " ".join(c.excerpt for c in answer.citations)
            faithfulness_scores.append(token_overlap(answer.text, cited_text))
            correctness_scores.append(token_overlap(answer.text, sample.expected_answer))

        sorted_lat = sorted(latencies)
        return BenchmarkMetrics(
            recall_at_1=statistics.mean(recalls_1),
            recall_at_5=statistics.mean(recalls_5),
            recall_at_10=statistics.mean(recalls_10),
            mrr=statistics.mean(mrr_scores),
            faithfulness=statistics.mean(faithfulness_scores),
            correctness=statistics.mean(correctness_scores),
            p50_latency_ms=sorted_lat[len(sorted_lat) // 2],
            p95_latency_ms=sorted_lat[max(int(len(sorted_lat) * 0.95) - 1, 0)],
            cost_per_query_usd=0.0002,
        )
