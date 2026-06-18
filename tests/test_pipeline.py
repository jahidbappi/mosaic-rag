from __future__ import annotations

from mosaic.eval.harness import BenchmarkHarness
from mosaic.pipeline import MosaicPipeline
from mosaic.types import Modality


def test_pipeline_indexes_documents() -> None:
    pipeline = MosaicPipeline()
    from mosaic.eval.datasets import load_sample_corpus

    count = pipeline.index(load_sample_corpus())
    assert count > 0


def test_retrieval_returns_ranked_chunks() -> None:
    pipeline = MosaicPipeline()
    from mosaic.eval.datasets import load_sample_corpus
    from mosaic.types import Query

    pipeline.index(load_sample_corpus())
    result = pipeline.retrieve(Query(text="hybrid retrieval", top_k=3))
    assert len(result.chunks) <= 3
    assert result.chunks[0].rank == 1


def test_answer_includes_citations() -> None:
    pipeline = MosaicPipeline()
    from mosaic.eval.datasets import load_sample_corpus
    from mosaic.types import Query

    pipeline.index(load_sample_corpus())
    answer = pipeline.answer(Query(text="recall at k", top_k=3))
    assert answer.text
    assert len(answer.citations) > 0


def test_benchmark_harness_runs() -> None:
    pipeline = MosaicPipeline()
    metrics = BenchmarkHarness(pipeline).run()
    assert 0.0 <= metrics.recall_at_5 <= 1.0
    assert metrics.p50_latency_ms >= 0


def test_modality_enum() -> None:
    assert Modality.TEXT.value == "text"
