from __future__ import annotations

from unittest.mock import patch

import pytest

from mosaic.eval.harness import BenchmarkHarness
from mosaic.eval.loaders.registry import load_builtin_bundle
from mosaic.eval.loaders.types import EvalBundle
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


def test_benchmark_harness_runs_builtin() -> None:
    pipeline = MosaicPipeline()
    bundle = load_builtin_bundle()
    metrics = BenchmarkHarness(pipeline, bundle=bundle).run()
    assert 0.0 <= metrics.recall_at_5 <= 1.0
    assert metrics.p50_latency_ms >= 0


def test_modality_enum() -> None:
    assert Modality.TEXT.value == "text"


def test_load_builtin_bundle() -> None:
    bundle = load_builtin_bundle()
    assert bundle.name == "builtin"
    assert len(bundle.documents) == 3
    assert len(bundle.samples) == 6


@patch("mosaic.eval.loaders.beir_scifact.load_dataset")
@patch("mosaic.eval.loaders.beir_scifact._download_qrels")
def test_scifact_loader_mocked(mock_qrels, mock_load_dataset) -> None:
    from mosaic.eval.loaders.beir_scifact import load_beir_scifact

    mock_qrels.return_value = {"1": ["100"], "2": ["200"]}
    mock_load_dataset.side_effect = [
        [{"_id": "100", "title": "T1", "text": "abstract one"}],
        [
            {"_id": "1", "title": "", "text": "query one"},
            {"_id": "2", "title": "", "text": "query two"},
        ],
    ]

    bundle = load_beir_scifact(max_samples=2, cache_dir=None, seed=42)
    assert isinstance(bundle, EvalBundle)
    assert bundle.name == "scifact"
    assert len(bundle.samples) == 2
    assert bundle.samples[0].relevant_chunk_ids == ["100"]


@pytest.mark.slow
def test_scifact_integration() -> None:
    from mosaic.eval.loaders.registry import load_eval_bundle

    bundle = load_eval_bundle("scifact", max_samples=5, seed=42)
    pipeline = MosaicPipeline()
    metrics = BenchmarkHarness(pipeline, bundle=bundle).run()
    assert 0.0 <= metrics.mrr <= 1.0
    assert len(bundle.samples) == 5
