from __future__ import annotations

import json
from pathlib import Path

from mosaic.chunking.chunkers import FixedSizeChunker, SemanticChunker
from mosaic.embeddings.providers import MockMultimodalEmbedder, MockTextEmbedder
from mosaic.eval.harness import AblationResult, BenchmarkHarness
from mosaic.pipeline import MosaicPipeline
from mosaic.rerank.rerankers import CrossEncoderReranker, ScoreFusionReranker


def run_ablations(output_dir: Path) -> list[AblationResult]:
    configs = [
        ("dense-fixed-mock", "dense", FixedSizeChunker(), MockTextEmbedder(), ScoreFusionReranker()),
        ("hybrid-semantic-mm", "hybrid", SemanticChunker(), MockMultimodalEmbedder(), ScoreFusionReranker()),
        ("hybrid-fixed-mm", "hybrid", FixedSizeChunker(), MockMultimodalEmbedder(), CrossEncoderReranker()),
        ("bm25-semantic-mock", "bm25", SemanticChunker(), MockTextEmbedder(), ScoreFusionReranker()),
    ]

    results: list[AblationResult] = []
    for name, strategy, chunker, embedder, reranker in configs:
        pipeline = MosaicPipeline(
            embedder=embedder,
            retriever_strategy=strategy,  # type: ignore[arg-type]
            reranker=reranker,
            chunker=chunker,
        )
        metrics = BenchmarkHarness(pipeline).run(chunk=True)
        result = AblationResult(
            name=name,
            strategy=strategy,
            chunker=chunker.name,
            embedder=embedder.name,
            reranker=reranker.name,
            metrics=metrics,
        )
        results.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "name": r.name,
            "strategy": r.strategy,
            "chunker": r.chunker,
            "embedder": r.embedder,
            "reranker": r.reranker,
            "metrics": r.metrics.model_dump(),
        }
        for r in results
    ]
    (output_dir / "ablation_results.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    leaderboard = sorted(serializable, key=lambda x: x["metrics"]["mrr"], reverse=True)
    (output_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")

    return results
