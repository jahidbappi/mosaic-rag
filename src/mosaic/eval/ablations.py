from __future__ import annotations

import json
from pathlib import Path

import subprocess

from mosaic.chunking.chunkers import FixedSizeChunker, SemanticChunker
from mosaic.embeddings.providers import MockMultimodalEmbedder, MockTextEmbedder
from mosaic.eval.embedders import EmbedderName, resolve_embedder
from mosaic.eval.harness import AblationResult, BenchmarkHarness, utc_timestamp
from mosaic.eval.loaders.registry import load_eval_bundle
from mosaic.eval.loaders.types import EvalBundle
from mosaic.pipeline import MosaicPipeline
from mosaic.rerank.rerankers import CrossEncoderReranker, ScoreFusionReranker


def run_ablations(
    output_dir: Path,
    *,
    dataset: str = "scifact",
    max_samples: int | None = 50,
    embedder: EmbedderName = "mock",
    cache_dir: Path | None = None,
    seed: int = 42,
) -> list[AblationResult]:
    bundle = load_eval_bundle(dataset, max_samples=max_samples, cache_dir=cache_dir, seed=seed)
    embedder_impl = resolve_embedder(embedder)
    timestamp = utc_timestamp()

    configs: list[tuple[str, str, object, object, object]] = [
        ("dense-fixed", "dense", FixedSizeChunker(), embedder_impl, ScoreFusionReranker()),
        ("hybrid-semantic", "hybrid", SemanticChunker(), embedder_impl, ScoreFusionReranker()),
        ("hybrid-fixed-xenc", "hybrid", FixedSizeChunker(), embedder_impl, CrossEncoderReranker()),
        ("bm25-semantic", "bm25", SemanticChunker(), MockTextEmbedder(), ScoreFusionReranker()),
    ]
    if bundle.multimodal:
        configs[1] = (
            "hybrid-semantic-mm",
            "hybrid",
            SemanticChunker(),
            MockMultimodalEmbedder(),
            ScoreFusionReranker(),
        )

    results: list[AblationResult] = []
    for name, strategy, chunker, emb, reranker in configs:
        pipeline = MosaicPipeline(
            embedder=emb,
            retriever_strategy=strategy,  # type: ignore[arg-type]
            reranker=reranker,
            chunker=chunker,
        )
        metrics = BenchmarkHarness(pipeline, bundle=bundle).run()
        results.append(
            AblationResult(
                name=name,
                strategy=strategy,
                chunker=chunker.name,
                embedder=emb.name,
                reranker=reranker.name,
                metrics=metrics,
                dataset=bundle.name,
                num_samples=len(bundle.samples),
                timestamp=timestamp,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = [_serialize_result(r, bundle, embedder, timestamp) for r in results]
    (output_dir / "ablation_results.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    leaderboard = sorted(serializable, key=lambda x: x["metrics"]["mrr"], reverse=True)
    commit_sha = _git_commit_sha()
    meta = {
        "dataset": bundle.name,
        "num_samples": len(bundle.samples),
        "num_corpus": bundle.num_corpus,
        "embedder": embedder_impl.name,
        "timestamp": timestamp,
        "citation": bundle.citation,
        "license": bundle.license,
        "source_url": bundle.source_url,
        "seed": seed,
        "multimodal": bundle.multimodal,
        "commit": commit_sha,
        "results": leaderboard,
    }
    (output_dir / "leaderboard.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    web_leaderboard = Path("web/src/data/leaderboard.json")
    if web_leaderboard.parent.exists():
        web_leaderboard.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        public_lb = Path("web/public/leaderboard.json")
        if public_lb.parent.exists():
            public_lb.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return results


def _serialize_result(
    result: AblationResult,
    bundle: EvalBundle,
    embedder: EmbedderName,
    timestamp: str,
) -> dict[str, object]:
    return {
        "name": result.name,
        "strategy": result.strategy,
        "chunker": result.chunker,
        "embedder": result.embedder,
        "reranker": result.reranker,
        "dataset": result.dataset,
        "num_samples": result.num_samples,
        "timestamp": timestamp,
        "metrics": result.metrics.model_dump(),
        "citation": bundle.citation,
        "source_url": bundle.source_url,
        "requested_embedder": embedder,
    }


def _git_commit_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
