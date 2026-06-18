from __future__ import annotations

import argparse
from pathlib import Path

from mosaic.eval.ablations import run_ablations
from mosaic.eval.embedders import resolve_embedder
from mosaic.eval.harness import BenchmarkHarness
from mosaic.eval.loaders.registry import load_eval_bundle
from mosaic.pipeline import MosaicPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mosaic benchmark ablations")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory for benchmark artifacts",
    )
    parser.add_argument("--quick", action="store_true", help="Run single default benchmark")
    parser.add_argument(
        "--dataset",
        choices=["builtin", "scifact", "docvqa"],
        default="scifact",
        help="Evaluation dataset (default: scifact)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap query count (default: 50 for scifact/docvqa, all for builtin)",
    )
    parser.add_argument(
        "--embedder",
        choices=["mock", "minilm", "bge-small", "clip", "ollama"],
        default="mock",
        help="Embedding backend (minilm/bge-small require pip install mosaic-rag[ml]; ollama requires local Ollama)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Dataset cache directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for query subsampling")
    args = parser.parse_args()

    max_samples = args.max_samples
    if max_samples is None and args.dataset != "builtin":
        max_samples = 50

    if args.quick:
        bundle = load_eval_bundle(
            args.dataset,
            max_samples=max_samples,
            cache_dir=args.cache_dir,
            seed=args.seed,
        )
        pipeline = MosaicPipeline(embedder=resolve_embedder(args.embedder))
        metrics = BenchmarkHarness(pipeline, bundle=bundle).run()
        print(metrics.model_dump_json(indent=2))
        return

    results = run_ablations(
        args.output,
        dataset=args.dataset,
        max_samples=max_samples,
        embedder=args.embedder,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    print(f"Completed {len(results)} ablation runs → {args.output}")
    best = max(results, key=lambda r: r.metrics.mrr)
    print(f"Best config: {best.name} (MRR={best.metrics.mrr:.3f}) on {best.dataset}")


if __name__ == "__main__":
    main()
