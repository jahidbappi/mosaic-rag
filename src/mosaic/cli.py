from __future__ import annotations

import argparse
from pathlib import Path

from mosaic.eval.ablations import run_ablations
from mosaic.eval.harness import BenchmarkHarness
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
    args = parser.parse_args()

    if args.quick:
        pipeline = MosaicPipeline()
        metrics = BenchmarkHarness(pipeline).run()
        print(metrics.model_dump_json(indent=2))
        return

    results = run_ablations(args.output)
    print(f"Completed {len(results)} ablation runs → {args.output}")
    best = max(results, key=lambda r: r.metrics.mrr)
    print(f"Best config: {best.name} (MRR={best.metrics.mrr:.3f})")


if __name__ == "__main__":
    main()
