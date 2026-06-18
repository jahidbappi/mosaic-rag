#!/usr/bin/env python3
"""Pre-download Mosaic evaluation datasets with license and citation info."""

from __future__ import annotations

import argparse
from pathlib import Path

from mosaic.eval.loaders.registry import load_eval_bundle

DATASETS = {
    "scifact": {
        "citation": "Thakur et al. (2021). BEIR. NeurIPS Datasets and Benchmarks.",
        "license": "Apache-2.0",
        "url": "https://huggingface.co/datasets/BeIR/scifact",
    },
    "docvqa": {
        "citation": "Mathew et al. (2021). DocVQA. WACV.",
        "license": "CC BY 4.0",
        "url": "https://huggingface.co/datasets/lmms-lab/DocVQA",
    },
    "builtin": {
        "citation": "Mosaic synthetic smoke corpus",
        "license": "MIT",
        "url": "https://github.com/jahidbappi/mosaic-rag",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Mosaic eval datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="scifact",
        help="Dataset to download",
    )
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "mosaic" / "datasets")
    args = parser.parse_args()

    info = DATASETS[args.dataset]
    print(f"Dataset: {args.dataset}")
    print(f"License: {info['license']}")
    print(f"Citation: {info['citation']}")
    print(f"Source: {info['url']}")
    print(f"Cache: {args.cache_dir}")

    bundle = load_eval_bundle(
        args.dataset,
        max_samples=args.max_samples if args.dataset != "builtin" else None,
        cache_dir=args.cache_dir,
    )
    print(f"Loaded {bundle.num_corpus} corpus entries, {bundle.num_queries} queries.")


if __name__ == "__main__":
    main()
