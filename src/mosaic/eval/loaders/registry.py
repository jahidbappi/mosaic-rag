from __future__ import annotations

from pathlib import Path

from mosaic.eval.datasets import load_benchmark_samples, load_sample_corpus
from mosaic.eval.loaders.beir_scifact import load_beir_scifact
from mosaic.eval.loaders.docvqa import load_docvqa
from mosaic.eval.loaders.types import EvalBundle

BUILTIN_CITATION = "Mosaic built-in smoke-test corpus (synthetic, for unit tests only)."


def load_builtin_bundle(*, seed: int = 42) -> EvalBundle:
    return EvalBundle(
        name="builtin",
        documents=load_sample_corpus(),
        samples=load_benchmark_samples(),
        citation=BUILTIN_CITATION,
        license="MIT (synthetic)",
        source_url="https://github.com/jahidbappi/mosaic-rag",
        num_corpus=len(load_sample_corpus()),
        num_queries=len(load_benchmark_samples()),
        seed=seed,
        chunk_during_index=True,
        multimodal=False,
        notes="Synthetic 9-chunk corpus for fast CI smoke tests.",
    )


def load_eval_bundle(
    name: str,
    *,
    max_samples: int | None = None,
    cache_dir: Path | None = None,
    seed: int = 42,
) -> EvalBundle:
    normalized = name.lower().strip()
    if normalized == "builtin":
        return load_builtin_bundle(seed=seed)
    if normalized == "scifact":
        return load_beir_scifact(max_samples=max_samples, cache_dir=cache_dir, seed=seed)
    if normalized == "docvqa":
        return load_docvqa(max_samples=max_samples, cache_dir=cache_dir, seed=seed)
    raise ValueError(f"Unknown dataset '{name}'. Choose from: builtin, scifact, docvqa")
