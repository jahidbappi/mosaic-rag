# Benchmark Methodology

## Datasets

| Dataset | Source | Default use |
|---------|--------|-------------|
| **SciFact** (default) | [BEIR/scifact](https://huggingface.co/datasets/BeIR/scifact) | Public leaderboard |
| **DocVQA** | [lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA) | Multimodal subset eval |
| **builtin** | Synthetic smoke corpus | Unit tests only |

See [DATASETS.md](../DATASETS.md) for citations and licenses.

## Metrics

| Metric | Definition |
|--------|------------|
| Recall@k | Fraction of queries where a relevant chunk appears in top-k |
| MRR | Mean reciprocal rank of first relevant chunk |
| Faithfulness | Token overlap between answer and cited excerpts |
| Correctness | Token overlap between answer and expected answer |
| p50/p95 latency | Retrieval + answer latency percentiles |
| Cost/query | Estimated USD per query (mock pricing) |

SciFact uses document-level relevance (chunk=False); the built-in corpus uses semantic chunking ablations.

## Ablation configs

1. `dense-fixed` — dense retrieval, fixed chunking
2. `hybrid-semantic` — hybrid retrieval, semantic chunking
3. `hybrid-fixed-xenc` — hybrid + fixed chunking + cross-encoder reranker
4. `bm25-semantic` — sparse baseline (no torch required)

## Reproduce

```bash
# CI-scale (no torch)
mosaic-benchmark --dataset scifact --max-samples 20 --embedder mock

# Default leaderboard
mosaic-benchmark --output benchmarks/results

# Full SciFact with real embeddings
pip install -e ".[ml]"
mosaic-benchmark --dataset scifact --max-samples 300 --embedder minilm
```

## Key findings (SciFact, 50 queries)

- **BM25** is a strong sparse baseline on scientific abstracts without GPU dependencies
- **Hybrid retrieval** combines dense and sparse signals; gains depend on embedder quality
- **Cross-encoder reranking** adds precision at top-k with modest latency cost
- Use `--embedder minilm` for production-faithful dense/hybrid numbers

See `benchmarks/results/leaderboard.json` after running ablations.
