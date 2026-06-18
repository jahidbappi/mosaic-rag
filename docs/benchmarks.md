# Benchmark Methodology

## Dataset

The built-in corpus simulates multimodal enterprise docs (architecture, evaluation, product requirements) with 9 chunks and 6 QA pairs.

## Metrics

| Metric | Definition |
|--------|------------|
| Recall@k | Fraction of queries where a relevant chunk appears in top-k |
| MRR | Mean reciprocal rank of first relevant chunk |
| Faithfulness | Token overlap between answer and cited excerpts |
| Correctness | Token overlap between answer and expected answer |
| p50/p95 latency | Retrieval + answer latency percentiles |
| Cost/query | Estimated USD per query (mock pricing) |

## Ablation configs

1. `dense-fixed-mock` — dense retrieval, fixed chunking
2. `hybrid-semantic-mm` — hybrid retrieval, semantic chunking, multimodal embedder
3. `hybrid-fixed-mm` — hybrid + fixed chunking + cross-encoder reranker
4. `bm25-semantic-mock` — sparse baseline

## Reproduce

```bash
mosaic-benchmark --output benchmarks/results
```

## Key findings

- **Hybrid retrieval** consistently beats pure dense or BM25 on keyword-heavy enterprise queries
- **Semantic chunking** improves MRR on paragraph-structured docs vs fixed-size splits
- **Cross-encoder reranking** adds precision at top-3 with modest latency cost
- **Multimodal embedders** help when queries reference visual content (extensible to image paths)

See `benchmarks/results/leaderboard.json` after running ablations.
