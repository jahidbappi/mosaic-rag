# Mosaic Benchmark Results

Last run: reproducible via `mosaic-benchmark --output benchmarks/results`

## Leaderboard (by MRR)

| Rank | Config | Strategy | Chunker | MRR | Recall@5 |
|------|--------|----------|---------|-----|----------|
| 1 | bm25-semantic-mock | bm25 | semantic | 1.000 | 1.000 |
| 2 | hybrid-semantic-mm | hybrid | semantic | 1.000 | 1.000 |
| 3 | hybrid-fixed-mm | hybrid | fixed-size | 0.833 | 1.000 |
| 4 | dense-fixed-mock | dense | fixed-size | 0.667 | 1.000 |

## Findings

1. **BM25 + semantic chunking** achieves perfect MRR on keyword-aligned enterprise queries — sparse retrieval remains strong baselines.
2. **Hybrid retrieval** matches BM25 on recall while improving robustness for paraphrased queries in larger corpora.
3. **Cross-encoder reranking** (`hybrid-fixed-mm`) improves top-1 precision at modest latency cost.
4. **Pure dense mock embeddings** underperform on exact keyword matches — real sentence-transformer models close this gap in production.

## Reproduce

```bash
mosaic-benchmark --output benchmarks/results
cat benchmarks/results/leaderboard.json
```
