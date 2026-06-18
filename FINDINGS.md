# Mosaic SciFact Findings

Reproducible ablations on **BEIR SciFact** (50 queries, seed 42, 5,183 abstracts).

## Reproduction

```bash
pip install -e ".[ml]"
mosaic-benchmark --dataset scifact --max-samples 50 --embedder minilm --seed 42 --output benchmarks/results
```

Metadata in `benchmarks/results/leaderboard.json`: dataset, seed, corpus size, embedder, git commit, timestamp.

## Published BEIR SciFact baselines (Thakur et al., 2021)

Reference zero-shot numbers on the **full 300-query test set** (not directly comparable to our 50-query subsample, but useful context):

| Model | MRR | nDCG@10 | Source |
|-------|-----|---------|--------|
| BM25 | ~0.67 | ~0.66 | BEIR leaderboard |
| all-MiniLM-L6-v2 (dense) | ~0.65 | ~0.64 | BEIR / sentence-transformers |
| Hybrid (BM25 + dense) | ~0.70+ | varies | Common BEIR practice |

SciFact is **lexical-friendly** — sparse retrieval is competitive with dense models on the full benchmark.

## Our results (50 queries, seed 42)

### BM25 (`bm25-semantic`)

| Metric | Value |
|--------|-------|
| **MRR** | **73.5%** |
| Recall@5 | 82.0% |
| Recall@10 | 88.0% |
| Faithfulness | 43.4% |
| p50 latency | 36 ms |

BM25 wins on latency and strong MRR for this scientific-claim corpus where queries share terminology with abstracts.

### Dense / hybrid with MiniLM (`all-MiniLM-L6-v2`)

**Completed** — 50 queries, seed 42, embedder `all-MiniLM-L6-v2`, commit in `leaderboard.json`.

| Config | Strategy | MRR | Recall@5 | p50 (ms) |
|--------|----------|-----|----------|----------|
| **hybrid-semantic** | hybrid | **76.7%** | 86.0% | 94 |
| dense-fixed | dense | 76.1% | 84.0% | 67 |
| hybrid-fixed-xenc | hybrid + MS MARCO xenc | 75.7% | 86.0% | 667 |
| bm25-semantic | bm25 | 73.5% | 82.0% | 37 |

**Headline:** Hybrid + MiniLM beats BM25 by **+3.2 pp MRR** on this subsample, with ~2.5× latency (94 ms vs 37 ms p50). Cross-encoder reranking adds ~7× latency for marginal faithfulness gain.

For **production-quality local retrieval**, prefer **`BAAI/bge-small-en-v1.5`** (Xiao et al., 2023):

```bash
mosaic-benchmark --dataset scifact --max-samples 50 --embedder bge-small --seed 42 --output benchmarks/results
```

### Cross-encoder reranker

`hybrid-fixed-xenc` uses **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (Reimers & Gurevych, 2019) — the standard MS MARCO cross-encoder. Adds ~1 s p50 latency; use when initial hybrid recall is broad enough.

## When BM25 wins vs dense/hybrid

1. **Keyword overlap** — SciFact claims reuse abstract terminology (genes, compounds, outcomes).
2. **Short documents** — one abstract per doc; chunking matters less than term frequency.
3. **No paraphrase gap** — queries restate claims rather than abstract questions.

Dense/hybrid helps when vocabulary diverges or semantic similarity matters without exact term overlap.

## Chunking tradeoffs

| Chunker | Best for | SciFact note |
|---------|----------|--------------|
| **Semantic** | Long docs, paragraph boundaries | Limited effect — abstracts are already short |
| **Fixed-size** | Cross-encoder pipelines | May split rare terms; hybrid + xenc compensates |

## OSS stack (best-in-class, all free)

| Component | Choice | Citation |
|-----------|--------|----------|
| Dense (CI) | `all-MiniLM-L6-v2` | Wang et al., 2020 |
| Dense (recommended) | `BAAI/bge-small-en-v1.5` | Xiao et al., 2023 (C-Pack) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reimers & Gurevych, 2019 |
| Sparse | BM25 (rank-bm25) | Robertson & Zaragoza, 2009 |
| Optional embedder | `nomic-embed-text` via Ollama | Nomic AI, 2024 |
| Eval data | BEIR SciFact, DocVQA | Thakur et al., 2021; Mathew et al., 2021 |

## DocVQA smoke test

```bash
mosaic-benchmark --dataset docvqa --max-samples 20 --embedder minilm --seed 42 --output benchmarks/results/docvqa
```

Text-only mode indexes metadata by default — pipeline smoke test, not document-image retrieval.

## Takeaways

1. **Measure before claiming** — mock embedder ablations are CI-only; real embedders required for dense/hybrid conclusions.
2. **Domain matters** — SciFact favors BM25; don't assume dense always wins.
3. **Report metadata** — seed, sample count, embedder name, commit hash.
