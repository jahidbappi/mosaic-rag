# API Reference

## `MosaicPipeline`

Main entry point.

```python
pipeline = MosaicPipeline(
    embedder=MockMultimodalEmbedder(),
    retriever_strategy="hybrid",  # dense | bm25 | hybrid
    reranker=ScoreFusionReranker(),
    chunker=SemanticChunker(),
)
pipeline.index(documents)
pipeline.retrieve(Query(text="...", top_k=5))
pipeline.answer(Query(text="...", top_k=5))
```

## Types

- `Document`, `DocumentChunk`, `Query`
- `RetrievalResult`, `ScoredChunk`
- `Answer`, `Citation`
- `BenchmarkMetrics`, `BenchmarkSample`

## CLI

```bash
mosaic-benchmark --quick
mosaic-benchmark --output benchmarks/results
```
