# Mosaic Benchmark Report

Generated as part of the Mosaic evaluation harness.

## How to regenerate

```bash
cd mosaic
pip install -e ".[dev]"
mosaic-benchmark --output benchmarks/results
```

## Ablation matrix

| Config | Strategy | Chunker | Embedder | Reranker |
|--------|----------|---------|----------|----------|
| dense-fixed-mock | dense | fixed-size | mock-text | score-fusion |
| hybrid-semantic-mm | hybrid | semantic | mock-multimodal | score-fusion |
| hybrid-fixed-mm | hybrid | fixed-size | mock-multimodal | cross-encoder |
| bm25-semantic-mock | bm25 | semantic | mock-text | score-fusion |

## Interpretation guide

- Optimize for **MRR** when users need the first result to be correct (search UX)
- Optimize for **Recall@5** when downstream LLM can synthesize from broader context
- Track **p95 latency** for production SLAs; rerankers trade latency for precision

## Next steps

- Plug in real document QA datasets (DocVQA, SlideVQA)
- Add LLM-as-judge faithfulness scoring with rubric
- Publish public leaderboard on GitHub Pages
