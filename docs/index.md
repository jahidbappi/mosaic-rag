# Mosaic Documentation

Mosaic is a multimodal RAG engine for documents that mix text and images.

## Install

```bash
pip install -e ".[dev]"
```

## Run a query

```python
from mosaic import MosaicPipeline
from mosaic.eval.datasets import load_sample_corpus
from mosaic.types import Query

pipeline = MosaicPipeline()
pipeline.index(load_sample_corpus())
print(pipeline.answer(Query(text="What is recall at k?", top_k=3)))
```

## Design principles

1. **Composable** — every stage is swappable
2. **Measurable** — every run produces metrics
3. **Reproducible** — pinned seeds and one-command benchmarks
