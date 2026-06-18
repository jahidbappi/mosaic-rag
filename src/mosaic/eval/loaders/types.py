from __future__ import annotations

from pydantic import BaseModel, Field

from mosaic.types import BenchmarkSample, Document


class EvalBundle(BaseModel):
    """Unified evaluation dataset: indexed corpus + labeled query samples."""

    name: str
    documents: list[Document]
    samples: list[BenchmarkSample]
    citation: str
    license: str
    source_url: str
    num_corpus: int
    num_queries: int
    seed: int = 42
    chunk_during_index: bool = False
    multimodal: bool = False
    notes: str = ""
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
