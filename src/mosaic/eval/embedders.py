from __future__ import annotations

import warnings
from typing import Literal

from mosaic.embeddings.base import BaseEmbedder
from mosaic.embeddings.providers import (
    CLIPEmbedder,
    MockMultimodalEmbedder,
    MockTextEmbedder,
    SentenceTransformerEmbedder,
)

EmbedderName = Literal["mock", "minilm", "clip"]


def resolve_embedder(name: EmbedderName) -> BaseEmbedder:
    if name == "mock":
        return MockTextEmbedder()
    if name == "minilm":
        try:
            return SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        except ImportError:
            warnings.warn(
                "sentence-transformers not installed; falling back to mock-text embedder. "
                "Install with: pip install mosaic-rag[ml]",
                stacklevel=2,
            )
            return MockTextEmbedder()
    if name == "clip":
        try:
            return CLIPEmbedder()
        except ImportError:
            warnings.warn(
                "clip embedder requires sentence-transformers; falling back to mock-multimodal.",
                stacklevel=2,
            )
            return MockMultimodalEmbedder()
    raise ValueError(f"Unknown embedder '{name}'. Choose from: mock, minilm, clip")
