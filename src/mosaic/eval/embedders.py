from __future__ import annotations

import warnings
from typing import Literal

from mosaic.embeddings.base import BaseEmbedder
from mosaic.embeddings.ollama import OllamaEmbedder
from mosaic.embeddings.providers import (
    CLIPEmbedder,
    MockMultimodalEmbedder,
    MockTextEmbedder,
    SentenceTransformerEmbedder,
)

EmbedderName = Literal["mock", "minilm", "bge-small", "clip", "ollama"]


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
    if name == "bge-small":
        try:
            return SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
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
    if name == "ollama":
        try:
            return OllamaEmbedder()
        except Exception as exc:
            warnings.warn(
                f"Ollama embedder unavailable ({exc}); falling back to mock-text.",
                stacklevel=2,
            )
            return MockTextEmbedder()
    raise ValueError(f"Unknown embedder '{name}'. Choose from: mock, minilm, bge-small, clip, ollama")
