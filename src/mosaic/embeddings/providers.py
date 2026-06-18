from __future__ import annotations

import hashlib

import numpy as np

from mosaic.embeddings.base import BaseEmbedder


def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
    digest = hashlib.sha256(text.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class MockTextEmbedder(BaseEmbedder):
    name = "mock-text"

    def embed_text(self, text: str) -> np.ndarray:
        return _hash_embed(f"text:{text}")

    def embed_image(self, image_path: str) -> np.ndarray:
        return _hash_embed(f"image:{image_path}")


class MockMultimodalEmbedder(BaseEmbedder):
    name = "mock-multimodal"

    def embed_text(self, text: str) -> np.ndarray:
        return _hash_embed(f"mm-text:{text}", dim=512)

    def embed_image(self, image_path: str) -> np.ndarray:
        return _hash_embed(f"mm-image:{image_path}", dim=512)


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        return np.asarray(self._model.encode(text), dtype=np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        # Fallback to filename embedding when no vision model loaded
        return _hash_embed(f"st-image:{image_path}")


class CLIPEmbedder(BaseEmbedder):
    name = "clip"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("clip-ViT-B-32")
        except Exception:
            self._model = None

    def embed_text(self, text: str) -> np.ndarray:
        if self._model is None:
            return _hash_embed(f"clip-text:{text}", dim=512)
        return np.asarray(self._model.encode(text), dtype=np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        if self._model is None:
            return _hash_embed(f"clip-image:{image_path}", dim=512)
        return np.asarray(self._model.encode(image_path), dtype=np.float32)
