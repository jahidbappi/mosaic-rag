from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

from mosaic.embeddings.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """Local embeddings via Ollama API (free). Requires `ollama pull nomic-embed-text`."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str | None = None,
    ) -> None:
        self.name = f"ollama/{model_name}"
        self._model = model_name
        self._base = base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    def embed_text(self, text: str) -> np.ndarray:
        payload = json.dumps({"model": self._model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self._base.rstrip('/')}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Ollama embeddings unavailable at {self._base}. "
                "Run: ollama pull nomic-embed-text"
            ) from exc

        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError("Ollama returned empty embedding")
        return np.asarray(embedding, dtype=np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        raise NotImplementedError("Ollama text embedders do not support images; use --embedder clip")
