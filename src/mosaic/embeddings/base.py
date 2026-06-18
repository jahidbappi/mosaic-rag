from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from mosaic.types import DocumentChunk, Modality


class BaseEmbedder(ABC):
    name: str

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_image(self, image_path: str) -> np.ndarray:
        raise NotImplementedError

    def embed_chunk(self, chunk: DocumentChunk) -> np.ndarray:
        if chunk.modality == Modality.IMAGE and chunk.image_path:
            return self.embed_image(chunk.image_path)
        return self.embed_text(chunk.content)
