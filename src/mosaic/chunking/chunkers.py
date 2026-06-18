from __future__ import annotations

from abc import ABC, abstractmethod

from mosaic.types import Document, DocumentChunk, Modality


class BaseChunker(ABC):
    name: str

    @abstractmethod
    def chunk(self, document: Document) -> list[DocumentChunk]:
        raise NotImplementedError


class FixedSizeChunker(BaseChunker):
    name = "fixed-size"

    def __init__(self, chunk_size: int = 200, overlap: int = 40) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        text = " ".join(
            c.content for c in document.chunks if c.modality == Modality.TEXT
        ) or document.title

        chunks: list[DocumentChunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            content = text[start:end].strip()
            if content:
                chunks.append(
                    DocumentChunk(
                        id=f"{document.id}-fixed-{index}",
                        document_id=document.id,
                        content=content,
                        modality=Modality.TEXT,
                    )
                )
                index += 1
            if end == len(text):
                break
            start = end - self.overlap

        for chunk in document.chunks:
            if chunk.modality == Modality.IMAGE:
                chunks.append(chunk.model_copy(deep=True))

        return chunks


class SemanticChunker(BaseChunker):
    name = "semantic"

    def chunk(self, document: Document) -> list[DocumentChunk]:
        paragraphs = [
            c.content.strip()
            for c in document.chunks
            if c.modality == Modality.TEXT and c.content.strip()
        ]
        if not paragraphs:
            paragraphs = [document.title]

        chunks: list[DocumentChunk] = []
        for i, paragraph in enumerate(paragraphs):
            chunks.append(
                DocumentChunk(
                    id=f"{document.id}-semantic-{i}",
                    document_id=document.id,
                    content=paragraph,
                    modality=Modality.TEXT,
                )
            )

        for chunk in document.chunks:
            if chunk.modality == Modality.IMAGE:
                chunks.append(chunk.model_copy(deep=True))

        return chunks
