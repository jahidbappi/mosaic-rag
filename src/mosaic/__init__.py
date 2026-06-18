"""Mosaic — Multimodal retrieval engine with rigorous evaluation."""

from mosaic.pipeline import MosaicPipeline
from mosaic.types import Answer, Document, Query, RetrievalResult

__version__ = "0.1.0"
__all__ = ["MosaicPipeline", "Document", "Query", "RetrievalResult", "Answer"]
