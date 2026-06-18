from __future__ import annotations

import random
from pathlib import Path

from datasets import load_dataset

from mosaic.eval.loaders.types import EvalBundle
from mosaic.types import BenchmarkSample, Document, DocumentChunk, Modality

CITATION = (
    "Mathew et al. (2021). DocVQA: A Dataset for VQA on Document Images. WACV."
)
LICENSE = "CC BY 4.0 (DocVQA)"
SOURCE_URL = "https://huggingface.co/datasets/lmms-lab/DocVQA"


def load_docvqa(
    *,
    max_samples: int | None = 50,
    cache_dir: Path | None = None,
    seed: int = 42,
    index_images: bool = False,
) -> EvalBundle:
    """Stream DocVQA validation split; text-only retrieval unless index_images=True."""
    cache = cache_dir or Path.home() / ".cache" / "mosaic" / "datasets"
    image_dir = cache / "docvqa" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    stream = load_dataset(
        "lmms-lab/DocVQA",
        "DocVQA",
        split="validation",
        streaming=True,
    )

    rows: list[dict] = []
    for row in stream:
        rows.append(row)
        if max_samples is not None and len(rows) >= max_samples:
            break

    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]

    documents: list[Document] = []
    samples: list[BenchmarkSample] = []
    for row in rows:
        doc_id = f"docvqa-{row['questionId']}"
        image_path: str | None = None
        if index_images:
            image_path = str(image_dir / f"{row['questionId']}.png")
            row["image"].save(image_path)

        answer_text = row["answers"][0] if row["answers"] else ""
        page_ref = f"page {row['ucsf_document_page_no']} ({row['ucsf_document_id']})"
        content = (
            f"Document: {page_ref}\n"
            f"Question type: {', '.join(row['question_types'])}\n"
            f"Reference answer: {answer_text}"
        )
        chunk = DocumentChunk(
            id=doc_id,
            document_id=doc_id,
            content=content,
            modality=Modality.IMAGE if index_images else Modality.TEXT,
            image_path=image_path,
            metadata={"docId": str(row["docId"]), "page": row["ucsf_document_page_no"]},
        )
        documents.append(Document(id=doc_id, title=page_ref, chunks=[chunk]))
        samples.append(
            BenchmarkSample(
                id=str(row["questionId"]),
                query=row["question"],
                expected_answer=answer_text,
                relevant_chunk_ids=[doc_id],
                image_path=image_path,
            )
        )

    return EvalBundle(
        name="docvqa",
        documents=documents,
        samples=samples,
        citation=CITATION,
        license=LICENSE,
        source_url=SOURCE_URL,
        num_corpus=len(documents),
        num_queries=len(samples),
        seed=seed,
        chunk_during_index=False,
        multimodal=index_images,
        notes=(
            "Text-only retrieval over OCR metadata and reference answers by default. "
            "Pass index_images=True with clip embedder for true multimodal indexing."
        ),
        metadata={"split": "validation", "index_images": index_images},
    )
