from __future__ import annotations

import random
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

from datasets import load_dataset

from mosaic.eval.loaders.types import EvalBundle
from mosaic.types import BenchmarkSample, Document, DocumentChunk, Modality

BEIR_SCIFACT_ZIP = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
CITATION = (
    "Thakur et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation "
    "of Information Retrieval Models. NeurIPS Datasets and Benchmarks Track."
)
LICENSE = "Apache-2.0 (BEIR); SciFact corpus derived from scientific claim verification research."
SOURCE_URL = "https://huggingface.co/datasets/BeIR/scifact"


def _download_qrels(cache_dir: Path) -> dict[str, list[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "scifact.zip"
    if not zip_path.exists():
        urlretrieve(BEIR_SCIFACT_ZIP, zip_path)  # noqa: S310

    qrels: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("scifact/qrels/test.tsv") as handle:
            lines = handle.read().decode("utf-8").splitlines()
    for line in lines[1:]:
        query_id, corpus_id, score = line.split("\t")
        if int(score) > 0:
            qrels[query_id].append(corpus_id)
    return dict(qrels)


def load_beir_scifact(
    *,
    max_samples: int | None = None,
    cache_dir: Path | None = None,
    seed: int = 42,
) -> EvalBundle:
    """Load BEIR SciFact test split from HuggingFace + official qrels."""
    cache = cache_dir or Path.home() / ".cache" / "mosaic" / "datasets"
    qrels = _download_qrels(cache)

    corpus_rows = load_dataset("BeIR/scifact", "corpus", split="corpus")
    query_rows = load_dataset("BeIR/scifact", "queries", split="queries")

    corpus_by_id = {str(row["_id"]): row for row in corpus_rows}
    query_by_id = {str(row["_id"]): row for row in query_rows}

    test_query_ids = sorted(qrels.keys(), key=int)
    if max_samples is not None and max_samples < len(test_query_ids):
        rng = random.Random(seed)
        test_query_ids = sorted(rng.sample(test_query_ids, max_samples), key=int)

    relevant_ids: set[str] = set()
    samples: list[BenchmarkSample] = []
    for query_id in test_query_ids:
        rel_ids = qrels[query_id]
        relevant_ids.update(rel_ids)
        query_row = query_by_id[query_id]
        expected = ""
        if rel_ids and rel_ids[0] in corpus_by_id:
            expected = corpus_by_id[rel_ids[0]]["text"][:240]
        samples.append(
            BenchmarkSample(
                id=query_id,
                query=query_row["text"],
                expected_answer=expected,
                relevant_chunk_ids=rel_ids,
            )
        )

    documents: list[Document] = []
    for corpus_id, row in corpus_by_id.items():
        text = row["text"]
        title = row.get("title") or ""
        content = f"{title}\n{text}".strip() if title else text
        documents.append(
            Document(
                id=corpus_id,
                title=title or corpus_id,
                chunks=[
                    DocumentChunk(
                        id=corpus_id,
                        document_id=corpus_id,
                        content=content,
                        modality=Modality.TEXT,
                    )
                ],
            )
        )

    return EvalBundle(
        name="scifact",
        documents=documents,
        samples=samples,
        citation=CITATION,
        license=LICENSE,
        source_url=SOURCE_URL,
        num_corpus=len(documents),
        num_queries=len(samples),
        seed=seed,
        chunk_during_index=False,
        multimodal=False,
        notes="Document-level IR benchmark; one abstract per corpus entry.",
        metadata={"split": "test", "qrels_source": BEIR_SCIFACT_ZIP},
    )
