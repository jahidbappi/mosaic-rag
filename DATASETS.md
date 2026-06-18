# Mosaic Evaluation Datasets

Mosaic benchmarks use **public, citable open-source datasets** — not synthetic corpora — for leaderboard results. A small built-in corpus remains for fast unit tests only.

## SciFact (BEIR)

| Field | Value |
|-------|-------|
| **Loader** | `mosaic.eval.loaders.beir_scifact` |
| **HuggingFace** | [BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact) |
| **Qrels** | [Official BEIR download](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip) |
| **Split** | Test (300 queries, ~5,183 abstracts) |
| **License** | Apache-2.0 (BEIR benchmark) |
| **Task** | Document-level information retrieval |

**Citation:**

> Nils Thakur, Nandan Thakur, Nikhil Reimers, Andreas Rücklé, Abhishek Sreenivas, Iryna Gurevych, and Jimmy Lin. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.** *NeurIPS 2021 Datasets and Benchmarks Track.*

SciFact is a scientific claim verification corpus packaged as an IR benchmark. Each abstract is one document; relevance labels come from the official BEIR qrels.

## DocVQA

| Field | Value |
|-------|-------|
| **Loader** | `mosaic.eval.loaders.docvqa` |
| **HuggingFace** | [lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA) |
| **Split** | Validation (streamed subset via `--max-samples`) |
| **License** | CC BY 4.0 |
| **Task** | Document visual QA (text-only retrieval by default) |

**Citation:**

> Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. **DocVQA: A Dataset for VQA on Document Images.** *WACV 2021.*

By default, DocVQA evaluation indexes **text metadata** (page reference, question type, reference answer). Set `index_images=True` in the loader and use `--embedder clip` for true multimodal indexing. We do not claim multimodal retrieval unless images are indexed.

## Built-in (synthetic)

The built-in corpus (`--dataset builtin`) is a 9-chunk smoke test for CI unit tests only. It is **not** used for the public leaderboard.

## Reproducibility

- **Seed:** `--seed 42` (default) for query subsampling
- **Cache:** `~/.cache/mosaic/datasets` (override with `--cache-dir`)
- **Sample counts:** `--max-samples 20` (CI), `50` (default ablations), `300` (full SciFact test)

## Download

```bash
pip install -e .
python scripts/download_datasets.py --dataset scifact --max-samples 50
python scripts/download_datasets.py --dataset docvqa --max-samples 20
```

## Run benchmarks

```bash
# CI-scale SciFact (mock embedder, no torch)
mosaic-benchmark --dataset scifact --max-samples 20 --embedder mock

# Default ablations (SciFact, 50 queries)
mosaic-benchmark --output benchmarks/results

# Full SciFact with real embeddings
pip install -e ".[ml]"
mosaic-benchmark --dataset scifact --max-samples 300 --embedder minilm
```

Leaderboard JSON includes `dataset`, `num_samples`, `embedder`, `timestamp`, and `citation`.
