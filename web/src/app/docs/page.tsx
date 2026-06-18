import { SiteNav } from "@/components/SiteNav";
import { SiteFooter } from "@/components/SiteFooter";

export default function DocsPage() {
  return (
    <>
      <SiteNav />
      <main className="min-h-screen bg-[#07070d] px-6 pb-20 pt-28 text-white">
        <div className="mx-auto max-w-3xl">
          <p className="text-sm uppercase tracking-[0.2em] text-emerald-300/80">Documentation</p>
          <h1 className="mt-4 text-4xl font-semibold">Quick Start</h1>
          <div className="prose prose-invert mt-8 max-w-none space-y-8 text-white/70">
            <section>
              <h2 className="text-xl font-medium text-white">Install</h2>
              <pre className="mt-4 overflow-x-auto rounded-xl border border-white/10 bg-black/40 p-4 text-sm text-emerald-200">{`git clone https://github.com/jahidbappi/mosaic-rag
cd mosaic-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"`}</pre>
            </section>
            <section>
              <h2 className="text-xl font-medium text-white">Basic usage</h2>
              <pre className="mt-4 overflow-x-auto rounded-xl border border-white/10 bg-black/40 p-4 text-sm text-emerald-200">{`from mosaic import MosaicPipeline
from mosaic.eval.datasets import load_sample_corpus
from mosaic.types import Query

pipeline = MosaicPipeline()
pipeline.index(load_sample_corpus())

result = pipeline.retrieve(Query(text="How does hybrid retrieval work?", top_k=5))
answer = pipeline.answer(Query(text="How does hybrid retrieval work?", top_k=5))
print(answer.text)
print(answer.citations)`}</pre>
            </section>
            <section>
              <h2 className="text-xl font-medium text-white">Run benchmarks</h2>
              <pre className="mt-4 overflow-x-auto rounded-xl border border-white/10 bg-black/40 p-4 text-sm text-emerald-200">{`mosaic-benchmark --output benchmarks/results
cat benchmarks/results/leaderboard.json`}</pre>
            </section>
            <section>
              <h2 className="text-xl font-medium text-white">Components</h2>
              <ul className="mt-4 list-disc space-y-2 pl-6">
                <li><strong className="text-white">Embeddings:</strong> mock-text, mock-multimodal, sentence-transformers, CLIP</li>
                <li><strong className="text-white">Retrieval:</strong> dense, BM25, hybrid</li>
                <li><strong className="text-white">Rerankers:</strong> score-fusion, cross-encoder</li>
                <li><strong className="text-white">Chunking:</strong> fixed-size, semantic</li>
              </ul>
            </section>
          </div>
        </div>
      </main>
      <SiteFooter />
    </>
  );
}
