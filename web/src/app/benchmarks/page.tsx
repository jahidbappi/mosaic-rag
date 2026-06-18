import { SiteNav } from "@/components/SiteNav";
import { SiteFooter } from "@/components/SiteFooter";
import leaderboardData from "@/data/leaderboard.json";

type LeaderboardRow = {
  name: string;
  strategy: string;
  embedder: string;
  metrics: {
    mrr: number;
    recall_at_5: number;
    faithfulness: number;
    p50_latency_ms: number;
  };
};

type LeaderboardFile = {
  dataset: string;
  num_samples: number;
  num_corpus: number;
  embedder: string;
  timestamp: string;
  citation: string;
  license?: string;
  source_url?: string;
  seed?: number;
  multimodal?: boolean;
  results: LeaderboardRow[];
};

function isWrapped(data: unknown): data is LeaderboardFile {
  return typeof data === "object" && data !== null && "results" in data;
}

export default function BenchmarksPage() {
  const meta = isWrapped(leaderboardData)
    ? leaderboardData
    : {
        dataset: "builtin",
        num_samples: (leaderboardData as LeaderboardRow[]).length,
        num_corpus: 0,
        embedder: "mock",
        timestamp: "",
        citation: "",
        seed: undefined as number | undefined,
        source_url: undefined as string | undefined,
        results: leaderboardData as LeaderboardRow[],
      };

  const rows = meta.results;

  return (
    <>
      <SiteNav />
      <main className="min-h-screen bg-[#07070d] px-6 pb-20 pt-28 text-white">
        <div className="mx-auto max-w-5xl">
          <p className="text-sm uppercase tracking-[0.2em] text-emerald-300/80">Evaluation</p>
          <h1 className="mt-4 text-4xl font-semibold">Benchmark Leaderboard</h1>
          <p className="mt-4 text-white/50">
            Reproducible ablation results on{" "}
            <span className="text-emerald-300">{meta.dataset}</span> ({meta.num_samples} queries,{" "}
            {meta.num_corpus > 0 ? `${meta.num_corpus.toLocaleString()} corpus docs` : "built-in corpus"}).
            Ranked by MRR.
          </p>

          <div className="mt-6 rounded-xl border border-white/10 bg-white/[0.02] p-4 text-sm text-white/60">
            <p>
              <span className="text-white/80">Embedder:</span> {meta.embedder}
              {meta.timestamp && (
                <>
                  {" "}
                  · <span className="text-white/80">Run:</span>{" "}
                  {new Date(meta.timestamp).toLocaleString()}
                </>
              )}
              {meta.seed !== undefined && (
                <>
                  {" "}
                  · <span className="text-white/80">Seed:</span> {meta.seed}
                </>
              )}
            </p>
            {meta.citation && (
              <p className="mt-2 text-xs leading-relaxed text-white/40">{meta.citation}</p>
            )}
            {meta.embedder.includes("mock") && (
              <p className="mt-2 text-xs text-amber-300/80">
                Mock embedder: dense/hybrid scores are not semantically meaningful. BM25 sparse
                retrieval reflects real SciFact IR performance. Use{" "}
                <code className="text-emerald-300">--embedder minilm</code> for full dense/hybrid
                benchmarks.
              </p>
            )}
            {meta.source_url && (
              <a
                href={meta.source_url}
                className="mt-2 inline-block text-emerald-300 hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                Dataset source →
              </a>
            )}
          </div>

          <div className="mt-10 overflow-x-auto rounded-2xl border border-white/10">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-white/10 bg-white/[0.03]">
                <tr>
                  <th className="px-4 py-3 font-medium text-white/60">Rank</th>
                  <th className="px-4 py-3 font-medium text-white/60">Config</th>
                  <th className="px-4 py-3 font-medium text-white/60">Strategy</th>
                  <th className="px-4 py-3 font-medium text-white/60">MRR</th>
                  <th className="px-4 py-3 font-medium text-white/60">Recall@5</th>
                  <th className="px-4 py-3 font-medium text-white/60">Faithfulness</th>
                  <th className="px-4 py-3 font-medium text-white/60">p50 (ms)</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={row.name} className="border-b border-white/5 hover:bg-white/[0.02]">
                    <td className="px-4 py-3 text-emerald-300">#{i + 1}</td>
                    <td className="px-4 py-3 font-medium">{row.name}</td>
                    <td className="px-4 py-3 text-white/60">{row.strategy}</td>
                    <td className="px-4 py-3">{(row.metrics.mrr * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3">{(row.metrics.recall_at_5 * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3">{(row.metrics.faithfulness * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3">{row.metrics.p50_latency_ms.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-8 text-sm text-white/40">
            Reproduce:{" "}
            <code className="text-emerald-300">
              mosaic-benchmark --dataset {meta.dataset} --max-samples {meta.num_samples} --embedder{" "}
              {meta.embedder} --output benchmarks/results
            </code>
          </p>
        </div>
      </main>
      <SiteFooter />
    </>
  );
}
