import { SiteNav } from "@/components/SiteNav";
import { SiteFooter } from "@/components/SiteFooter";
import leaderboard from "@/data/leaderboard.json";

export default function BenchmarksPage() {
  return (
    <>
      <SiteNav />
      <main className="min-h-screen bg-[#07070d] px-6 pb-20 pt-28 text-white">
        <div className="mx-auto max-w-5xl">
          <p className="text-sm uppercase tracking-[0.2em] text-emerald-300/80">Evaluation</p>
          <h1 className="mt-4 text-4xl font-semibold">Benchmark Leaderboard</h1>
          <p className="mt-4 text-white/50">
            Reproducible ablation results on the built-in multimodal QA corpus. Ranked by MRR.
          </p>
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
                {leaderboard.map((row, i) => (
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
            Reproduce: <code className="text-emerald-300">mosaic-benchmark --output benchmarks/results</code>
          </p>
        </div>
      </main>
      <SiteFooter />
    </>
  );
}
