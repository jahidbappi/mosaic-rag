"use client";

import { motion } from "framer-motion";
import { Layers, GitBranch, BarChart3 } from "lucide-react";
import Link from "next/link";
import { SiteNav } from "@/components/SiteNav";
import { SiteFooter } from "@/components/SiteFooter";

export default function Home() {
  return (
    <>
      <SiteNav />
      <main>
        <section className="relative min-h-screen overflow-hidden px-6 pb-20 pt-32">
          <div className="pointer-events-none absolute inset-0">
            <div className="absolute -left-32 top-20 h-96 w-96 rounded-full bg-emerald-600/20 blur-[120px]" />
            <div className="absolute right-0 top-1/3 h-80 w-80 rounded-full bg-teal-500/15 blur-[100px]" />
          </div>
          <div className="relative mx-auto flex max-w-6xl flex-col items-center text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-emerald-200"
            >
              <Layers className="h-4 w-4" />
              Multimodal retrieval engine
            </motion.div>
            <motion.h1
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="max-w-4xl text-5xl font-semibold tracking-tight text-white md:text-7xl"
            >
              Retrieve. Rerank.{" "}
              <span className="bg-gradient-to-r from-emerald-300 via-teal-300 to-cyan-300 bg-clip-text text-transparent">
                Ground.
              </span>
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mt-6 max-w-2xl text-lg text-white/60 md:text-xl"
            >
              Mosaic assembles text and image chunks into cited, grounded answers — with pluggable
              embeddings, hybrid retrieval, rerankers, and honest benchmark ablations.
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mt-10 flex flex-wrap justify-center gap-4"
            >
              <Link href="/docs" className="rounded-full bg-emerald-500 px-8 py-3 text-sm font-medium text-white hover:bg-emerald-400">
                Read the Docs
              </Link>
              <Link href="/benchmarks" className="rounded-full border border-white/10 px-8 py-3 text-sm font-medium text-white/80 hover:bg-white/5">
                View Benchmarks
              </Link>
            </motion.div>
          </div>
        </section>

        <section className="border-t border-white/5 px-6 py-24">
          <div className="mx-auto grid max-w-6xl gap-6 md:grid-cols-3">
            {[
              { icon: Layers, title: "Pluggable pipeline", desc: "Swap embedders, retrievers, rerankers, and chunkers independently." },
              { icon: GitBranch, title: "Hybrid retrieval", desc: "Dense + BM25 fusion for keyword-heavy enterprise documents." },
              { icon: BarChart3, title: "Rigorous evals", desc: "Recall@k, MRR, faithfulness, latency, and cost per query." },
            ].map((item) => (
              <div key={item.title} className="rounded-2xl border border-white/5 bg-white/[0.03] p-6">
                <item.icon className="h-6 w-6 text-emerald-400" />
                <h3 className="mt-4 text-lg font-medium text-white">{item.title}</h3>
                <p className="mt-2 text-sm text-white/50">{item.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="border-t border-white/5 bg-[#050508] px-6 py-24">
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="text-3xl font-semibold text-white">Built like internal FAANG tooling</h2>
            <p className="mt-4 text-white/50">
              Typed Python API, pytest coverage, ruff + mypy, CI on every push, and reproducible
              one-command benchmarks — not a notebook demo.
            </p>
            <pre className="mt-8 overflow-x-auto rounded-2xl border border-white/10 bg-black/40 p-6 text-left text-sm text-emerald-200/90">
{`pip install mosaic-rag
mosaic-benchmark --output benchmarks/results`}
            </pre>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
