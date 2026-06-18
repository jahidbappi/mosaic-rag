import Link from "next/link";
import { Code2 } from "lucide-react";

const IRIS_URL = process.env.NEXT_PUBLIC_IRIS_URL ?? "https://iris-puce.vercel.app";

export function SiteNav() {
  return (
    <header className="fixed top-0 z-50 w-full border-b border-white/5 bg-[#07070d]/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="text-lg font-semibold tracking-tight text-white">
          Mosaic
        </Link>
        <nav className="hidden items-center gap-8 text-sm text-white/60 md:flex">
          <Link href="/docs" className="hover:text-white transition-colors">Docs</Link>
          <Link href="/benchmarks" className="hover:text-white transition-colors">Benchmarks</Link>
          <a href={IRIS_URL} className="hover:text-white transition-colors" target="_blank" rel="noopener noreferrer">Iris</a>
        </nav>
        <div className="flex items-center gap-3">
          <a
            href="https://github.com/jahidbappi/mosaic-rag"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-white/10 p-2 text-white/70 hover:bg-white/5"
            aria-label="GitHub"
          >
            <Code2 className="h-4 w-4" />
          </a>
          <a
            href="https://github.com/jahidbappi/mosaic-rag"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full bg-emerald-500 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-400"
          >
            View on GitHub
          </a>
        </div>
      </div>
    </header>
  );
}
