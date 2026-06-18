import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

const siteUrl = process.env.NEXT_PUBLIC_APP_URL ?? "https://mosaic-rag.vercel.app";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: { default: "Mosaic — Multimodal RAG Engine", template: "%s · Mosaic" },
  description:
    "Open-source multimodal retrieval engine with hybrid search, rerankers, and rigorous benchmark ablations.",
  keywords: ["RAG", "multimodal", "retrieval", "benchmark", "AI", "embeddings"],
  authors: [{ name: "Md. Jahidul Islam", url: "https://github.com/jahidbappi" }],
  openGraph: {
    type: "website",
    url: siteUrl,
    title: "Mosaic — Multimodal RAG Engine",
    description: "Pluggable retrieval, reranking, and reproducible eval harness.",
    siteName: "Mosaic",
  },
  twitter: {
    card: "summary_large_image",
    title: "Mosaic — Multimodal RAG Engine",
    description: "Hybrid retrieval + benchmark ablations for multimodal documents.",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#07070d]`}>
        {children}
      </body>
    </html>
  );
}
