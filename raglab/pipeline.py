"""Assemble a retriever from a config name; shared by the evaluator and the API."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from .corpus import build_chunks, read_pages
from .llm import OpenAIEmbedder, Usage
from .retrieval import BM25Retriever, DenseRetriever, EmbeddingCache, HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # OPENAI_API_KEY and RAG_* settings, on any OS
PAGES = ROOT / "data" / "pages.jsonl"
CACHE = ROOT / ".cache" / "embeddings.json"
EMBED_MODEL = "text-embedding-3-small"
RETRIEVERS = ("bm25", "dense", "hybrid")


def load_chunks(chunking: str):
    return build_chunks(read_pages(PAGES), chunking)


def make_retriever(kind: str, chunking: str, usage: Usage | None = None, embedder=None, cache=None):
    chunks = load_chunks(chunking)
    if kind == "bm25":
        return BM25Retriever(chunks)
    usage = usage or Usage()
    embedder = embedder or OpenAIEmbedder(EMBED_MODEL, usage)
    cache = cache or EmbeddingCache(CACHE)
    dense = DenseRetriever(chunks, embedder, cache)
    if kind == "dense":
        return dense
    if kind == "hybrid":
        return HybridRetriever([BM25Retriever(chunks), dense])
    raise ValueError(f"unknown retriever {kind!r}; choose from {RETRIEVERS}")
