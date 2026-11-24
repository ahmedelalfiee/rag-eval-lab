"""Retrievers: BM25 (lexical), dense (embeddings), and a hybrid that fuses both.

Every retriever returns a ranked list of (chunk, score). Dense embeddings are cached on disk,
keyed by model and text hash, so re-running the evaluation does not pay for them twice.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Protocol

from .corpus import Chunk

_TOKEN = re.compile(r"[a-z0-9_]+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it my of on or should the "
    "this to what when where which why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]: ...


class BM25Retriever:
    """Okapi BM25 over title + section + text, implemented directly (no index server needed)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df: dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [self._counts(d) for d in self.docs]

    @staticmethod
    def _counts(tokens: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in tokens:
            out[t] = out.get(t, 0) + 1
        return out

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl, s = len(self.docs[i]), 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [(self.chunks[i], s) for s, i in scores[:k]]


class Embedder(Protocol):
    name: str
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, list[float]] = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def key(model: str, text: str) -> str:
        return model + ":" + hashlib.sha256(text.encode()).hexdigest()[:24]

    def embed(self, embedder: Embedder, texts: list[str], batch: int = 128) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if self.key(embedder.name, t) not in self.data]
        for i in range(0, len(missing), batch):
            part = missing[i:i + batch]
            for t, v in zip(part, embedder.embed(part)):
                self.data[self.key(embedder.name, t)] = v
        if missing:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data))
        return [self.data[self.key(embedder.name, t)] for t in texts]


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, cache: EmbeddingCache):
        self.chunks, self.embedder, self.cache = chunks, embedder, cache
        texts = [f"{c.title} - {c.section}\n{c.text}" for c in chunks]
        self.vectors = [_norm(v) for v in cache.embed(embedder, texts)]

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = _norm(self.cache.embed(self.embedder, [query])[0])
        scored = sorted(((sum(a * b for a, b in zip(q, v)), i) for i, v in enumerate(self.vectors)), reverse=True)
        return [(self.chunks[i], s) for s, i in scored[:k]]


class HybridRetriever:
    """Reciprocal rank fusion: score = sum over retrievers of 1 / (rrf_k + rank).

    RRF needs no score calibration between BM25 and cosine similarity, which live on different scales.
    """

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, depth: int = 50):
        self.retrievers, self.rrf_k, self.depth = retrievers, rrf_k, depth

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for r in self.retrievers:
            for rank, (c, _) in enumerate(r.search(query, self.depth), start=1):
                fused[c.id] = fused.get(c.id, 0.0) + 1.0 / (self.rrf_k + rank)
                by_id[c.id] = c
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(by_id[i], s) for i, s in top]
