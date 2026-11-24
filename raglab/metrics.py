"""Retrieval metrics at the page level: a hit means a retrieved chunk comes from a gold page."""
from __future__ import annotations

from .corpus import Chunk


def first_hit_rank(results: list[Chunk], gold_pages: list[str]) -> int | None:
    for rank, c in enumerate(results, start=1):
        if c.page in gold_pages:
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r) / len(ranks) if ranks else 0.0
