"""Thin model clients that record tokens, latency and cost for every call."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

# USD per 1M tokens (input, output). Checked against OpenAI's pricing page in Nov 2026;
# update here if prices change. Embeddings: input price only.
PRICES = {
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float = 0.0
    per_model: dict = field(default_factory=dict)

    def add(self, model: str, tin: int, tout: int, secs: float) -> None:
        pin, pout = PRICES.get(model, (0.0, 0.0))
        cost = tin / 1e6 * pin + tout / 1e6 * pout
        self.calls += 1
        self.input_tokens += tin
        self.output_tokens += tout
        self.seconds += secs
        self.cost_usd += cost
        m = self.per_model.setdefault(model, {"calls": 0, "cost_usd": 0.0})
        m["calls"] += 1
        m["cost_usd"] += cost


class OpenAIChat:
    def __init__(self, model: str, usage: Usage):
        from openai import OpenAI
        self.client, self.model, self.usage = OpenAI(api_key=os.environ["OPENAI_API_KEY"]), model, usage

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **({"response_format": {"type": "json_object"}} if json_mode else {}),
        )
        u = r.usage
        self.usage.add(self.model, u.prompt_tokens, u.completion_tokens, time.perf_counter() - t0)
        return r.choices[0].message.content or ""


class OpenAIEmbedder:
    def __init__(self, model: str, usage: Usage):
        from openai import OpenAI
        self.client, self.name, self.usage = OpenAI(api_key=os.environ["OPENAI_API_KEY"]), model, usage

    def embed(self, texts: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        r = self.client.embeddings.create(model=self.name, input=texts)
        self.usage.add(self.name, r.usage.prompt_tokens, 0, time.perf_counter() - t0)
        return [d.embedding for d in r.data]
