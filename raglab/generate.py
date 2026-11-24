"""Grounded answering: answer only from retrieved chunks, cite them, or abstain."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Chunk

ABSTAIN = "I don't know based on the FastAPI documentation."

SYSTEM = f"""You answer questions about FastAPI using ONLY the numbered documentation excerpts provided.
Rules:
- Every sentence that states a fact must cite at least one excerpt, like [2] or [1][3].
- If the excerpts do not contain the answer, reply exactly: {ABSTAIN}
- Be concise: at most 4 sentences. Include the exact parameter, class or command names from the docs."""


@dataclass
class Answer:
    text: str
    cited: list[Chunk]
    abstained: bool


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.title} > {c.section})\n{c.text}" for i, c in enumerate(chunks, start=1))


def parse_citations(text: str, chunks: list[Chunk]) -> list[Chunk]:
    nums = []
    for n in re.findall(r"\[(\d+)\]", text):
        i = int(n)
        if 1 <= i <= len(chunks) and i not in nums:
            nums.append(i)
    return [chunks[i - 1] for i in nums]


def answer(llm, question: str, chunks: list[Chunk]) -> Answer:
    text = llm.complete(SYSTEM, f"Documentation excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}").strip()
    abstained = ABSTAIN.lower().rstrip(".") in text.lower()
    return Answer(text, [] if abstained else parse_citations(text, chunks), abstained)
