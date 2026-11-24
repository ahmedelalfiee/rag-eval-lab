"""LLM-as-judge for answer quality, as two separate, narrow, binary checks.

v1 asked one prompt for both verdicts, giving it the reference answer AND the retrieved
excerpts. An audit of its "unfaithful" verdicts showed it judged faithfulness against the
reference instead of the excerpts: correct extra details from the docs were flagged as
unsupported. v2 isolates each check so it only sees what it is supposed to judge:
  - correctness:  question + reference + answer      (no excerpts)
  - faithfulness: excerpts + answer                  (no reference)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

CORRECT_SYSTEM = """You check an answer from a documentation assistant against a reference answer.
Reply with JSON only: {"correct": true|false, "reason": "<one short sentence>"}
correct = the answer states the key facts of the reference (wording can differ). Extra details,
alternatives or caveats do NOT make it incorrect unless they contradict the reference."""

FAITHFUL_SYSTEM = """You check whether an answer is supported by documentation excerpts.
Reply with JSON only: {"faithful": true|false, "unsupported": "<the first unsupported claim, or empty>"}
faithful = every factual claim in the answer can be found in, or directly follows from, the excerpts.
Judge ONLY against the excerpts, not your own knowledge. Ignore citation markers like [1]."""


@dataclass
class Verdict:
    correct: bool
    faithful: bool
    reason: str


def _ask(llm, system: str, user: str) -> dict | None:
    try:
        d = json.loads(llm.complete(system, user, json_mode=True))
        return d if isinstance(d, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def judge(llm, question: str, reference: str, answer_text: str, context: str) -> Verdict:
    c = _ask(llm, CORRECT_SYSTEM, f"Question: {question}\n\nReference answer: {reference}\n\nAnswer to check: {answer_text}")
    f = _ask(llm, FAITHFUL_SYSTEM, f"Excerpts:\n{context}\n\nAnswer to check: {answer_text}")
    if c is None or f is None:
        return Verdict(False, False, "judge returned invalid JSON")
    reason = str(c.get("reason", ""))
    if not f.get("faithful") and f.get("unsupported"):
        reason += f" | unsupported: {f['unsupported']}"
    return Verdict(bool(c.get("correct")), bool(f.get("faithful")), reason)
