"""LLM-as-judge for answer quality, with deliberately narrow, binary questions.

Binary verdicts with a one-line reason are easier to audit and agree with than 1-10 scores.
The judge's agreement with human labels is measured separately (see eval/human_labels.jsonl).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

SYSTEM = """You grade answers from a documentation assistant. Reply with JSON only:
{"correct": true|false, "faithful": true|false, "reason": "<one short sentence>"}
- correct: the answer agrees with the reference answer on the key facts (wording can differ; extra correct detail is fine).
- faithful: every factual claim in the answer is supported by the provided excerpts (ignore citation formatting)."""


@dataclass
class Verdict:
    correct: bool
    faithful: bool
    reason: str


def judge(llm, question: str, reference: str, answer_text: str, context: str) -> Verdict:
    user = (f"Question: {question}\n\nReference answer: {reference}\n\nAnswer to grade: {answer_text}\n\n"
            f"Excerpts the answer was based on:\n{context}")
    try:
        d = json.loads(llm.complete(SYSTEM, user, json_mode=True))
        return Verdict(bool(d.get("correct")), bool(d.get("faithful")), str(d.get("reason", "")))
    except (json.JSONDecodeError, AttributeError):
        return Verdict(False, False, "judge returned invalid JSON")
