"""Deterministic stand-ins for the model APIs, so the whole pipeline is testable offline."""
import hashlib
import json
import re

from raglab.generate import ABSTAIN


class FakeEmbedder:
    """Hashed bag-of-words vectors: similar wording -> similar vectors. Enough to exercise dense code paths."""
    name = "fake-embed"

    def __init__(self, dim: int = 64):
        self.dim, self.calls = dim, 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9_]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            out.append(v)
        return out


class FakeAnswerLLM:
    """Answers by quoting excerpt [1] if it shares a keyword with the question, else abstains."""

    def complete(self, system, user, json_mode=False):
        context, question = user.rsplit("Question:", 1)
        first = context.split("[2]")[0].lower()
        question = question.lower()
        keywords = [w for w in re.findall(r"[a-z_]{6,}", question)]
        if any(w in first for w in keywords):
            return "Use the documented option [1]."
        return ABSTAIN


class FakeJudge:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user, json_mode=False):
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
