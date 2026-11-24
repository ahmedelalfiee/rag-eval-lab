"""Run the evaluation and write results tables.

    python -m raglab.evaluate retrieval                     # every retriever x chunking (dense needs OPENAI_API_KEY)
    python -m raglab.evaluate retrieval --only bm25         # offline, no key
    python -m raglab.evaluate e2e --retriever hybrid bm25 --chunking section --k 5 --model gpt-5.4-mini gpt-5.4-nano
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

from .generate import answer, format_context
from .judge import judge
from .llm import OpenAIChat, Usage
from .metrics import first_hit_rank, mrr, recall_at_k
from .pipeline import RETRIEVERS, ROOT, make_retriever

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS = ROOT / "results"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_retrieval(retriever, questions: list[dict], depth: int = 20) -> dict:
    qs = [q for q in questions if q["answerable"]]
    ranks = [first_hit_rank([c for c, _ in retriever.search(q["question"], depth)], q["gold_pages"]) for q in qs]
    return {"n": len(qs), **{f"recall@{k}": round(recall_at_k(ranks, k), 3) for k in (1, 3, 5, 10)},
            "mrr": round(mrr(ranks), 3)}


def eval_end_to_end(retriever, llm, judge_llm, questions: list[dict], k: int, gen_usage: Usage) -> tuple[dict, list[dict]]:
    rows = []
    for q in questions:
        hits = [c for c, _ in retriever.search(q["question"], k)]
        before = gen_usage.cost_usd
        t0 = time.perf_counter()
        a = answer(llm, q["question"], hits)
        secs = time.perf_counter() - t0
        row = {"id": q["id"], "answerable": q["answerable"], "answer": a.text, "abstained": a.abstained,
               "seconds": round(secs, 2), "cost_usd": round(gen_usage.cost_usd - before, 6),
               "cited_pages": [c.page for c in a.cited]}
        if q["answerable"]:
            row["citation_hit"] = any(c.page in q["gold_pages"] for c in a.cited)
            if a.abstained:
                row.update(correct=False, faithful=True, reason="abstained")
            else:
                v = judge(judge_llm, q["question"], q["reference"], a.text, format_context(hits))
                row.update(correct=v.correct, faithful=v.faithful, reason=v.reason)
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else 0.0  # noqa: E731
    summary = {
        "correct_pct": pct([r["correct"] for r in ans]),
        "faithful_pct": pct([r["faithful"] for r in ans]),
        "citation_hit_pct": pct([r["citation_hit"] for r in ans]),
        "false_abstain_pct": pct([r["abstained"] for r in ans]),
        "abstain_unanswerable_pct": pct([r["abstained"] for r in unans]),
        "p50_seconds": round(statistics.median(r["seconds"] for r in rows), 2),
        "cost_per_1k_questions_usd": round(1000 * sum(r["cost_usd"] for r in rows) / len(rows), 3),
    }
    return summary, rows


def _write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = path.with_suffix(".md")
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--only", nargs="*", choices=RETRIEVERS, default=list(RETRIEVERS))
    e = sub.add_parser("e2e")
    e.add_argument("--retriever", nargs="+", choices=RETRIEVERS, default=["hybrid"])
    e.add_argument("--chunking", nargs="+", choices=("section", "fixed"), default=["section"])
    e.add_argument("--k", nargs="+", type=int, default=[5])
    e.add_argument("--model", nargs="+", default=["gpt-5.4-mini"])
    e.add_argument("--judge", default="gpt-5.4-mini")
    a = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    questions = load_questions()
    usage = Usage()

    if a.cmd == "retrieval":
        rows = []
        for kind in a.only:
            for chunking in ("section", "fixed"):
                rows.append({"retriever": kind, "chunking": chunking,
                             **eval_retrieval(make_retriever(kind, chunking, usage), questions)})
        _write_table(RESULTS / "retrieval.csv", rows)
        if usage.calls:
            print(f"embedding cost: ${usage.cost_usd:.4f}")
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (see .env.example) to run the end-to-end evaluation.")
    judge_usage = Usage()
    judge_llm = OpenAIChat(a.judge, judge_usage)
    rows = []
    (RESULTS / "runs").mkdir(exist_ok=True)
    for kind in a.retriever:
        for chunking in a.chunking:
            retriever = make_retriever(kind, chunking, usage)
            for k in a.k:
                for model in a.model:
                    gen_usage = Usage()
                    summary, per_q = eval_end_to_end(retriever, OpenAIChat(model, gen_usage), judge_llm, questions, k, gen_usage)
                    name = f"{kind}_{chunking}_k{k}_{model}"
                    (RESULTS / "runs" / f"{name}.jsonl").write_text(
                        "\n".join(json.dumps(x, ensure_ascii=False) for x in per_q) + "\n", encoding="utf-8")
                    rows.append({"retriever": kind, "chunking": chunking, "k": k, "model": model, **summary})
    _write_table(RESULTS / "end_to_end.csv", rows)
    print(f"judge cost: ${judge_usage.cost_usd:.3f}")


if __name__ == "__main__":
    main()
