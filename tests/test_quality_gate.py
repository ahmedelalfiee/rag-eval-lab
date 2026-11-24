"""Quality gate: CI fails if offline retrieval quality regresses.

Thresholds sit a little below the current BM25 baseline (recall@5 = 0.885, MRR = 0.807 on
section chunks). A chunking or tokenizer change that hurts retrieval breaks the build.
"""
from raglab.evaluate import eval_retrieval, load_questions
from raglab.pipeline import make_retriever

MIN_RECALL_AT_5 = 0.85
MIN_MRR = 0.78


def test_bm25_retrieval_does_not_regress():
    m = eval_retrieval(make_retriever("bm25", "section"), load_questions())
    assert m["recall@5"] >= MIN_RECALL_AT_5, m
    assert m["mrr"] >= MIN_MRR, m
