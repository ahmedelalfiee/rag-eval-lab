import json
from pathlib import Path

import pytest

from raglab.corpus import Chunk, build_chunks, chunk_fixed, chunk_sections, clean, read_pages
from raglab.evaluate import eval_end_to_end, eval_retrieval, load_questions
from raglab.generate import ABSTAIN, answer, parse_citations
from raglab.judge import judge
from raglab.llm import Usage
from raglab.metrics import first_hit_rank, mrr, recall_at_k
from raglab.pipeline import PAGES
from raglab.retrieval import BM25Retriever, DenseRetriever, EmbeddingCache, HybridRetriever
from tests.fakes import FakeAnswerLLM, FakeEmbedder, FakeJudge

PAGES_DATA = read_pages(PAGES)
QUESTIONS = load_questions()


def C(i, page, text="x"):
    return Chunk(f"{page}#{i}", page, page, page, text)


# ---------------------------------------------------------------- corpus
def test_clean_removes_anchors_includes_and_tags():
    md = '## Title { #title }\n{* ../../docs_src/a.py hl[1] *}\nA <dfn title="x">term</dfn>.\n/// tip\nHi\n///'
    out = clean(md)
    assert out.startswith("## Title\n") and "{*" not in out and "<dfn" not in out and "///" not in out
    assert "term" in out and "Hi" in out


def test_section_chunks_follow_headings():
    text = "# Page\n\n" + "intro words here " * 10 + "\n\n## Part A\n\n" + "alpha " * 30 + "\n\n## Part B\n\n" + "beta " * 30
    chunks = chunk_sections("p", text)
    assert [c.section for c in chunks] == ["Page", "Part A", "Part B"]
    assert len({c.id for c in chunks}) == len(chunks)


def test_fixed_chunks_overlap():
    chunks = chunk_fixed("p", " ".join(f"w{i}" for i in range(450)), size=200, overlap=40)
    assert len(chunks) == 3 and chunks[1].text.split()[0] == "w160"


def test_snapshot_has_no_excluded_pages():
    assert not any(p.startswith(("reference/", "release-notes")) for p in PAGES_DATA)
    assert len(PAGES_DATA) > 100


# ---------------------------------------------------------------- eval set integrity
def test_question_ids_unique_and_gold_pages_exist():
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
    for q in QUESTIONS:
        for p in q["gold_pages"]:
            assert p in PAGES_DATA, q["id"]


def test_evidence_is_quoted_from_gold_page():
    for q in QUESTIONS:
        if q["answerable"]:
            assert q["evidence"] in PAGES_DATA[q["gold_pages"][0]], q["id"]


# ---------------------------------------------------------------- retrieval
def test_bm25_ranks_the_matching_chunk_first():
    chunks = [C(0, "a", "uploading files with UploadFile"), C(1, "b", "path parameters and types")]
    assert BM25Retriever(chunks).search("how to upload files", 2)[0][0].page == "a"


def test_rrf_rewards_agreement():
    class Fixed:
        def __init__(self, order): self.order = order
        def search(self, q, k): return [(c, 1.0) for c in self.order][:k]
    a, b, c = C(0, "a"), C(1, "b"), C(2, "c")
    fused = HybridRetriever([Fixed([a, b, c]), Fixed([b, c, a])]).search("q", 3)
    assert fused[0][0].page == "b"           # ranked 2nd and 1st beats 1st and 3rd


def test_embedding_cache_avoids_repeat_calls(tmp_path):
    emb, cache = FakeEmbedder(), EmbeddingCache(tmp_path / "e.json")
    chunks = [C(0, "a", "upload files"), C(1, "b", "path params")]
    DenseRetriever(chunks, emb, cache)
    calls = emb.calls
    DenseRetriever(chunks, emb, EmbeddingCache(tmp_path / "e.json"))
    assert emb.calls == calls


def test_metrics():
    ranks = [1, 3, None, 6]
    assert recall_at_k(ranks, 5) == 0.5 and mrr(ranks) == pytest.approx((1 + 1 / 3 + 1 / 6) / 4)
    assert first_hit_rank([C(0, "x"), C(1, "gold")], ["gold"]) == 2


# ---------------------------------------------------------------- generation and judge
def test_citations_are_parsed_deduped_and_bounded():
    chunks = [C(0, "a"), C(1, "b")]
    assert [c.page for c in parse_citations("x [2] y [1][2] z [9]", chunks)] == ["b", "a"]


def test_abstention_is_detected():
    a = answer(FakeAnswerLLM(), "zzzzzz unknown question", [C(0, "a", "nothing related")])
    assert a.abstained and a.cited == [] and a.text == ABSTAIN


def test_judge_parses_json_and_survives_garbage():
    ok = judge(FakeJudge({"correct": True, "faithful": False, "reason": "r"}), "q", "ref", "ans", "ctx")
    assert ok.correct and not ok.faithful
    bad = judge(FakeJudge("not json"), "q", "ref", "ans", "ctx")
    assert not bad.correct and "invalid" in bad.reason


def test_end_to_end_with_fakes_produces_all_metrics():
    retriever = BM25Retriever(build_chunks(PAGES_DATA, "section"))
    summary, rows = eval_end_to_end(retriever, FakeAnswerLLM(), FakeJudge({"correct": True, "faithful": True, "reason": ""}),
                                    QUESTIONS, k=5, gen_usage=Usage())
    assert len(rows) == len(QUESTIONS)
    assert set(summary) == {"correct_pct", "faithful_pct", "citation_hit_pct", "false_abstain_pct",
                            "abstain_unanswerable_pct", "p50_seconds", "cost_per_1k_questions_usd"}


def test_usage_costs():
    u = Usage()
    u.add("gpt-5.4-mini", 1_000_000, 100_000, 1.0)
    assert u.cost_usd == pytest.approx(0.75 + 0.45)
