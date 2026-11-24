from fastapi.testclient import TestClient

import raglab.api as api
from raglab.pipeline import make_retriever
from tests.fakes import FakeAnswerLLM


def test_ask_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(api, "make_retriever", lambda *a, **k: make_retriever("bm25", "section"))
    monkeypatch.setattr(api, "OpenAIChat", lambda *a, **k: FakeAnswerLLM())
    with TestClient(api.app) as c:
        r = c.post("/ask", json={"question": "How do I receive uploaded files with UploadFile?"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"answer", "abstained", "sources"}
        assert c.post("/ask", json={"question": "x"}).status_code == 422
        assert c.get("/health").json()["retriever"] == "hybrid"


def test_docs_url():
    assert api.docs_url("tutorial/query-params") == "https://fastapi.tiangolo.com/tutorial/query-params/"
    assert api.docs_url("index") == "https://fastapi.tiangolo.com/"
    assert api.docs_url("advanced/index") == "https://fastapi.tiangolo.com/advanced/"
