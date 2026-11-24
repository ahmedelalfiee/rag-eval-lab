"""HTTP API: POST /ask {"question": "..."} -> grounded answer with sources.

    uvicorn raglab.api:app --port 8000
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generate import answer
from .llm import OpenAIChat, Usage
from .pipeline import make_retriever

RETRIEVER = os.getenv("RAG_RETRIEVER", "hybrid")
CHUNKING = os.getenv("RAG_CHUNKING", "section")
MODEL = os.getenv("RAG_MODEL", "gpt-5.4-mini")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["usage"] = Usage()
    state["retriever"] = make_retriever(RETRIEVER, CHUNKING, state["usage"])
    state["llm"] = OpenAIChat(MODEL, state["usage"])
    yield
    state.clear()


app = FastAPI(title="FastAPI docs assistant", lifespan=lifespan)


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    page: str
    section: str
    url: str


class Reply(BaseModel):
    answer: str
    abstained: bool
    sources: list[Source]


@app.post("/ask", response_model=Reply)
def ask(body: Ask) -> Reply:
    if "retriever" not in state:
        raise HTTPException(503, "not ready")
    hits = [c for c, _ in state["retriever"].search(body.question, TOP_K)]
    a = answer(state["llm"], body.question, hits)
    return Reply(answer=a.text, abstained=a.abstained, sources=[
        Source(page=c.page, section=c.section, url=docs_url(c.page)) for c in a.cited])


def docs_url(page: str) -> str:
    path = page.removesuffix("index").rstrip("/")
    return "https://fastapi.tiangolo.com/" + (path + "/" if path else "")


@app.get("/health")
def health() -> dict:
    u = state.get("usage")
    return {"retriever": RETRIEVER, "chunking": CHUNKING, "model": MODEL,
            "llm_calls": u.calls if u else 0, "cost_usd": round(u.cost_usd, 4) if u else 0}
