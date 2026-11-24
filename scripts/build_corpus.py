"""Snapshot the FastAPI docs into data/pages.jsonl.

    git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git
    git -C fastapi checkout 05e6df7e906c3decba377c37548019973f40bf15 && git -C fastapi sparse-checkout set docs/en/docs
    python scripts/build_corpus.py fastapi/docs/en/docs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglab.corpus import build_chunks, load_pages, save_pages  # noqa: E402

docs = Path(sys.argv[1])
pages = load_pages(docs)
out = Path(__file__).resolve().parent.parent / "data" / "pages.jsonl"
save_pages(pages, out)
words = sum(len(t.split()) for t in pages.values())
print(f"{len(pages)} pages, {words:,} words -> {out}")
for s in ("section", "fixed"):
    print(s, len(build_chunks(pages, s)), "chunks")
