"""Turn the FastAPI markdown docs into clean pages and chunks.

Two chunking strategies are compared in the evaluation:
  - "section": one chunk per markdown section (## / ###), split further only if too long.
  - "fixed":   fixed windows of N words with overlap, ignoring document structure.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# Pages that are not product documentation (people lists, release notes, banners, API stubs).
EXCLUDE = re.compile(
    r"^(_llm-test|agent-banner|translation-banner|release-notes|fastapi-people|newsletter|"
    r"external-links|management|translations|help-fastapi|contributing|about/|reference/)"
)

_ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}")            # "## Title { #title }"
_INCLUDE = re.compile(r"^\{\*.*\*\}\s*$", re.M)       # "{* ../../docs_src/x.py hl[9] *}"
_ADMONITION = re.compile(r"^///.*$", re.M)            # "/// tip" ... "///"
_TAG = re.compile(r"</?(dfn|abbr|span|div|img|a|br|details|summary)[^>]*>")
_SPACES = re.compile(r"\n{3,}")


@dataclass
class Chunk:
    id: str          # "<page>#<n>"
    page: str        # e.g. "tutorial/query-params"
    title: str       # page title
    section: str     # nearest heading
    text: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def clean(md: str) -> str:
    md = _INCLUDE.sub("", md)
    md = _ADMONITION.sub("", md)
    md = _TAG.sub("", md)
    md = "\n".join(_ANCHOR.sub("", line) for line in md.splitlines())
    return _SPACES.sub("\n\n", md).strip()


def load_pages(docs_dir: Path) -> dict[str, str]:
    pages = {}
    for path in sorted(docs_dir.rglob("*.md")):
        rel = path.relative_to(docs_dir).with_suffix("").as_posix()
        if EXCLUDE.match(rel):
            continue
        text = clean(path.read_text(encoding="utf-8"))
        if len(text.split()) >= 30:
            pages[rel] = text
    return pages


def _title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else fallback


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def chunk_sections(page: str, text: str, max_words: int = 350) -> list[Chunk]:
    title = _title(text, page)
    parts = re.split(r"^(#{2,3}\s+.+)$", text, flags=re.M)
    sections = [(title, parts[0])]
    for i in range(1, len(parts), 2):
        sections.append((parts[i].lstrip("#").strip(), parts[i] + "\n" + parts[i + 1]))
    out = []
    for heading, body in sections:
        if len(body.split()) < 15:
            continue
        for piece in _split_words(body.strip(), max_words, 50):
            out.append(Chunk(f"{page}#{len(out)}", page, title, heading, piece))
    return out


def chunk_fixed(page: str, text: str, size: int = 200, overlap: int = 40) -> list[Chunk]:
    title = _title(text, page)
    return [Chunk(f"{page}#{i}", page, title, title, piece)
            for i, piece in enumerate(_split_words(text, size, overlap))]


STRATEGIES = {"section": chunk_sections, "fixed": chunk_fixed}


def build_chunks(pages: dict[str, str], strategy: str) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    return [c for page, text in pages.items() for c in fn(page, text)]


def save_pages(pages: dict[str, str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for page, text in pages.items():
            f.write(json.dumps({"page": page, "text": text}, ensure_ascii=False) + "\n")


def read_pages(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        return {d["page"]: d["text"] for d in map(json.loads, f)}
