"""Load care-transition guidance documents (Markdown with a small front-matter header)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_document(text: str, doc_id: str) -> dict[str, Any]:
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        _, header, body = text.split("---", 2)
        for line in header.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
    return {
        "id": doc_id,
        "title": meta.get("title", doc_id.replace("-", " ").title()),
        "tags": meta.get("tags", ""),
        "references": meta.get("references", ""),
        "text": " ".join(body.split()),
    }


def load_guidance(directory: str | Path) -> list[dict[str, Any]]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"knowledge directory not found: {directory}")
    docs = [
        parse_document(p.read_text(encoding="utf-8"), p.stem)
        for p in sorted(directory.glob("*.md"))
    ]
    if not docs:
        raise ValueError(f"no guidance documents in {directory}")
    return docs
