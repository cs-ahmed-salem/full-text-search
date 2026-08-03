"""Temp-file backed store for pageable search-result sessions.

A completed brute-force Corrective RAG search can return an arbitrary number of
relevant records. Rather than hold them all in memory or re-run the (expensive)
LLM grading on every page request, the ranked results are written once to a
JSONL file plus a small metadata sidecar. Subsequent page requests simply slice
the persisted file.

Layout (under the store directory)::

    {session_id}.jsonl       one ranked SearchResult per line
    {session_id}.meta.json   query, rewritten_query, total_elements, created_at
"""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from pathlib import Path

from fulltext_search.algorithms.base import SearchResult
from fulltext_search.datasources.base import Document
from fulltext_search.results.page import SearchPage

DEFAULT_DIR_NAME = "fulltext_search_sessions"


class SessionNotFoundError(KeyError):
    """Raised when a session id has no persisted results."""


def _default_directory() -> Path:
    return Path(tempfile.gettempdir()) / DEFAULT_DIR_NAME


def _document_to_json(document: Document) -> dict:
    return {
        "id": document.id,
        "content": document.content,
        "metadata": dict(document.metadata),
    }


def _document_from_json(data: dict | None) -> Document | None:
    if data is None:
        return None
    return Document(
        id=str(data.get("id", "")),
        content=str(data.get("content", "")),
        metadata=dict(data.get("metadata") or {}),
    )


def _result_to_json(result: SearchResult) -> dict:
    return {
        "document_id": result.document_id,
        "score": result.score,
        "document": (
            _document_to_json(result.document)
            if result.document is not None
            else None
        ),
        "highlights": list(result.highlights),
        "metadata": dict(result.metadata),
    }


def _result_from_json(data: dict) -> SearchResult:
    return SearchResult(
        document_id=str(data.get("document_id", "")),
        score=float(data.get("score", 0.0)),
        document=_document_from_json(data.get("document")),
        highlights=tuple(data.get("highlights") or ()),
        metadata=dict(data.get("metadata") or {}),
    )


class SessionStore:
    """Persist ranked search results and serve them page by page."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory else _default_directory()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise ValueError(
                f"SessionStore directory must be a real directory, got {self.directory}"
            )

    def _data_path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.jsonl"

    def _meta_path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.meta.json"

    def create(
        self,
        results: list[SearchResult],
        *,
        query: str,
        rewritten_query: str | None = None,
    ) -> str:
        """Persist ``results`` and return the new session id."""

        session_id = uuid.uuid4().hex
        data_path = self._data_path(session_id)
        with data_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(_result_to_json(result)))
                handle.write("\n")

        meta = {
            "query": query,
            "rewritten_query": rewritten_query,
            "total_elements": len(results),
            "created_at": time.time(),
        }
        self._meta_path(session_id).write_text(
            json.dumps(meta), encoding="utf-8"
        )
        return session_id

    def _read_meta(self, session_id: str) -> dict:
        meta_path = self._meta_path(session_id)
        if not meta_path.exists():
            raise SessionNotFoundError(session_id)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def get_page(
        self,
        session_id: str,
        *,
        page: int = 0,
        size: int = 20,
    ) -> SearchPage:
        """Return the ``page``-th slice of ``size`` results for a session."""

        if page < 0:
            raise ValueError(f"page must be >= 0, got {page}")
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")

        meta = self._read_meta(session_id)
        data_path = self._data_path(session_id)
        if not data_path.exists():
            raise SessionNotFoundError(session_id)

        total_elements = int(meta.get("total_elements", 0))
        total_pages = (total_elements + size - 1) // size if size else 0
        start = page * size
        end = start + size

        content: list[SearchResult] = []
        with data_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index < start:
                    continue
                if index >= end:
                    break
                line = line.strip()
                if line:
                    content.append(_result_from_json(json.loads(line)))

        last = end >= total_elements
        return SearchPage(
            content=content,
            page=page,
            size=size,
            total_elements=total_elements,
            total_pages=total_pages,
            last=last,
            session_id=session_id,
            rewritten_query=meta.get("rewritten_query"),
        )

    def delete(self, session_id: str) -> None:
        """Remove a session's files if present."""

        self._data_path(session_id).unlink(missing_ok=True)
        self._meta_path(session_id).unlink(missing_ok=True)
