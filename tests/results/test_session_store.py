"""Tests for the temp-file backed search-result session store."""

from __future__ import annotations

import pytest

from fulltext_search.algorithms.base import SearchResult
from fulltext_search.datasources.base import Document
from fulltext_search.results.session_store import (
    SessionNotFoundError,
    SessionStore,
)


def _results(n: int) -> list[SearchResult]:
    return [
        SearchResult(
            document_id=str(i),
            score=1.0,
            document=Document(
                id=str(i),
                content=f"content {i}",
                metadata={"title": f"Doc {i}"},
            ),
            metadata={"relevance": "relevant", "rationale": "because"},
        )
        for i in range(n)
    ]


def test_create_and_get_first_page(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session_id = store.create(
        _results(5), query="q", rewritten_query="better q"
    )

    page = store.get_page(session_id, page=0, size=2)

    assert [r.document_id for r in page.content] == ["0", "1"]
    assert page.total_elements == 5
    assert page.total_pages == 3
    assert page.page == 0
    assert page.size == 2
    assert page.last is False
    assert page.rewritten_query == "better q"
    assert page.session_id == session_id


def test_last_page_flagged(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session_id = store.create(_results(5), query="q")

    page = store.get_page(session_id, page=2, size=2)

    assert [r.document_id for r in page.content] == ["4"]
    assert page.last is True


def test_page_preserves_document_and_metadata(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session_id = store.create(_results(1), query="q")

    page = store.get_page(session_id, page=0, size=10)
    result = page.content[0]

    assert result.document is not None
    assert result.document.metadata["title"] == "Doc 0"
    assert result.metadata["relevance"] == "relevant"


def test_empty_results(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session_id = store.create([], query="q")

    page = store.get_page(session_id, page=0, size=10)

    assert page.content == []
    assert page.total_elements == 0
    assert page.total_pages == 0
    assert page.last is True


def test_missing_session_raises(tmp_path) -> None:
    store = SessionStore(tmp_path)
    with pytest.raises(SessionNotFoundError):
        store.get_page("does-not-exist", page=0, size=10)


def test_paging_validation(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session_id = store.create(_results(1), query="q")
    with pytest.raises(ValueError):
        store.get_page(session_id, page=-1, size=10)
    with pytest.raises(ValueError):
        store.get_page(session_id, page=0, size=0)


def test_delete_removes_session(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session_id = store.create(_results(1), query="q")

    store.delete(session_id)

    with pytest.raises(SessionNotFoundError):
        store.get_page(session_id, page=0, size=10)
