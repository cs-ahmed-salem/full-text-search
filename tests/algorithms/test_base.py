"""Unit tests for search algorithm contracts."""

from __future__ import annotations

import pytest

from fulltext_search.algorithms.base import SearchAlgorithm, SearchResult
from fulltext_search.datasources.base import Document
from tests.helpers import StubSearchAlgorithm


def test_search_algorithm_is_abstract() -> None:
    with pytest.raises(TypeError):
        SearchAlgorithm()  # type: ignore[abstract]


def test_stub_algorithm_indexes_and_searches() -> None:
    algo = StubSearchAlgorithm()
    algo.index(
        [
            Document(id="1", content="alpha beta"),
            Document(id="2", content="gamma delta"),
        ]
    )

    results = algo.search("beta")

    assert len(results) == 1
    assert results[0] == SearchResult(
        document_id="1",
        score=1.0,
        document=Document(id="1", content="alpha beta"),
    )


def test_stub_algorithm_respects_limit() -> None:
    algo = StubSearchAlgorithm()
    algo.index(
        Document(id=str(i), content=f"shared term {i}") for i in range(5)
    )

    results = algo.search("shared", limit=2)

    assert len(results) == 2


def test_stub_algorithm_clear() -> None:
    algo = StubSearchAlgorithm()
    algo.index([Document(id="1", content="hello")])
    algo.clear()

    assert algo.search("hello") == []


def test_base_clear_default_raises() -> None:
    class IncompleteAlgorithm(SearchAlgorithm):
        def index(self, documents):  # type: ignore[no-untyped-def]
            return None

        def search(self, query, *, limit=10):  # type: ignore[no-untyped-def]
            return []

    with pytest.raises(NotImplementedError):
        IncompleteAlgorithm().clear()
