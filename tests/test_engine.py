"""Unit tests for the SearchEngine facade."""

from __future__ import annotations

from typing import Iterable
from unittest.mock import MagicMock

import httpx
import pytest

from fulltext_search.algorithms.base import SearchAlgorithm, SearchResult
from fulltext_search.algorithms.corrective_rag import CorrectiveRAGAnswer
from fulltext_search.datasources.base import DataSource, Document
from fulltext_search.datasources.paging_api import PagingApiSource
from fulltext_search.engine import SearchEngine, enrich_task_document


class _FakeAlgorithm(SearchAlgorithm):
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}

    def index(self, documents: Iterable[Document]) -> None:
        for document in documents:
            self.documents[document.id] = document

    def clear(self) -> None:
        self.documents.clear()

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        hits = [
            SearchResult(document_id=doc.id, score=1.0, document=doc)
            for doc in self.documents.values()
            if query.lower() in doc.content.lower()
        ]
        return hits[:limit]

    def answer(self, query: str, *, limit: int = 10) -> CorrectiveRAGAnswer:
        return CorrectiveRAGAnswer(
            results=self.search(query, limit=limit),
        )


class _ListSource(DataSource):
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def iter_documents(self) -> Iterable[Document]:
        if not self._connected:
            raise RuntimeError("not connected")
        yield from self._documents


def test_enrich_task_document() -> None:
    doc = Document(
        id="1",
        content="Do the thing",
        metadata={
            "title": "Bundesbank Files",
            "code": "AC378",
            "currentState": "READY",
            "preconditions": ["A", "B"],
        },
    )
    enriched = enrich_task_document(doc)
    assert "Code: AC378" in enriched.content
    assert "Title: Bundesbank Files" in enriched.content
    assert "Preconditions: A; B" in enriched.content


def test_search_engine_indexes_and_asks() -> None:
    source = _ListSource(
        [
            Document(
                id="1",
                content="wire funds",
                metadata={"title": "Request wires", "code": "T03"},
            )
        ]
    )
    engine = SearchEngine(
        source,
        algorithm=_FakeAlgorithm(),
        document_transform=enrich_task_document,
    )

    assert engine.document_count == 1
    indexed = next(iter(engine.algorithm.documents.values()))  # type: ignore[attr-defined]
    assert "Title: Request wires" in indexed.content

    result = engine.ask("wires")
    assert [r.document_id for r in result.results] == ["1"]
    assert result.results[0].document_id == "1"


def test_search_engine_from_paging_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-internal-pass") == "secret"
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "id": "1",
                        "description": "desc",
                        "title": "Task",
                        "code": "T1",
                    }
                ],
                "last": True,
            },
        )

    monkeypatch.setattr(
        "fulltext_search.engine.ensure_llm_ready",
        lambda **_: "openai/test",
    )
    monkeypatch.setattr(
        "fulltext_search.engine.configure_default_lm",
        lambda model: MagicMock(model=model),
    )

    # Build source with mock transport via from_paging_api path pieces.
    source = PagingApiSource(
        "https://example.test/tasks",
        headers={"x-internal-pass": "secret"},
        page_size=10,
    )
    source._client = httpx.Client(transport=httpx.MockTransport(handler))

    engine = SearchEngine(source, algorithm=_FakeAlgorithm())
    assert engine.document_count == 1
    assert engine.search("Task")[0].document_id == "1"


def test_from_env_requires_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FULLTEXT_SEARCH_API_URL", raising=False)
    monkeypatch.setattr(
        "fulltext_search.engine.ensure_llm_ready",
        lambda **_: "openai/test",
    )
    with pytest.raises(ValueError, match="FULLTEXT_SEARCH_API_URL"):
        SearchEngine.from_env(algorithm=_FakeAlgorithm())


def test_ask_requires_answer_method() -> None:
    class SearchOnly(SearchAlgorithm):
        def index(self, documents: Iterable[Document]) -> None:
            return None

        def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
            return []

    engine = SearchEngine(
        _ListSource([]),
        algorithm=SearchOnly(),
        document_transform=None,
    )
    with pytest.raises(TypeError, match="does not implement answer"):
        engine.ask("q")
