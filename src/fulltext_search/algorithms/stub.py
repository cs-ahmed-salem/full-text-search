"""Temporary in-memory substring search used for demos and tests."""

from __future__ import annotations

from typing import Iterable

from fulltext_search.algorithms.base import SearchAlgorithm, SearchResult
from fulltext_search.datasources.base import Document


class StubSearchAlgorithm(SearchAlgorithm):
    """Minimal case-insensitive substring matcher.

    Replace with real algorithms under ``fulltext_search.algorithms`` later.
    """

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}

    def index(self, documents: Iterable[Document]) -> None:
        for document in documents:
            self._documents[document.id] = document

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        needle = query.casefold()
        hits: list[SearchResult] = []
        for document in self._documents.values():
            if needle in document.content.casefold():
                hits.append(
                    SearchResult(
                        document_id=document.id,
                        score=1.0,
                        document=document,
                    )
                )
            if len(hits) >= limit:
                break
        return hits

    def clear(self) -> None:
        self._documents.clear()
