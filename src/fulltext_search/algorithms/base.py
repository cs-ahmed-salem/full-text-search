"""Base types for search algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from fulltext_search.datasources.base import Document


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single hit returned by a search algorithm."""

    document_id: str
    score: float
    document: Document | None = None
    highlights: Sequence[str] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchAlgorithm(ABC):
    """Abstract base class for full-text search algorithms.

    Implementations index documents and answer queries. Concrete algorithms
    (BM25, inverted index, etc.) will subclass this later.
    """

    @abstractmethod
    def index(self, documents: Iterable[Document]) -> None:
        """Add or update documents in the algorithm's index."""

    @abstractmethod
    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Return ranked results for ``query``."""

    def clear(self) -> None:
        """Remove all indexed documents. Optional override."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement clear()"
        )
