"""Spring Data-style page envelope for search results."""

from __future__ import annotations

from dataclasses import dataclass, field

from fulltext_search.algorithms.base import SearchResult


@dataclass(frozen=True, slots=True)
class SearchPage:
    """A single page of ranked search results.

    Mirrors Spring Data's ``Page`` shape so clients can paginate a completed
    brute-force search without re-running LLM grading.
    """

    content: list[SearchResult] = field(default_factory=list)
    page: int = 0
    size: int = 20
    total_elements: int = 0
    total_pages: int = 0
    last: bool = True
    session_id: str = ""
    rewritten_query: str | None = None
