"""Pageable search result sessions backed by temp files."""

from fulltext_search.results.page import SearchPage
from fulltext_search.results.session_store import (
    SessionNotFoundError,
    SessionStore,
)

__all__ = [
    "SearchPage",
    "SessionNotFoundError",
    "SessionStore",
]
