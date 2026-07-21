"""Full-text search over arbitrary data sources."""

from fulltext_search.algorithms.base import SearchAlgorithm, SearchResult
from fulltext_search.datasources.base import DataSource, Document

__all__ = [
    "DataSource",
    "Document",
    "SearchAlgorithm",
    "SearchResult",
]

__version__ = "0.1.0"
