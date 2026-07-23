"""Full-text search over arbitrary data sources."""

from fulltext_search.algorithms.base import SearchAlgorithm, SearchResult
from fulltext_search.algorithms.corrective_rag import (
    CorrectiveRAGAlgorithm,
    CorrectiveRAGAnswer,
)
from fulltext_search.datasources.base import DataSource, Document
from fulltext_search.datasources.paging_api import PagingApiSource
from fulltext_search.engine import SearchEngine, enrich_task_document

__all__ = [
    "CorrectiveRAGAlgorithm",
    "CorrectiveRAGAnswer",
    "DataSource",
    "Document",
    "PagingApiSource",
    "SearchAlgorithm",
    "SearchEngine",
    "SearchResult",
    "enrich_task_document",
]

__version__ = "0.1.0"
