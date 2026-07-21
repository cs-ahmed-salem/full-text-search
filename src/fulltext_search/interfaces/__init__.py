"""gRPC interface for full-text search."""

from fulltext_search.interfaces.client import FullTextSearchClient
from fulltext_search.interfaces.server import create_server, serve
from fulltext_search.interfaces.servicer import FullTextSearchServicer

__all__ = [
    "FullTextSearchClient",
    "FullTextSearchServicer",
    "create_server",
    "serve",
]
