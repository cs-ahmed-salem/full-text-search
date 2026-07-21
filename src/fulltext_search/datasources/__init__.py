"""Data source connectors for documents to index and search."""

from fulltext_search.datasources.base import DataSource, Document
from fulltext_search.datasources.local import LocalFileSource
from fulltext_search.datasources.postgres import PostgresTableSource
from fulltext_search.datasources.remote import RemoteEndpointSource

__all__ = [
    "DataSource",
    "Document",
    "LocalFileSource",
    "PostgresTableSource",
    "RemoteEndpointSource",
]
