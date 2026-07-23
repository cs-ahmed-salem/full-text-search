"""Data source connectors for documents to index and search."""

from fulltext_search.datasources.base import DataSource, Document
from fulltext_search.datasources.jsonl import JsonlFileSource
from fulltext_search.datasources.local import LocalFileSource
from fulltext_search.datasources.paging import (
    DataSourcePager,
    PageSource,
    as_page_source,
)
from fulltext_search.datasources.paging_api import PagingApiSource
from fulltext_search.datasources.postgres import PostgresTableSource
from fulltext_search.datasources.remote import RemoteEndpointSource

__all__ = [
    "DataSource",
    "DataSourcePager",
    "Document",
    "JsonlFileSource",
    "LocalFileSource",
    "PageSource",
    "PagingApiSource",
    "PostgresTableSource",
    "RemoteEndpointSource",
    "as_page_source",
]
