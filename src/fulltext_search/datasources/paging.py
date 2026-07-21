"""Paging support for data sources.

The core :class:`~fulltext_search.datasources.base.DataSource` contract only
streams documents one at a time. For scalable, map-reduce style retrieval we
want to process the corpus in pages. This module defines an optional
``PageSource`` protocol plus a thin adapter that turns any ``DataSource`` (or
plain iterable of documents) into pages without changing existing sources.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Protocol, runtime_checkable

from fulltext_search.common.batching import iter_document_pages
from fulltext_search.datasources.base import DataSource, Document


@runtime_checkable
class PageSource(Protocol):
    """A source that can yield documents in pages.

    Sources that can page natively (e.g. an API exposing ``offset``/``limit``)
    may implement this directly for efficiency. Anything that only implements
    :meth:`~fulltext_search.datasources.base.DataSource.iter_documents` can be
    wrapped with :class:`DataSourcePager`.
    """

    def iter_pages(self, page_size: int) -> Iterator[list[Document]]:
        """Yield lists of at most ``page_size`` documents."""
        ...


class DataSourcePager:
    """Adapt a :class:`DataSource` or iterable of documents into a ``PageSource``.

    When wrapping a ``DataSource`` the pager manages ``connect``/``close`` via
    the context-manager protocol so callers can page a source without
    materializing it fully in memory.
    """

    def __init__(self, source: DataSource | Iterable[Document]) -> None:
        self._source = source

    def iter_pages(self, page_size: int) -> Iterator[list[Document]]:
        if isinstance(self._source, DataSource):
            with self._source as source:
                yield from iter_document_pages(
                    source.iter_documents(), page_size
                )
        else:
            yield from iter_document_pages(self._source, page_size)


def as_page_source(source: DataSource | Iterable[Document]) -> PageSource:
    """Return a ``PageSource`` for ``source``.

    Sources that natively expose ``iter_pages`` are returned as-is (their native
    paging is preserved); everything else is wrapped in a
    :class:`DataSourcePager`.
    """

    if isinstance(source, PageSource):
        return source
    return DataSourcePager(source)
