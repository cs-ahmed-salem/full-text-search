"""Batching helpers for map-reduce style processing over documents."""

from __future__ import annotations

from itertools import islice
from typing import Iterable, Iterator, TypeVar

from fulltext_search.datasources.base import Document

T = TypeVar("T")


def batched(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive ``size``-length lists from ``iterable``.

    The final batch may be shorter. This lazily consumes ``iterable`` so it
    works for both in-memory sequences and streaming iterators (e.g. a
    ``DataSource.iter_documents`` stream).
    """

    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")

    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def iter_document_pages(
    documents: Iterable[Document],
    page_size: int,
) -> Iterator[list[Document]]:
    """Yield pages of documents for map-reduce retrieval.

    Accepts any iterable of :class:`Document` -- a materialized list, a
    ``dict.values()`` view, or a lazy datasource stream -- and pages it into
    lists of at most ``page_size`` documents.
    """

    return batched(documents, page_size)
