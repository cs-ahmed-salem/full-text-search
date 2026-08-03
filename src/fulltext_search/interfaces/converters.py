"""Conversion helpers between domain models and protobuf messages."""

from __future__ import annotations

from fulltext_search.algorithms.base import SearchResult
from fulltext_search.datasources.base import Document
from fulltext_search.interfaces.v1 import fulltext_search_pb2 as pb2
from fulltext_search.results.page import SearchPage


def document_to_pb(document: Document) -> pb2.Document:
    return pb2.Document(
        id=document.id,
        content=document.content,
        metadata={str(k): str(v) for k, v in document.metadata.items()},
    )


def document_from_pb(message: pb2.Document) -> Document:
    return Document(
        id=message.id,
        content=message.content,
        metadata=dict(message.metadata),
    )


def search_result_to_pb(result: SearchResult) -> pb2.SearchResult:
    message = pb2.SearchResult(
        document_id=result.document_id,
        score=result.score,
        highlights=list(result.highlights),
        metadata={str(k): str(v) for k, v in result.metadata.items()},
    )
    if result.document is not None:
        message.document.CopyFrom(document_to_pb(result.document))
    return message


def search_result_from_pb(message: pb2.SearchResult) -> SearchResult:
    document = None
    if message.HasField("document"):
        document = document_from_pb(message.document)
    return SearchResult(
        document_id=message.document_id,
        score=message.score,
        document=document,
        highlights=tuple(message.highlights),
        metadata=dict(message.metadata),
    )


def search_page_to_pb(page: SearchPage) -> pb2.PageableSearchResponse:
    return pb2.PageableSearchResponse(
        content=[search_result_to_pb(result) for result in page.content],
        session_id=page.session_id,
        page=page.page,
        size=page.size,
        total_elements=page.total_elements,
        total_pages=page.total_pages,
        last=page.last,
        rewritten_query=page.rewritten_query or "",
    )


def search_page_from_pb(message: pb2.PageableSearchResponse) -> SearchPage:
    return SearchPage(
        content=[
            search_result_from_pb(item) for item in message.content
        ],
        page=message.page,
        size=message.size,
        total_elements=message.total_elements,
        total_pages=message.total_pages,
        last=message.last,
        session_id=message.session_id,
        rewritten_query=message.rewritten_query or None,
    )
