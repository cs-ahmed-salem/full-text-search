"""Conversion helpers between domain models and protobuf messages."""

from __future__ import annotations

from fulltext_search.algorithms.base import SearchResult
from fulltext_search.datasources.base import Document
from fulltext_search.interfaces.v1 import fulltext_search_pb2 as pb2


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
