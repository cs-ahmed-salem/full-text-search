"""gRPC client for the FullTextSearch service."""

from __future__ import annotations

from types import TracebackType
from typing import Iterable

import grpc

from fulltext_search.algorithms.base import SearchResult
from fulltext_search.datasources.base import Document
from fulltext_search.interfaces.converters import (
    document_from_pb,
    document_to_pb,
    search_page_from_pb,
)
from fulltext_search.interfaces.v1 import fulltext_search_pb2 as pb2
from fulltext_search.interfaces.v1 import fulltext_search_pb2_grpc as pb2_grpc
from fulltext_search.results.page import SearchPage


class FullTextSearchClient:
    """Thin client around the FullTextSearch gRPC stub."""

    def __init__(
        self,
        target: str = "localhost:50051",
        *,
        channel: grpc.Channel | None = None,
    ) -> None:
        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(target)
        self._stub = pb2_grpc.FullTextSearchStub(self._channel)

    def index(self, documents: Iterable[Document]) -> int:
        request = pb2.IndexRequest(
            documents=[document_to_pb(document) for document in documents]
        )
        response = self._stub.Index(request)
        return int(response.indexed_count)

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        response = self._stub.Search(
            pb2.SearchRequest(query=query, limit=limit)
        )
        results: list[SearchResult] = []
        for item in response.results:
            document = None
            if item.HasField("document"):
                document = document_from_pb(item.document)
            results.append(
                SearchResult(
                    document_id=item.document_id,
                    score=item.score,
                    document=document,
                    highlights=tuple(item.highlights),
                    metadata=dict(item.metadata),
                )
            )
        return results

    def create_full_search(
        self,
        query: str,
        *,
        page: int = 0,
        size: int = 20,
    ) -> SearchPage:
        """Run a brute-force full-corpus search and return the first page."""

        response = self._stub.CreateFullSearch(
            pb2.CreateFullSearchRequest(query=query, page=page, size=size)
        )
        return search_page_from_pb(response)

    def get_search_page(
        self,
        session_id: str,
        *,
        page: int = 0,
        size: int = 20,
    ) -> SearchPage:
        """Fetch a page from an existing brute-force search session."""

        response = self._stub.GetSearchPage(
            pb2.GetSearchPageRequest(
                session_id=session_id, page=page, size=size
            )
        )
        return search_page_from_pb(response)

    def clear(self) -> bool:
        response = self._stub.Clear(pb2.ClearRequest())
        return bool(response.cleared)

    def close(self) -> None:
        if self._owns_channel:
            self._channel.close()

    def __enter__(self) -> FullTextSearchClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
