"""gRPC servicer wiring domain algorithms to the FullTextSearch service."""

from __future__ import annotations

from typing import Any

import grpc

from fulltext_search.algorithms.base import SearchAlgorithm
from fulltext_search.interfaces.converters import (
    document_from_pb,
    search_page_to_pb,
    search_result_to_pb,
)
from fulltext_search.interfaces.v1 import fulltext_search_pb2 as pb2
from fulltext_search.interfaces.v1 import fulltext_search_pb2_grpc as pb2_grpc
from fulltext_search.results.session_store import SessionNotFoundError

DEFAULT_PAGE_SIZE = 20


class FullTextSearchServicer(pb2_grpc.FullTextSearchServicer):
    """Expose a :class:`SearchAlgorithm` over gRPC.

    Pass an ``engine`` (anything implementing ``search_all`` /
    ``get_search_page``, e.g. :class:`~fulltext_search.engine.SearchEngine`) to
    enable the pageable brute-force ``CreateFullSearch`` / ``GetSearchPage``
    RPCs. When omitted those RPCs return ``UNIMPLEMENTED``.
    """

    def __init__(
        self,
        algorithm: SearchAlgorithm,
        *,
        engine: Any | None = None,
    ) -> None:
        self._algorithm = algorithm
        self._engine = engine

    def Index(
        self,
        request: pb2.IndexRequest,
        context: grpc.ServicerContext,
    ) -> pb2.IndexResponse:
        documents = [document_from_pb(doc) for doc in request.documents]
        self._algorithm.index(documents)
        return pb2.IndexResponse(indexed_count=len(documents))

    def Search(
        self,
        request: pb2.SearchRequest,
        context: grpc.ServicerContext,
    ) -> pb2.SearchResponse:
        limit = request.limit if request.limit > 0 else 10
        results = self._algorithm.search(request.query, limit=limit)
        return pb2.SearchResponse(
            results=[search_result_to_pb(result) for result in results]
        )

    def CreateFullSearch(
        self,
        request: pb2.CreateFullSearchRequest,
        context: grpc.ServicerContext,
    ) -> pb2.PageableSearchResponse:
        if self._engine is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "CreateFullSearch requires an engine-backed servicer",
            )
        page = request.page if request.page > 0 else 0
        size = request.size if request.size > 0 else DEFAULT_PAGE_SIZE
        result_page = self._engine.search_all(
            request.query, page=page, size=size
        )
        return search_page_to_pb(result_page)

    def GetSearchPage(
        self,
        request: pb2.GetSearchPageRequest,
        context: grpc.ServicerContext,
    ) -> pb2.PageableSearchResponse:
        if self._engine is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "GetSearchPage requires an engine-backed servicer",
            )
        page = request.page if request.page > 0 else 0
        size = request.size if request.size > 0 else DEFAULT_PAGE_SIZE
        try:
            result_page = self._engine.get_search_page(
                request.session_id, page=page, size=size
            )
        except SessionNotFoundError:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Unknown search session: {request.session_id}",
            )
        return search_page_to_pb(result_page)

    def Clear(
        self,
        request: pb2.ClearRequest,
        context: grpc.ServicerContext,
    ) -> pb2.ClearResponse:
        try:
            self._algorithm.clear()
        except NotImplementedError as exc:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))
        return pb2.ClearResponse(cleared=True)
