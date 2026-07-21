"""gRPC servicer wiring domain algorithms to the FullTextSearch service."""

from __future__ import annotations

import grpc

from fulltext_search.algorithms.base import SearchAlgorithm
from fulltext_search.interfaces.converters import (
    document_from_pb,
    search_result_to_pb,
)
from fulltext_search.interfaces.v1 import fulltext_search_pb2 as pb2
from fulltext_search.interfaces.v1 import fulltext_search_pb2_grpc as pb2_grpc


class FullTextSearchServicer(pb2_grpc.FullTextSearchServicer):
    """Expose a :class:`SearchAlgorithm` over gRPC."""

    def __init__(self, algorithm: SearchAlgorithm) -> None:
        self._algorithm = algorithm

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
