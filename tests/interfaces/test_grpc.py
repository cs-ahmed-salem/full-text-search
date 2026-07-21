"""Unit tests for the gRPC interfaces subpackage."""

from __future__ import annotations

from concurrent import futures

import grpc
import pytest

from fulltext_search.algorithms.stub import StubSearchAlgorithm
from fulltext_search.datasources.base import Document
from fulltext_search.interfaces.client import FullTextSearchClient
from fulltext_search.interfaces.servicer import FullTextSearchServicer
from fulltext_search.interfaces.v1 import fulltext_search_pb2_grpc as pb2_grpc


@pytest.fixture
def live_grpc():
    algorithm = StubSearchAlgorithm()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb2_grpc.add_FullTextSearchServicer_to_server(
        FullTextSearchServicer(algorithm),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield f"127.0.0.1:{port}", algorithm
    finally:
        server.stop(grace=None)


def test_grpc_index_search_clear(live_grpc) -> None:
    target, _algorithm = live_grpc

    with FullTextSearchClient(target) as client:
        indexed = client.index(
            [
                Document(id="1", content="hello world", metadata={"k": "v"}),
                Document(id="2", content="goodbye"),
            ]
        )
        assert indexed == 2

        results = client.search("hello")
        assert len(results) == 1
        assert results[0].document_id == "1"
        assert results[0].document is not None
        assert results[0].document.metadata["k"] == "v"

        assert client.clear() is True
        assert client.search("hello") == []


def test_grpc_search_respects_limit(live_grpc) -> None:
    target, _algorithm = live_grpc

    with FullTextSearchClient(target) as client:
        client.index(
            Document(id=str(i), content=f"shared {i}") for i in range(5)
        )
        results = client.search("shared", limit=2)
        assert len(results) == 2
