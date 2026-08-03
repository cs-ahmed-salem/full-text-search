"""Unit tests for the gRPC interfaces subpackage."""

from __future__ import annotations

from concurrent import futures

import grpc
import pytest

from fulltext_search.algorithms.base import SearchResult
from fulltext_search.algorithms.corrective_rag import CorrectiveRAGAnswer
from fulltext_search.algorithms.stub import StubSearchAlgorithm
from fulltext_search.datasources.base import Document
from fulltext_search.interfaces.client import FullTextSearchClient
from fulltext_search.interfaces.servicer import FullTextSearchServicer
from fulltext_search.interfaces.v1 import fulltext_search_pb2_grpc as pb2_grpc
from fulltext_search.results.session_store import SessionStore


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


class FakeEngine:
    """Minimal engine exposing the pageable brute-force surface."""

    def __init__(self, store: SessionStore, answer: CorrectiveRAGAnswer) -> None:
        self.session_store = store
        self._answer = answer

    def search_all(self, query, *, page=0, size=20):
        session_id = self.session_store.create(
            self._answer.results,
            query=query,
            rewritten_query=self._answer.rewritten_query,
        )
        return self.session_store.get_page(session_id, page=page, size=size)

    def get_search_page(self, session_id, *, page=0, size=20):
        return self.session_store.get_page(session_id, page=page, size=size)


@pytest.fixture
def live_grpc_engine(tmp_path):
    results = [
        SearchResult(
            document_id=str(i),
            score=1.0,
            document=Document(id=str(i), content=f"content {i}"),
            metadata={"relevance": "relevant", "rationale": "because"},
        )
        for i in range(5)
    ]
    answer = CorrectiveRAGAnswer(
        results=results,
        records=[r.document for r in results],
        rewritten_query="better q",
    )
    engine = FakeEngine(SessionStore(tmp_path), answer)
    algorithm = StubSearchAlgorithm()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb2_grpc.add_FullTextSearchServicer_to_server(
        FullTextSearchServicer(algorithm, engine=engine),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield f"127.0.0.1:{port}"
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


def test_grpc_full_search_pagination(live_grpc_engine) -> None:
    target = live_grpc_engine

    with FullTextSearchClient(target) as client:
        first = client.create_full_search("anything", page=0, size=2)

        assert [r.document_id for r in first.content] == ["0", "1"]
        assert first.total_elements == 5
        assert first.total_pages == 3
        assert first.last is False
        assert first.rewritten_query == "better q"
        assert first.session_id

        last = client.get_search_page(first.session_id, page=2, size=2)
        assert [r.document_id for r in last.content] == ["4"]
        assert last.last is True
        assert last.content[0].document is not None
        assert last.content[0].metadata["relevance"] == "relevant"


def test_grpc_get_search_page_unknown_session(live_grpc_engine) -> None:
    target = live_grpc_engine

    with FullTextSearchClient(target) as client:
        with pytest.raises(grpc.RpcError) as excinfo:
            client.get_search_page("missing", page=0, size=2)
        assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


def test_grpc_full_search_unimplemented_without_engine(live_grpc) -> None:
    target, _algorithm = live_grpc

    with FullTextSearchClient(target) as client:
        with pytest.raises(grpc.RpcError) as excinfo:
            client.create_full_search("q")
        assert excinfo.value.code() == grpc.StatusCode.UNIMPLEMENTED
