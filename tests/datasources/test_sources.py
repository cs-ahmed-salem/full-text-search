"""Unit tests for data sources."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from fulltext_search.datasources.base import DataSource, Document
from fulltext_search.datasources.local import LocalFileSource
from fulltext_search.datasources.paging import as_page_source
from fulltext_search.datasources.paging_api import PagingApiSource
from fulltext_search.datasources.postgres import PostgresTableSource, _quote_ident
from fulltext_search.datasources.remote import RemoteEndpointSource


def test_data_source_is_abstract() -> None:
    with pytest.raises(TypeError):
        DataSource()  # type: ignore[abstract]


def test_local_file_source_reads_directory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.txt").write_text("gamma", encoding="utf-8")

    with LocalFileSource(tmp_path, pattern="**/*.txt") as source:
        docs = list(source.iter_documents())

    assert {d.content for d in docs} == {"alpha", "beta", "gamma"}
    assert all(isinstance(d, Document) for d in docs)
    assert all("path" in d.metadata for d in docs)


def test_local_file_source_reads_single_file(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# hello", encoding="utf-8")

    with LocalFileSource(path) as source:
        docs = list(source.iter_documents())

    assert len(docs) == 1
    assert docs[0].content == "# hello"
    assert docs[0].id == str(path.resolve())


def test_local_file_source_requires_connect(tmp_path: Path) -> None:
    source = LocalFileSource(tmp_path)
    with pytest.raises(RuntimeError, match="not connected"):
        list(source.iter_documents())


def test_local_file_source_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "gone"
    with pytest.raises(FileNotFoundError):
        LocalFileSource(missing).connect()


def test_remote_endpoint_source_list_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.test/docs"
        return httpx.Response(
            200,
            json=[
                {"id": "1", "content": "one", "tag": "a"},
                {"id": "2", "content": "two"},
            ],
        )

    transport = httpx.MockTransport(handler)
    source = RemoteEndpointSource("https://example.test/docs")
    source._client = httpx.Client(transport=transport)

    try:
        docs = list(source.iter_documents())
    finally:
        source.close()

    assert docs[0] == Document(
        id="1",
        content="one",
        metadata={"tag": "a", "source_url": "https://example.test/docs"},
    )
    assert docs[1].id == "2"


def test_remote_endpoint_source_wrapped_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"documents": [{"id": "x", "content": "wrapped"}]},
        )

    source = RemoteEndpointSource("https://example.test/docs")
    source._client = httpx.Client(transport=httpx.MockTransport(handler))

    try:
        docs = list(source.iter_documents())
    finally:
        source.close()

    assert docs == [
        Document(
            id="x",
            content="wrapped",
            metadata={"source_url": "https://example.test/docs"},
        )
    ]


def test_remote_endpoint_source_requires_connect() -> None:
    source = RemoteEndpointSource("https://example.test/docs")
    with pytest.raises(RuntimeError, match="not connected"):
        list(source.iter_documents())


def test_paging_api_source_walks_pages() -> None:
    seen_pages: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-internal-pass") == "secret"
        assert request.url.params.get("filter") == "ready"
        page = int(request.url.params["page"])
        size = int(request.url.params["size"])
        seen_pages.append(page)
        assert size == 2

        pages = {
            0: {
                "content": [
                    {"id": "1", "description": "one", "title": "t1"},
                    {"id": "2", "description": "two", "title": "t2"},
                ],
                "last": False,
            },
            1: {
                "content": [
                    {"id": "3", "description": "three", "title": "t3"},
                ],
                "last": True,
            },
        }
        return httpx.Response(200, json=pages[page])

    source = PagingApiSource(
        "https://example.test/tasks",
        headers={"x-internal-pass": "secret"},
        params={"filter": "ready"},
        page_size=2,
    )
    source._client = httpx.Client(transport=httpx.MockTransport(handler))

    try:
        docs = list(source.iter_documents())
    finally:
        source.close()

    assert seen_pages == [0, 1]
    assert [d.id for d in docs] == ["1", "2", "3"]
    assert docs[0] == Document(
        id="1",
        content="one",
        metadata={
            "title": "t1",
            "source_url": "https://example.test/tasks",
        },
    )


def test_paging_api_source_iter_pages_is_native() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        size = int(request.url.params["size"])
        assert size == 1
        pages = {
            0: {"content": [{"id": "a", "description": "alpha"}], "last": False},
            1: {"content": [{"id": "b", "description": "beta"}], "last": True},
        }
        return httpx.Response(200, json=pages[page])

    source = PagingApiSource("https://example.test/tasks", page_size=10)
    source._client = httpx.Client(transport=httpx.MockTransport(handler))

    try:
        pages = list(source.iter_pages(1))
    finally:
        source.close()

    assert len(pages) == 2
    assert pages[0][0].id == "a"
    assert pages[1][0].content == "beta"
    assert as_page_source(source) is source


def test_paging_api_source_stops_on_empty_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        if page == 0:
            return httpx.Response(
                200,
                json={
                    "content": [{"id": "1", "description": "only"}],
                    "last": False,
                },
            )
        return httpx.Response(200, json={"content": [], "last": True})

    source = PagingApiSource("https://example.test/tasks", page_size=1)
    source._client = httpx.Client(transport=httpx.MockTransport(handler))

    try:
        docs = list(source.iter_documents())
    finally:
        source.close()

    assert [d.id for d in docs] == ["1"]


def test_paging_api_source_requires_connect() -> None:
    source = PagingApiSource("https://example.test/tasks")
    with pytest.raises(RuntimeError, match="not connected"):
        list(source.iter_documents())


def test_paging_api_source_rejects_invalid_page_size() -> None:
    with pytest.raises(ValueError, match="page_size"):
        PagingApiSource("https://example.test/tasks", page_size=0)


def test_postgres_table_source_requires_table_or_query() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PostgresTableSource("postgresql://localhost/db")

    with pytest.raises(ValueError, match="exactly one"):
        PostgresTableSource(
            "postgresql://localhost/db",
            table="docs",
            query="SELECT 1",
        )


def test_postgres_table_source_iter_documents() -> None:
    rows = [
        {"id": 1, "content": "hello", "author": "a"},
        {"id": 2, "content": "world", "author": "b"},
    ]
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__iter__.return_value = iter(rows)

    conn = MagicMock()
    conn.cursor.return_value = cursor

    source = PostgresTableSource(
        "postgresql://localhost/db",
        table="documents",
        schema="public",
    )
    source._conn = conn

    docs = list(source.iter_documents())

    assert docs[0] == Document(
        id="1",
        content="hello",
        metadata={"author": "a", "table": "documents", "schema": "public"},
    )
    assert docs[1].content == "world"
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args.args[0]
    assert '"public"."documents"' in sql


def test_postgres_custom_query() -> None:
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__iter__.return_value = iter(
        [{"doc_id": "a", "body": "text", "extra": 1}]
    )

    conn = MagicMock()
    conn.cursor.return_value = cursor

    source = PostgresTableSource(
        "postgresql://localhost/db",
        query="SELECT doc_id, body, extra FROM custom WHERE active = %(active)s",
        id_column="doc_id",
        content_column="body",
        query_params={"active": True},
    )
    source._conn = conn

    docs = list(source.iter_documents())

    assert docs == [
        Document(id="a", content="text", metadata={"extra": 1}),
    ]
    cursor.execute.assert_called_once_with(
        "SELECT doc_id, body, extra FROM custom WHERE active = %(active)s",
        {"active": True},
    )


def test_quote_ident_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        _quote_ident('bad";drop')
