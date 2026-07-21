"""Performance tests for indexing and search throughput."""

from __future__ import annotations

import pytest

from fulltext_search.datasources.base import Document
from fulltext_search.datasources.local import LocalFileSource
from tests.helpers import StubSearchAlgorithm

pytestmark = pytest.mark.performance


def _make_corpus(size: int) -> list[Document]:
    return [
        Document(
            id=str(i),
            content=f"document {i} contains searchable token-{i % 50} and filler text",
        )
        for i in range(size)
    ]


def test_index_throughput(benchmark) -> None:
    documents = _make_corpus(2_000)

    def run() -> int:
        algo = StubSearchAlgorithm()
        algo.index(documents)
        return len(documents)

    count = benchmark(run)
    assert count == 2_000


def test_search_throughput(benchmark) -> None:
    algo = StubSearchAlgorithm()
    algo.index(_make_corpus(5_000))

    def run() -> int:
        return len(algo.search("token-7", limit=20))

    hits = benchmark(run)
    assert hits > 0


def test_local_source_read_throughput(benchmark, tmp_path) -> None:
    for i in range(200):
        (tmp_path / f"doc-{i}.txt").write_text(
            f"file {i} content token-{i % 10}",
            encoding="utf-8",
        )

    def run() -> int:
        with LocalFileSource(tmp_path, pattern="*.txt") as source:
            return sum(1 for _ in source.iter_documents())

    count = benchmark(run)
    assert count == 200
