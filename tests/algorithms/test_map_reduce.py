"""Tests for batching and map-reduce lexical retrieval (no LLM)."""

from __future__ import annotations

import pytest

from fulltext_search.algorithms.map_reduce import (
    default_lexical_score,
    map_reduce_retrieve,
    map_reduce_score_all,
)
from fulltext_search.common.batching import batched, iter_document_pages
from fulltext_search.datasources.base import Document


def _docs(*contents: str) -> list[Document]:
    return [
        Document(id=str(i), content=content)
        for i, content in enumerate(contents)
    ]


def test_batched_yields_chunks_and_remainder() -> None:
    assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]


def test_batched_rejects_bad_size() -> None:
    with pytest.raises(ValueError):
        list(batched(range(3), 0))


def test_iter_document_pages_streams_lazily() -> None:
    def gen():
        for i in range(3):
            yield Document(id=str(i), content=f"doc {i}")

    pages = list(iter_document_pages(gen(), 2))

    assert [len(page) for page in pages] == [2, 1]


def test_default_lexical_score_rewards_coverage() -> None:
    query = "quick brown fox"
    high = Document(id="a", content="the quick brown fox jumps")
    low = Document(id="b", content="a quick meal")
    none = Document(id="c", content="totally unrelated text")

    high_score = default_lexical_score(query, high)
    low_score = default_lexical_score(query, low)
    assert high_score > low_score
    assert default_lexical_score(query, none) == 0.0


def test_default_lexical_score_empty_query() -> None:
    assert default_lexical_score("", Document(id="a", content="text")) == 0.0


def test_map_reduce_matches_serial_top_k() -> None:
    documents = _docs(
        "alpha beta gamma",
        "alpha beta",
        "alpha",
        "delta epsilon",
        "beta gamma alpha extra",
    )
    pages = list(iter_document_pages(documents, 2))

    # Parallel result should equal a serial ranking by score, id.
    parallel = map_reduce_retrieve(
        pages, "alpha beta", limit=3, max_workers=4
    )

    serial_scores = sorted(
        (
            (-default_lexical_score("alpha beta", d), d.id)
            for d in documents
            if default_lexical_score("alpha beta", d) > 0
        )
    )
    expected_ids = [doc_id for _, doc_id in serial_scores][:3]

    assert [c.document_id for c in parallel] == expected_ids


def test_map_reduce_respects_limit() -> None:
    documents = _docs(*[f"shared term {i}" for i in range(10)])
    pages = list(iter_document_pages(documents, 3))

    results = map_reduce_retrieve(pages, "shared", limit=4)

    assert len(results) == 4


def test_map_reduce_dedupes_document_ids() -> None:
    # Same id appearing across pages must collapse to a single candidate.
    duplicate = Document(id="dup", content="alpha beta")
    pages = [[duplicate], [duplicate], [Document(id="x", content="alpha")]]

    results = map_reduce_retrieve(pages, "alpha", limit=10)

    ids = [c.document_id for c in results]
    assert ids.count("dup") == 1
    assert set(ids) == {"dup", "x"}


def test_map_reduce_single_worker_equivalent() -> None:
    documents = _docs("alpha beta", "alpha", "beta gamma", "alpha beta gamma")
    pages = list(iter_document_pages(documents, 2))

    one = map_reduce_retrieve(
        list(pages), "alpha beta", limit=5, max_workers=1
    )
    many = map_reduce_retrieve(
        list(iter_document_pages(documents, 2)),
        "alpha beta",
        limit=5,
        max_workers=4,
    )

    assert [c.document_id for c in one] == [c.document_id for c in many]


def test_map_reduce_validates_arguments() -> None:
    with pytest.raises(ValueError):
        map_reduce_retrieve([], "q", limit=0)
    with pytest.raises(ValueError):
        map_reduce_retrieve([], "q", max_workers=0)
    with pytest.raises(ValueError):
        map_reduce_retrieve([], "q", recall_per_page=0)


def test_score_all_includes_zero_score_docs() -> None:
    documents = _docs("alpha beta", "gamma only", "delta")
    pages = list(iter_document_pages(documents, 2))

    results = map_reduce_score_all(pages, "alpha")

    # Every document is returned, even those with no lexical match.
    assert {c.document_id for c in results} == {"0", "1", "2"}
    scores = {c.document_id: c.score for c in results}
    assert scores["0"] > 0.0
    assert scores["1"] == 0.0
    assert scores["2"] == 0.0


def test_score_all_sorted_and_deduped() -> None:
    duplicate = Document(id="dup", content="alpha alpha")
    pages = [
        [duplicate, Document(id="x", content="alpha")],
        [duplicate, Document(id="y", content="nothing")],
    ]

    results = map_reduce_score_all(pages, "alpha")

    ids = [c.document_id for c in results]
    assert ids.count("dup") == 1
    # Highest score first, ties broken by id; zero-score doc last.
    assert ids[0] == "dup"
    assert ids[-1] == "y"


def test_score_all_validates_workers() -> None:
    with pytest.raises(ValueError):
        map_reduce_score_all([], "q", max_workers=0)


def test_score_all_single_worker_equivalent() -> None:
    documents = _docs("alpha beta", "alpha", "beta gamma", "zzz")

    one = map_reduce_score_all(
        list(iter_document_pages(documents, 2)), "alpha beta", max_workers=1
    )
    many = map_reduce_score_all(
        list(iter_document_pages(documents, 2)), "alpha beta", max_workers=4
    )

    assert [c.document_id for c in one] == [c.document_id for c in many]
