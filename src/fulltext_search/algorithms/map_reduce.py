"""Threaded map-reduce lexical retrieval.

The ``map`` step scores each page of documents independently (in parallel) and
keeps a per-page shortlist. The ``reduce`` step merges shortlists, deduping by
``document_id`` and keeping the global top-k. Pages are consumed lazily with a
bounded number of in-flight tasks so a huge, streaming corpus never has to be
materialized all at once.

The lexical scorer is intentionally LLM-free and cheap so map-reduce stays
inexpensive; the expensive LLM grading/generation runs only on the small
reduced candidate set.
"""

from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Iterable

from fulltext_search.datasources.base import Document

Scorer = Callable[[str, Document], float]

_TOKEN_RE = re.compile(r"[0-9a-z]+")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A scored document produced by the retrieval map-reduce."""

    document_id: str
    score: float
    document: Document


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def default_lexical_score(query: str, document: Document) -> float:
    """Score a document against ``query`` by term coverage and density.

    Returns ``0.0`` when no query term appears. Otherwise the score is the
    fraction of distinct query terms present plus a small density bonus, so
    documents matching more of the query rank first and, among those, denser
    matches rank higher.
    """

    query_terms = set(_tokenize(query))
    if not query_terms:
        return 0.0

    content_tokens = _tokenize(document.content)
    if not content_tokens:
        return 0.0

    counts = Counter(content_tokens)
    matched = [term for term in query_terms if counts[term] > 0]
    if not matched:
        return 0.0

    coverage = len(matched) / len(query_terms)
    total_hits = sum(counts[term] for term in matched)
    density = total_hits / len(content_tokens)
    return coverage + 0.1 * density


def _score_page(
    page: list[Document],
    query: str,
    scorer: Scorer,
    recall_per_page: int,
) -> list[Candidate]:
    scored: list[Candidate] = []
    for document in page:
        score = scorer(query, document)
        if score > 0:
            scored.append(Candidate(document.id, score, document))

    scored.sort(key=lambda c: (-c.score, c.document_id))
    return scored[:recall_per_page]


def map_reduce_retrieve(
    pages: Iterable[list[Document]],
    query: str,
    *,
    scorer: Scorer = default_lexical_score,
    limit: int = 10,
    recall_per_page: int | None = None,
    max_workers: int = 4,
) -> list[Candidate]:
    """Retrieve the top ``limit`` candidates across ``pages`` via map-reduce.

    Pages are scored in parallel threads; at most ``max_workers`` pages are in
    flight at once so lazy/streaming page iterators are consumed incrementally.
    Results are deduped by ``document_id`` (keeping the highest score) and
    returned sorted by score descending, ``document_id`` ascending for stable
    ordering.
    """

    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    effective_recall = (
        recall_per_page if recall_per_page is not None else limit
    )
    if effective_recall < 1:
        raise ValueError(
            f"recall_per_page must be >= 1, got {recall_per_page}"
        )

    best: dict[str, Candidate] = {}

    def _merge(candidates: list[Candidate]) -> None:
        for candidate in candidates:
            current = best.get(candidate.document_id)
            if current is None or candidate.score > current.score:
                best[candidate.document_id] = candidate

    page_iter = iter(pages)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight = set()
        for _ in range(max_workers):
            page = next(page_iter, None)
            if page is None:
                break
            in_flight.add(
                executor.submit(
                    _score_page, page, query, scorer, effective_recall
                )
            )

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                _merge(future.result())
                page = next(page_iter, None)
                if page is not None:
                    in_flight.add(
                        executor.submit(
                            _score_page, page, query, scorer, effective_recall
                        )
                    )

    merged = sorted(best.values(), key=lambda c: (-c.score, c.document_id))
    return merged[:limit]


def _score_page_all(
    page: list[Document],
    query: str,
    scorer: Scorer,
) -> list[Candidate]:
    return [
        Candidate(document.id, scorer(query, document), document)
        for document in page
    ]


def map_reduce_score_all(
    pages: Iterable[list[Document]],
    query: str,
    *,
    scorer: Scorer = default_lexical_score,
    max_workers: int = 4,
) -> list[Candidate]:
    """Score *every* document across ``pages`` via parallel map-reduce.

    Unlike :func:`map_reduce_retrieve`, this performs no top-k truncation and
    keeps every document, including those with a lexical score of ``0.0``. It
    powers the brute-force Corrective RAG path where the LLM grades the whole
    corpus rather than a reduced candidate pool.

    Pages are scored in parallel threads; at most ``max_workers`` pages are in
    flight at once so lazy/streaming page iterators are consumed incrementally.
    Results are deduped by ``document_id`` (keeping the highest score) and
    returned sorted by score descending, ``document_id`` ascending for stable
    ordering.
    """

    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    best: dict[str, Candidate] = {}

    def _merge(candidates: list[Candidate]) -> None:
        for candidate in candidates:
            current = best.get(candidate.document_id)
            if current is None or candidate.score > current.score:
                best[candidate.document_id] = candidate

    page_iter = iter(pages)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight = set()
        for _ in range(max_workers):
            page = next(page_iter, None)
            if page is None:
                break
            in_flight.add(
                executor.submit(_score_page_all, page, query, scorer)
            )

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                _merge(future.result())
                page = next(page_iter, None)
                if page is not None:
                    in_flight.add(
                        executor.submit(_score_page_all, page, query, scorer)
                    )

    return sorted(best.values(), key=lambda c: (-c.score, c.document_id))
