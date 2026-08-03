"""Tests for the DSPy Corrective RAG algorithm with mocked predictors.

The LLM predictors (grade / rewrite) are replaced with deterministic fakes so
no network calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace

import dspy
import pytest

from fulltext_search.algorithms.corrective_rag import (
    CorrectiveRAGAlgorithm,
    CorrectiveRAGAnswer,
    CorrectiveRAGModule,
)
from fulltext_search.datasources.base import Document


class FakeGrade:
    """Grade ``relevant`` when the document contains any relevant term."""

    def __init__(self, relevant_terms: list[str]) -> None:
        self.relevant_terms = [t.casefold() for t in relevant_terms]
        self.calls: list[str] = []

    def __call__(self, *, question: str, document: str) -> SimpleNamespace:
        self.calls.append(document)
        content = document.casefold()
        relevance = (
            "relevant"
            if any(term in content for term in self.relevant_terms)
            else "irrelevant"
        )
        return SimpleNamespace(relevance=relevance, rationale="because")


class FakeRewrite:
    def __init__(self, new_query: str) -> None:
        self.new_query = new_query
        self.calls = 0

    def __call__(self, *, question: str, rationale: str) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(rewritten_query=self.new_query)


def _build_algo(
    documents: list[Document],
    *,
    grade: FakeGrade,
    rewrite: FakeRewrite | None = None,
    min_relevant: int = 1,
) -> CorrectiveRAGAlgorithm:
    module = CorrectiveRAGModule(min_relevant=min_relevant)
    module.grade = grade
    module.rewrite = rewrite or FakeRewrite("unused")

    algo = CorrectiveRAGAlgorithm(
        lm=dspy.LM("openai/gpt-4o-mini"),
        module=module,
        candidate_pool=10,
        min_relevant=min_relevant,
    )
    algo.index(documents)
    return algo


def test_search_returns_top_k_by_relevance() -> None:
    documents = [
        Document(id="a", content="alpha content here"),
        Document(id="b", content="beta only content"),
    ]
    grade = FakeGrade(relevant_terms=["alpha"])
    algo = _build_algo(documents, grade=grade)

    top1 = algo.search("alpha beta", limit=1)
    assert [r.document_id for r in top1] == ["a"]
    assert top1[0].metadata["relevance"] == "relevant"

    top2 = algo.answer("alpha beta", top_k=2)
    assert [r.id for r in top2.records] == ["a", "b"]
    assert top2.results[0].metadata["relevance"] == "relevant"
    assert top2.results[1].metadata["relevance"] == "irrelevant"
    assert "lexical_score" in top2.results[0].metadata


def test_search_ranks_relevant_above_ambiguous() -> None:
    documents = [
        Document(id="rel", content="alpha alpha"),
        Document(id="amb", content="alpha maybe"),
    ]

    class MixedGrade:
        def __call__(self, *, question: str, document: str) -> SimpleNamespace:
            if "maybe" in document:
                return SimpleNamespace(relevance="ambiguous", rationale="x")
            return SimpleNamespace(relevance="relevant", rationale="x")

    module = CorrectiveRAGModule(min_relevant=1)
    module.grade = MixedGrade()
    algo = CorrectiveRAGAlgorithm(
        lm=dspy.LM("openai/gpt-4o-mini"), module=module
    )
    algo.index(documents)

    results = algo.search("alpha", limit=5)

    assert [r.document_id for r in results] == ["rel", "amb"]
    assert results[0].score > results[1].score


def test_rewrite_triggered_when_no_relevant_docs() -> None:
    documents = [
        Document(id="a", content="alpha content"),
        Document(id="b", content="alpha extra"),
    ]
    # Original query matches nothing lexically -> no candidates -> rewrite.
    grade = FakeGrade(relevant_terms=["alpha"])
    rewrite = FakeRewrite("alpha")
    algo = _build_algo(documents, grade=grade, rewrite=rewrite)

    result = algo.answer("zzz", top_k=5)

    assert rewrite.calls == 1
    assert result.rewritten_query == "alpha"
    assert {r.id for r in result.records} == {"a", "b"}


def test_no_rewrite_when_relevant_docs_found() -> None:
    documents = [Document(id="a", content="alpha content")]
    grade = FakeGrade(relevant_terms=["alpha"])
    rewrite = FakeRewrite("alpha")
    algo = _build_algo(documents, grade=grade, rewrite=rewrite)

    result = algo.answer("alpha", top_k=5)

    assert rewrite.calls == 0
    assert result.rewritten_query is None


def test_answer_returns_datasource_records() -> None:
    documents = [
        Document(
            id="a",
            content="alpha content",
            metadata={"title": "Task A", "code": "T01"},
        )
    ]
    grade = FakeGrade(relevant_terms=["alpha"])
    algo = _build_algo(documents, grade=grade)

    result = algo.answer("alpha", top_k=5)

    assert isinstance(result, CorrectiveRAGAnswer)
    assert result.records == documents
    assert result.records[0].metadata["title"] == "Task A"
    assert [r.document_id for r in result.results] == ["a"]


def test_search_matches_answer_results() -> None:
    documents = [Document(id="a", content="alpha content")]
    grade = FakeGrade(relevant_terms=["alpha"])
    algo = _build_algo(documents, grade=grade)

    hits = algo.search("alpha", limit=5)
    answer = algo.answer("alpha", top_k=5)

    assert isinstance(hits, list)
    assert isinstance(answer, CorrectiveRAGAnswer)
    assert [h.document_id for h in hits] == [r.id for r in answer.records]


def test_top_k_must_be_positive() -> None:
    documents = [Document(id="a", content="alpha content")]
    grade = FakeGrade(relevant_terms=["alpha"])
    algo = _build_algo(documents, grade=grade)

    with pytest.raises(ValueError, match="top_k must be > 0"):
        algo.answer("alpha", top_k=0)

    with pytest.raises(ValueError, match="top_k must be > 0"):
        algo.search("alpha", limit=-1)


def test_clear_empties_index() -> None:
    documents = [Document(id="a", content="alpha content")]
    grade = FakeGrade(relevant_terms=["alpha"])
    algo = _build_algo(documents, grade=grade)

    algo.clear()

    assert algo.search("alpha", limit=5) == []


def test_constructor_validates_arguments() -> None:
    with pytest.raises(ValueError):
        CorrectiveRAGAlgorithm(batch_size=0)
    with pytest.raises(ValueError):
        CorrectiveRAGAlgorithm(candidate_pool=0)


def test_answer_all_keeps_only_relevant() -> None:
    documents = [
        Document(id="a", content="alpha content here"),
        Document(id="b", content="beta only content"),
        Document(id="c", content="alpha extra alpha"),
    ]
    grade = FakeGrade(relevant_terms=["alpha"])
    algo = _build_algo(documents, grade=grade)

    result = algo.answer_all("alpha")

    # Only the alpha docs are relevant; beta is dropped entirely.
    assert {r.id for r in result.records} == {"a", "c"}
    assert all(
        hit.metadata["relevance"] == "relevant" for hit in result.results
    )
    # Ranked by lexical score: "c" has two hits, so it outranks "a".
    assert [r.document_id for r in result.results] == ["c", "a"]


def test_answer_all_grades_every_document() -> None:
    documents = [
        Document(id="a", content="alpha"),
        Document(id="b", content="totally unrelated"),
    ]
    grade = FakeGrade(relevant_terms=["alpha"])
    algo = _build_algo(documents, grade=grade)

    algo.answer_all("alpha")

    # Both docs are graded even though "b" has zero lexical score.
    assert len(grade.calls) == 2


def test_answer_all_rewrite_regrades_non_relevant() -> None:
    documents = [
        Document(id="a", content="beta beta"),
        Document(id="b", content="gamma"),
    ]

    class SwitchGrade:
        """Irrelevant on the first query, relevant on the rewritten one."""

        def __init__(self) -> None:
            self.seen_questions: list[str] = []

        def __call__(
            self, *, question: str, document: str
        ) -> SimpleNamespace:
            self.seen_questions.append(question)
            relevance = "relevant" if question == "beta" else "irrelevant"
            return SimpleNamespace(relevance=relevance, rationale="x")

    module = CorrectiveRAGModule(min_relevant=1)
    module.grade = SwitchGrade()
    rewrite = FakeRewrite("beta")
    module.rewrite = rewrite
    algo = CorrectiveRAGAlgorithm(
        lm=dspy.LM("openai/gpt-4o-mini"), module=module
    )
    algo.index(documents)

    result = algo.answer_all("zzz")

    assert rewrite.calls == 1
    assert result.rewritten_query == "beta"
    assert {r.id for r in result.records} == {"a", "b"}
