"""Tests for the DSPy Corrective RAG algorithm with mocked predictors.

The LLM predictors (grade / rewrite / generate) are replaced with deterministic
fakes so no network calls are made.
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


class FakeGenerate:
    def __init__(self) -> None:
        self.contexts: list[str] = []

    def __call__(self, *, context: str, question: str) -> SimpleNamespace:
        self.contexts.append(context)
        return SimpleNamespace(answer=f"answer::{question}")


def _build_algo(
    documents: list[Document],
    *,
    grade: FakeGrade,
    rewrite: FakeRewrite | None = None,
    generate: FakeGenerate | None = None,
    min_relevant: int = 1,
) -> CorrectiveRAGAlgorithm:
    module = CorrectiveRAGModule(min_relevant=min_relevant)
    module.grade = grade
    module.rewrite = rewrite or FakeRewrite("unused")
    module.generate = generate or FakeGenerate()

    algo = CorrectiveRAGAlgorithm(
        lm=dspy.LM("openai/gpt-4o-mini"),
        module=module,
        candidate_pool=10,
        min_relevant=min_relevant,
    )
    algo.index(documents)
    return algo


def test_search_returns_only_accepted_hits() -> None:
    documents = [
        Document(id="a", content="alpha content here"),
        Document(id="b", content="beta only content"),
    ]
    grade = FakeGrade(relevant_terms=["alpha"])
    algo = _build_algo(documents, grade=grade)

    results = algo.search("alpha beta", limit=5)

    assert [r.document_id for r in results] == ["a"]
    assert results[0].metadata["relevance"] == "relevant"
    assert "lexical_score" in results[0].metadata


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

    result = algo.answer("zzz", limit=5)

    assert rewrite.calls == 1
    assert result.rewritten_query == "alpha"
    assert {r.document_id for r in result.results} == {"a", "b"}


def test_no_rewrite_when_relevant_docs_found() -> None:
    documents = [Document(id="a", content="alpha content")]
    grade = FakeGrade(relevant_terms=["alpha"])
    rewrite = FakeRewrite("alpha")
    algo = _build_algo(documents, grade=grade, rewrite=rewrite)

    result = algo.answer("alpha", limit=5)

    assert rewrite.calls == 0
    assert result.rewritten_query is None


def test_answer_generates_from_context() -> None:
    documents = [Document(id="a", content="alpha content")]
    grade = FakeGrade(relevant_terms=["alpha"])
    generate = FakeGenerate()
    algo = _build_algo(documents, grade=grade, generate=generate)

    result = algo.answer("alpha", limit=5)

    assert isinstance(result, CorrectiveRAGAnswer)
    assert result.answer == "answer::alpha"
    assert [r.document_id for r in result.results] == ["a"]
    assert "alpha content" in generate.contexts[0]


def test_search_and_answer_shapes_differ() -> None:
    documents = [Document(id="a", content="alpha content")]
    grade = FakeGrade(relevant_terms=["alpha"])
    algo = _build_algo(documents, grade=grade)

    hits = algo.search("alpha", limit=5)
    answer = algo.answer("alpha", limit=5)

    assert isinstance(hits, list)
    assert isinstance(answer, CorrectiveRAGAnswer)


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
