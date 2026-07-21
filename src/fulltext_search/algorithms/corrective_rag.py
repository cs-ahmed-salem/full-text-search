"""Corrective RAG (CRAG) search algorithm built on DSPy.

Implements the corrective retrieval workflow described in
https://www.meilisearch.com/blog/corrective-rag :

1. **Retrieve** candidates via scalable map-reduce lexical recall over document
   pages (see :mod:`fulltext_search.algorithms.map_reduce`).
2. **Grade** each candidate for relevance with an LLM.
3. **Correct** weak retrieval: when too few relevant documents are found, rewrite
   the query and re-search the same corpus, then grade the new candidates.
4. **Generate** an answer from the validated context (only in :meth:`answer`).

``search`` returns graded/corrected hits; ``answer`` additionally generates a
grounded answer. Retrieval scales via map-reduce; the LLM is used only on the
small reduced candidate set (grade / rewrite / generate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

import dspy

from fulltext_search.algorithms.base import SearchAlgorithm, SearchResult
from fulltext_search.algorithms.map_reduce import (
    Candidate,
    Scorer,
    default_lexical_score,
    map_reduce_retrieve,
)
from fulltext_search.common.batching import iter_document_pages
from fulltext_search.common.llms import configure_default_lm
from fulltext_search.datasources.base import Document

Relevance = Literal["relevant", "ambiguous", "irrelevant"]

_RELEVANCE_WEIGHTS: dict[str, float] = {
    "relevant": 1.0,
    "ambiguous": 0.5,
    "irrelevant": 0.0,
}

RetrieveFn = Callable[[str, int], list[Candidate]]


class GradeDocument(dspy.Signature):
    """Grade how relevant a document is to answering the question.

    Use ``relevant`` when the document directly helps answer the question,
    ``ambiguous`` when it is only tangentially related, and ``irrelevant`` when
    it does not help at all.
    """

    question: str = dspy.InputField()
    document: str = dspy.InputField()
    relevance: Relevance = dspy.OutputField()
    rationale: str = dspy.OutputField(desc="one short sentence justifying the grade")


class RewriteQuery(dspy.Signature):
    """Rewrite a search query that returned weak results into a better one.

    Produce a single improved query that is more likely to retrieve relevant
    documents from the same corpus (add synonyms, disambiguate, or rephrase).
    """

    question: str = dspy.InputField()
    rationale: str = dspy.InputField(desc="why the previous retrieval was weak")
    rewritten_query: str = dspy.OutputField()


class GenerateAnswer(dspy.Signature):
    """Answer the question using only the provided context.

    Ground the answer in the context and ignore any low-confidence or
    irrelevant material. If the context does not contain the answer, say so.
    """

    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


@dataclass(frozen=True, slots=True)
class GradedCandidate:
    """A retrieval candidate annotated with an LLM relevance grade."""

    candidate: Candidate
    relevance: str
    rationale: str

    @property
    def relevance_weight(self) -> float:
        return _RELEVANCE_WEIGHTS.get(self.relevance, 0.0)

    @property
    def is_accepted(self) -> bool:
        return self.relevance != "irrelevant"


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    """Output of the retrieve + grade + correct stage."""

    graded: list[GradedCandidate]
    rewritten_query: str | None


@dataclass(frozen=True, slots=True)
class CorrectiveRAGAnswer:
    """A generated answer plus its supporting graded hits."""

    answer: str
    results: list[SearchResult] = field(default_factory=list)
    rewritten_query: str | None = None


def _normalize_relevance(value: object) -> str:
    text = str(value).strip().casefold()
    if text in _RELEVANCE_WEIGHTS:
        return text
    return "ambiguous"


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


class CorrectiveRAGModule(dspy.Module):
    """DSPy module wiring grade -> correct -> generate.

    Retrieval is injected as a callable so the same corpus is searched for both
    the original and rewritten queries.
    """

    def __init__(
        self,
        *,
        min_relevant: int = 1,
        max_document_chars: int = 2000,
    ) -> None:
        super().__init__()
        self.grade = dspy.Predict(GradeDocument)
        self.rewrite = dspy.Predict(RewriteQuery)
        self.generate = dspy.ChainOfThought(GenerateAnswer)
        self.min_relevant = min_relevant
        self.max_document_chars = max_document_chars

    def grade_candidates(
        self,
        question: str,
        candidates: list[Candidate],
    ) -> list[GradedCandidate]:
        graded: list[GradedCandidate] = []
        for candidate in candidates:
            document_text = _truncate(
                candidate.document.content, self.max_document_chars
            )
            prediction = self.grade(question=question, document=document_text)
            graded.append(
                GradedCandidate(
                    candidate=candidate,
                    relevance=_normalize_relevance(
                        getattr(prediction, "relevance", "ambiguous")
                    ),
                    rationale=str(getattr(prediction, "rationale", "")),
                )
            )
        return graded

    def correct_and_retrieve(
        self,
        question: str,
        retrieve: RetrieveFn,
        limit: int,
    ) -> CorrectionResult:
        candidates = retrieve(question, limit)
        graded = self.grade_candidates(question, candidates)

        relevant = [g for g in graded if g.relevance == "relevant"]
        rewritten_query: str | None = None

        if len(relevant) < self.min_relevant:
            rationale = (
                f"Only {len(relevant)} of {len(graded)} retrieved documents "
                "were relevant to the question."
            )
            prediction = self.rewrite(question=question, rationale=rationale)
            rewritten_query = str(
                getattr(prediction, "rewritten_query", "")
            ).strip()

            if rewritten_query and rewritten_query != question:
                seen = {g.candidate.document_id for g in graded}
                extra = [
                    candidate
                    for candidate in retrieve(rewritten_query, limit)
                    if candidate.document_id not in seen
                ]
                graded.extend(self.grade_candidates(rewritten_query, extra))
            else:
                rewritten_query = None

        return CorrectionResult(graded=graded, rewritten_query=rewritten_query)

    def build_context(
        self,
        graded: list[GradedCandidate],
        limit: int,
    ) -> str:
        accepted = _rank_graded(graded)[:limit]
        blocks: list[str] = []
        for item in accepted:
            content = _truncate(
                item.candidate.document.content, self.max_document_chars
            )
            blocks.append(f"[{item.candidate.document_id}] {content}")
        return "\n\n".join(blocks)

    def forward(
        self,
        question: str,
        retrieve: RetrieveFn,
        limit: int = 10,
    ) -> dspy.Prediction:
        correction = self.correct_and_retrieve(question, retrieve, limit)
        context = self.build_context(correction.graded, limit)
        prediction = self.generate(context=context, question=question)
        return dspy.Prediction(
            answer=str(getattr(prediction, "answer", "")),
            graded=correction.graded,
            rewritten_query=correction.rewritten_query,
        )


def _rank_graded(graded: list[GradedCandidate]) -> list[GradedCandidate]:
    accepted = [g for g in graded if g.is_accepted]
    accepted.sort(
        key=lambda g: (
            -g.relevance_weight,
            -g.candidate.score,
            g.candidate.document_id,
        )
    )
    return accepted


class CorrectiveRAGAlgorithm(SearchAlgorithm):
    """Corrective RAG over an in-memory corpus with scalable map-reduce recall.

    Parameters
    ----------
    lm:
        A configured :class:`dspy.LM`. When ``None`` the default LM is built and
        installed from the environment via
        :func:`fulltext_search.common.llms.configure_default_lm` on first use.
    batch_size:
        Page size for map-reduce retrieval.
    max_workers:
        Threads used to score pages in parallel during retrieval.
    recall_per_batch:
        Candidates kept per page during the map step (defaults to the search
        limit).
    candidate_pool:
        Number of lexical candidates fed into LLM grading per query.
    min_relevant:
        Minimum number of ``relevant`` documents before the corrective query
        rewrite is skipped.
    scorer:
        Lexical scoring function for the map step.
    module:
        A pre-built (e.g. DSPy-optimized) :class:`CorrectiveRAGModule`. When
        ``None`` one is created lazily.
    """

    def __init__(
        self,
        *,
        lm: dspy.LM | None = None,
        batch_size: int = 256,
        max_workers: int = 4,
        recall_per_batch: int | None = None,
        candidate_pool: int = 20,
        min_relevant: int = 1,
        max_document_chars: int = 2000,
        scorer: Scorer = default_lexical_score,
        module: CorrectiveRAGModule | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if candidate_pool < 1:
            raise ValueError(
                f"candidate_pool must be >= 1, got {candidate_pool}"
            )

        self._documents: dict[str, Document] = {}
        self._lm = lm
        self._configured = False
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.recall_per_batch = recall_per_batch
        self.candidate_pool = candidate_pool
        self.min_relevant = min_relevant
        self.max_document_chars = max_document_chars
        self.scorer = scorer
        self._module = module

    def index(self, documents: Iterable[Document]) -> None:
        for document in documents:
            self._documents[document.id] = document

    def clear(self) -> None:
        self._documents.clear()

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        module = self._ensure_module()
        pool = max(limit, self.candidate_pool)
        correction = module.correct_and_retrieve(
            query, self._retrieve, pool
        )
        return self._to_search_results(correction.graded, limit)

    def answer(self, query: str, *, limit: int = 10) -> CorrectiveRAGAnswer:
        module = self._ensure_module()
        pool = max(limit, self.candidate_pool)
        prediction = module(query, self._retrieve, pool)
        results = self._to_search_results(prediction.graded, limit)
        return CorrectiveRAGAnswer(
            answer=prediction.answer,
            results=results,
            rewritten_query=prediction.rewritten_query,
        )

    def _retrieve(self, query: str, limit: int) -> list[Candidate]:
        pages = iter_document_pages(
            self._documents.values(), self.batch_size
        )
        return map_reduce_retrieve(
            pages,
            query,
            scorer=self.scorer,
            limit=limit,
            recall_per_page=self.recall_per_batch,
            max_workers=self.max_workers,
        )

    def _to_search_results(
        self,
        graded: list[GradedCandidate],
        limit: int,
    ) -> list[SearchResult]:
        ranked = _rank_graded(graded)[:limit]
        results: list[SearchResult] = []
        for item in ranked:
            candidate = item.candidate
            results.append(
                SearchResult(
                    document_id=candidate.document_id,
                    score=item.relevance_weight,
                    document=candidate.document,
                    metadata={
                        "relevance": item.relevance,
                        "rationale": item.rationale,
                        "lexical_score": candidate.score,
                    },
                )
            )
        return results

    def _ensure_module(self) -> CorrectiveRAGModule:
        if self._lm is not None:
            if not self._configured:
                dspy.configure(lm=self._lm)
                self._configured = True
        elif dspy.settings.lm is None:
            self._lm = configure_default_lm()
            self._configured = True

        if self._module is None:
            self._module = CorrectiveRAGModule(
                min_relevant=self.min_relevant,
                max_document_chars=self.max_document_chars,
            )
        return self._module
