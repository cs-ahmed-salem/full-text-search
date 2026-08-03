"""High-level search engine facade.

Wires a :class:`~fulltext_search.datasources.base.DataSource` to a
:class:`~fulltext_search.algorithms.base.SearchAlgorithm` and exposes a small
library API for indexing and querying.

Typical usage::

    from fulltext_search import SearchEngine

    engine = SearchEngine.from_paging_api(
        url="http://localhost:8080/api/v1/.../tasks",
        headers={"x-internal-pass": "..."},
    )
    result = engine.ask("Should I send an email to complete task AC378?", top_k=5)
    for record in result.records:
        print(record.id, record.metadata.get("title"))

Or from environment variables (see :meth:`SearchEngine.from_env`)::

    engine = SearchEngine.from_env()
    result = engine.ask("...")
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Iterable

from fulltext_search.algorithms.base import SearchAlgorithm, SearchResult
from fulltext_search.algorithms.corrective_rag import (
    CorrectiveRAGAlgorithm,
    CorrectiveRAGAnswer,
)
from fulltext_search.common.env import ensure_llm_ready
from fulltext_search.common.llms import configure_default_lm
from fulltext_search.datasources.base import DataSource, Document
from fulltext_search.datasources.paging_api import PagingApiSource
from fulltext_search.results import SearchPage, SessionStore

DocumentTransform = Callable[[Document], Document]

ENV_API_URL = "FULLTEXT_SEARCH_API_URL"
ENV_API_PASS = "FULLTEXT_SEARCH_API_PASS"
ENV_API_HEADERS = "FULLTEXT_SEARCH_API_HEADERS"
ENV_API_PARAMS = "FULLTEXT_SEARCH_API_PARAMS"
ENV_API_PAGE_SIZE = "FULLTEXT_SEARCH_API_PAGE_SIZE"


def enrich_task_document(document: Document) -> Document:
    """Fold task title/code/state into searchable content.

    Useful for Spring-style task payloads where ``description`` is the primary
    content field and identifiers live in metadata.
    """

    title = str(document.metadata.get("title") or "")
    code = str(document.metadata.get("code") or "")
    state = str(document.metadata.get("currentState") or "")
    preconditions = document.metadata.get("preconditions") or []

    parts: list[str] = []
    if code:
        parts.append(f"Code: {code}")
    if title:
        parts.append(f"Title: {title}")
    if state:
        parts.append(f"State: {state}")
    parts.append(f"Description: {document.content}")
    if preconditions:
        parts.append(
            "Preconditions: " + "; ".join(str(item) for item in preconditions)
        )

    return Document(
        id=document.id,
        content="\n".join(parts),
        metadata=document.metadata,
    )


class SearchEngine:
    """Index a data source and answer queries with a search algorithm.

    Parameters
    ----------
    source:
        Document provider. Indexed immediately unless ``autoload`` is false.
    algorithm:
        Search backend. Defaults to :class:`CorrectiveRAGAlgorithm` with the
        environment-configured LM.
    document_transform:
        Optional per-document rewrite applied before indexing (e.g.
        :func:`enrich_task_document`).
    default_top_k:
        Default top-k for :meth:`ask` / :meth:`search` (must be ``> 0``).
    autoload:
        When true (default), documents are loaded from ``source`` during init.
    session_store:
        Store for pageable brute-force search sessions. Defaults to a temp-dir
        :class:`~fulltext_search.results.SessionStore`.
    """

    def __init__(
        self,
        source: DataSource,
        algorithm: SearchAlgorithm | None = None,
        *,
        document_transform: DocumentTransform | None = enrich_task_document,
        default_top_k: int = 10,
        default_limit: int | None = None,
        autoload: bool = True,
        session_store: SessionStore | None = None,
    ) -> None:
        if algorithm is None:
            model = ensure_llm_ready()
            algorithm = CorrectiveRAGAlgorithm(lm=configure_default_lm(model))
            self.model = model
        else:
            self.model = getattr(getattr(algorithm, "_lm", None), "model", None)

        top_k = default_limit if default_limit is not None else default_top_k
        if top_k < 1:
            raise ValueError(f"default_top_k must be > 0, got {top_k}")

        self.source = source
        self.algorithm = algorithm
        self.document_transform = document_transform
        self.default_top_k = top_k
        self.default_limit = top_k  # backwards-compatible alias
        self.document_count = 0
        self.session_store = session_store or SessionStore()

        if autoload:
            self.reload()

    @classmethod
    def from_paging_api(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        page_size: int = 50,
        algorithm: SearchAlgorithm | None = None,
        document_transform: DocumentTransform | None = enrich_task_document,
        default_top_k: int = 10,
        default_limit: int | None = None,
        **paging_kwargs: Any,
    ) -> SearchEngine:
        """Build an engine backed by :class:`PagingApiSource`."""

        source = PagingApiSource(
            url,
            headers=headers,
            params=params,
            page_size=page_size,
            **paging_kwargs,
        )
        return cls(
            source,
            algorithm,
            document_transform=document_transform,
            default_top_k=default_top_k,
            default_limit=default_limit,
        )

    @classmethod
    def from_env(
        cls,
        *,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        page_size: int | None = None,
        algorithm: SearchAlgorithm | None = None,
        document_transform: DocumentTransform | None = enrich_task_document,
        default_top_k: int = 10,
        default_limit: int | None = None,
    ) -> SearchEngine:
        """Build a paging-API engine from environment / ``.env`` configuration.

        Recognized variables:

        * ``FULLTEXT_SEARCH_API_URL`` -- required unless ``url`` is passed
        * ``FULLTEXT_SEARCH_API_PASS`` -- sets the ``x-internal-pass`` header
        * ``FULLTEXT_SEARCH_API_HEADERS`` -- JSON object of extra headers
        * ``FULLTEXT_SEARCH_API_PARAMS`` -- JSON object of extra GET params
        * ``FULLTEXT_SEARCH_API_PAGE_SIZE`` -- page size (default 50)
        """

        ensure_llm_ready()

        api_url = url or os.getenv(ENV_API_URL)
        if not api_url:
            raise ValueError(
                f"Provide url=... or set {ENV_API_URL} in the environment"
            )

        api_headers = {
            **_string_dict_env(ENV_API_HEADERS),
            **(headers or {}),
        }
        api_pass = os.getenv(ENV_API_PASS)
        if api_pass and not _has_header(api_headers, "x-internal-pass"):
            api_headers["x-internal-pass"] = api_pass

        api_params = {
            **_json_object_env(ENV_API_PARAMS),
            **(params or {}),
        }
        resolved_page_size = page_size or int(
            os.getenv(ENV_API_PAGE_SIZE, "50")
        )

        return cls.from_paging_api(
            api_url,
            headers=api_headers,
            params=api_params,
            page_size=resolved_page_size,
            algorithm=algorithm,
            document_transform=document_transform,
            default_top_k=default_top_k,
            default_limit=default_limit,
        )

    def reload(self) -> int:
        """Re-index all documents from the configured source."""

        transform = self.document_transform or _identity
        with self.source as source:
            documents = [transform(doc) for doc in source.iter_documents()]

        if hasattr(self.algorithm, "clear"):
            try:
                self.algorithm.clear()
            except NotImplementedError:
                pass

        self.algorithm.index(documents)
        self.document_count = len(documents)
        return self.document_count

    def index(self, documents: Iterable[Document]) -> None:
        """Index additional documents (applies ``document_transform``)."""

        transform = self.document_transform or _identity
        batch = [transform(doc) for doc in documents]
        self.algorithm.index(batch)
        self.document_count += len(batch)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Return the top-k datasource records by relevance for ``query``."""

        k = self._resolve_top_k(top_k=top_k, limit=limit)
        return self.algorithm.search(query, limit=k)

    def ask(
        self,
        query: str,
        *,
        top_k: int | None = None,
        limit: int | None = None,
    ) -> CorrectiveRAGAnswer:
        """Run Corrective RAG and return top-k datasource records.

        Requires an algorithm that implements ``answer`` (e.g.
        :class:`CorrectiveRAGAlgorithm`).
        """

        answer_fn = getattr(self.algorithm, "answer", None)
        if answer_fn is None:
            raise TypeError(
                f"{type(self.algorithm).__name__} does not implement answer(); "
                "use search() or pass a CorrectiveRAGAlgorithm"
            )
        k = self._resolve_top_k(top_k=top_k, limit=limit)
        return answer_fn(query, top_k=k)

    def search_all(
        self,
        query: str,
        *,
        page: int = 0,
        size: int = 20,
    ) -> SearchPage:
        """Brute-force search the whole corpus and return the first page.

        Grades every indexed document, persists the ranked ``relevant`` records
        to a session, and returns the requested page. Use the returned
        ``session_id`` with :meth:`get_search_page` to page further without
        re-running LLM grading.

        Requires an algorithm implementing ``answer_all`` (e.g.
        :class:`CorrectiveRAGAlgorithm`).
        """

        answer_all_fn = getattr(self.algorithm, "answer_all", None)
        if answer_all_fn is None:
            raise TypeError(
                f"{type(self.algorithm).__name__} does not implement "
                "answer_all(); pass a CorrectiveRAGAlgorithm"
            )
        _validate_paging(page=page, size=size)

        answer = answer_all_fn(query)
        session_id = self.session_store.create(
            answer.results,
            query=query,
            rewritten_query=answer.rewritten_query,
        )
        return self.session_store.get_page(session_id, page=page, size=size)

    def get_search_page(
        self,
        session_id: str,
        *,
        page: int = 0,
        size: int = 20,
    ) -> SearchPage:
        """Return a page from a previously created brute-force search session."""

        _validate_paging(page=page, size=size)
        return self.session_store.get_page(session_id, page=page, size=size)

    def _resolve_top_k(
        self,
        *,
        top_k: int | None,
        limit: int | None,
    ) -> int:
        if top_k is not None and limit is not None and top_k != limit:
            raise ValueError("Pass only one of top_k or limit")
        k = self.default_top_k if top_k is None and limit is None else (
            top_k if top_k is not None else limit
        )
        assert k is not None
        if k < 1:
            raise ValueError(f"top_k must be > 0, got {k}")
        return k


def _validate_paging(*, page: int, size: int) -> None:
    if page < 0:
        raise ValueError(f"page must be >= 0, got {page}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")


def _identity(document: Document) -> Document:
    return document


def _json_object_env(name: str) -> dict[str, Any]:
    raw = os.getenv(name)
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _string_dict_env(name: str) -> dict[str, str]:
    return {str(k): str(v) for k, v in _json_object_env(name).items()}


def _has_header(headers: dict[str, str], name: str) -> bool:
    target = name.lower()
    return any(key.lower() == target for key in headers)
