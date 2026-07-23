"""Evaluate the Corrective RAG algorithm on the task-search dataset.

Loads ``tests/data/tasks.jsonl`` as the corpus and evaluates retrieval against
the ground-truth ``expected_relevant_task_ids`` in
``tests/data/hard_queries.jsonl``. Reports Hit@k / Recall@k / MRR for the
corrective-RAG pipeline and, for comparison, a pure lexical baseline.

LLM configuration is read from ``.env`` via
:mod:`fulltext_search.common.llms`. Azure OpenAI works out of the box: if no
API key is set, the harness mints an Azure AD token with the Azure CLI (``az``)
and exposes it via ``AZURE_AD_TOKEN``.

The reusable pieces (:func:`ensure_llm_ready`, :func:`load_corpus`,
:func:`load_queries`, :func:`evaluate`, :func:`aggregate`) are imported by the
gated evaluation test in ``tests/algorithms/test_corrective_rag_eval.py``.

Usage::

    python scripts/eval_corrective_rag.py --num-queries 15 --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_DIR = REPO_ROOT / "tests" / "data"
TASKS_PATH = DATA_DIR / "tasks.jsonl"
QUERIES_PATH = DATA_DIR / "hard_queries.jsonl"
AZURE_TOKEN_RESOURCE = "https://cognitiveservices.azure.com"

_API_KEY_VARS = (
    "FULLTEXT_SEARCH_LM_API_KEY",
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_KEY",
)

if TYPE_CHECKING:
    from fulltext_search.algorithms.corrective_rag import (
        CorrectiveRAGAlgorithm,
    )
    from fulltext_search.datasources.base import Document


# --------------------------------------------------------------------------- #
# Environment / credentials
# --------------------------------------------------------------------------- #
def load_environment() -> None:
    """Load ``.env`` from the repo root into ``os.environ``."""

    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _parse_dotenv(env_path)
        return
    load_dotenv(env_path)


def _parse_dotenv(env_path: Path) -> None:
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(
            key.strip(), value.strip().strip('"').strip("'")
        )


def ensure_llm_ready() -> str:
    """Ensure an LM can be configured, returning the resolved model id.

    Raises ``RuntimeError`` (or a subprocess error) when no usable credentials
    are available, so callers such as the gated test can skip cleanly.
    """

    load_environment()
    from fulltext_search.common.llms import default_model

    model = default_model()
    has_key = any(os.getenv(var) for var in _API_KEY_VARS)

    if model.startswith("azure/"):
        if has_key or os.getenv("AZURE_AD_TOKEN"):
            return model
        token = subprocess.check_output(
            [
                "az",
                "account",
                "get-access-token",
                "--resource",
                AZURE_TOKEN_RESOURCE,
                "--query",
                "accessToken",
                "-o",
                "tsv",
            ]
        ).decode().strip()
        os.environ["AZURE_AD_TOKEN"] = token
        # Reasoning-capable Azure deployments need a generous output budget.
        os.environ.setdefault("FULLTEXT_SEARCH_LM_MAX_TOKENS", "16000")
        return model

    if not has_key:
        raise RuntimeError(
            f"No API key found for model {model!r}; set one of "
            f"{', '.join(_API_KEY_VARS)}."
        )
    return model


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def task_to_document(task: dict[str, Any]) -> "Document":
    from fulltext_search.datasources.base import Document

    parts = [
        f"Title: {task.get('title', '')}",
        f"Domain: {task.get('domain', '')}",
        f"Owner: {task.get('owner', '')}",
        f"Status: {task.get('status', '')}",
        f"Description: {task.get('description', '')}",
    ]
    start_conditions = task.get("start_conditions") or []
    if start_conditions:
        parts.append("Start conditions: " + "; ".join(start_conditions))
    completion_markers = task.get("completion_markers") or []
    if completion_markers:
        parts.append("Completion markers: " + "; ".join(completion_markers))

    return Document(
        id=str(task["id"]),
        content="\n".join(parts),
        metadata={
            "title": task.get("title", ""),
            "status": task.get("status", ""),
            "domain": task.get("domain", ""),
        },
    )


def load_corpus(path: Path | str = TASKS_PATH) -> list["Document"]:
    return [task_to_document(task) for task in _read_jsonl(Path(path))]


def load_queries(
    path: Path | str = QUERIES_PATH,
    *,
    num_queries: int | None = None,
    seed: int = 13,
) -> list[dict[str, Any]]:
    queries = [
        q
        for q in _read_jsonl(Path(path))
        if q.get("expected_relevant_task_ids")
    ]
    if num_queries is not None:
        rng = random.Random(seed)
        rng.shuffle(queries)
        queries = queries[:num_queries]
    return queries


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class QueryEval:
    query: str
    expected: set[str]
    crag: dict[str, float]
    lexical: dict[str, float]
    rewritten_query: str | None
    top_title: str | None
    top_relevance: str | None


def compute_metrics(
    expected: set[str],
    ranked_ids: list[str],
    k: int,
) -> dict[str, float]:
    top_k = ranked_ids[:k]
    found = expected.intersection(top_k)
    recall = len(found) / len(expected) if expected else 0.0
    rr = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in expected:
            rr = 1.0 / rank
            break
    return {"hit": float(bool(found)), "recall": recall, "rr": rr}


def evaluate(
    algo: "CorrectiveRAGAlgorithm",
    queries: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[QueryEval]:
    """Run CRAG and a lexical baseline over ``queries`` and collect metrics."""

    module = algo._ensure_module()
    pool = max(limit, algo.candidate_pool)

    records: list[QueryEval] = []
    for query in queries:
        text = query["query"]
        expected = {str(x) for x in query["expected_relevant_task_ids"]}

        lexical = algo._retrieve(text, pool)
        lexical_metrics = compute_metrics(
            expected, [c.document_id for c in lexical], limit
        )

        correction = module.correct_and_retrieve(text, algo._retrieve, pool)
        results = algo._to_search_results(correction.graded, limit)
        crag_metrics = compute_metrics(
            expected, [r.document_id for r in results], limit
        )

        top = results[0] if results else None
        top_title = None
        top_relevance = None
        if top is not None:
            if top.document is not None:
                top_title = str(top.document.metadata.get("title", ""))
            top_relevance = str(top.metadata.get("relevance", ""))

        records.append(
            QueryEval(
                query=text,
                expected=expected,
                crag=crag_metrics,
                lexical=lexical_metrics,
                rewritten_query=correction.rewritten_query,
                top_title=top_title,
                top_relevance=top_relevance,
            )
        )
    return records


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(records: list[QueryEval]) -> dict[str, float]:
    return {
        "queries": float(len(records)),
        "rewrites": float(
            sum(1 for r in records if r.rewritten_query)
        ),
        "crag_hit": _mean([r.crag["hit"] for r in records]),
        "crag_recall": _mean([r.crag["recall"] for r in records]),
        "crag_mrr": _mean([r.crag["rr"] for r in records]),
        "lexical_hit": _mean([r.lexical["hit"] for r in records]),
        "lexical_recall": _mean([r.lexical["recall"] for r in records]),
        "lexical_mrr": _mean([r.lexical["rr"] for r in records]),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(TASKS_PATH))
    parser.add_argument("--queries", default=str(QUERIES_PATH))
    parser.add_argument("--num-queries", type=int, default=15)
    parser.add_argument("--limit", type=int, default=5, help="top-k")
    parser.add_argument("--candidate-pool", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--grade-workers", type=int, default=8)
    parser.add_argument("--min-relevant", type=int, default=1)
    parser.add_argument("--answers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    model = ensure_llm_ready()

    from fulltext_search.algorithms.corrective_rag import (
        CorrectiveRAGAlgorithm,
    )
    from fulltext_search.common.llms import configure_default_lm

    lm = configure_default_lm(model)
    print(f"LM: {lm.model}")

    documents = load_corpus(args.tasks)
    print(f"Corpus: {len(documents)} tasks")

    algo = CorrectiveRAGAlgorithm(
        lm=lm,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        candidate_pool=args.candidate_pool,
        grade_workers=args.grade_workers,
        min_relevant=args.min_relevant,
    )
    algo.index(documents)

    queries = load_queries(
        args.queries, num_queries=args.num_queries, seed=args.seed
    )
    print(f"Evaluating {len(queries)} queries (top-{args.limit})\n")

    started = time.time()
    records = evaluate(algo, queries, limit=args.limit)
    elapsed = time.time() - started

    for i, record in enumerate(records, start=1):
        marker = "HIT " if record.crag["hit"] else "miss"
        rewrite = (
            f" | rewrite -> {record.rewritten_query!r}"
            if record.rewritten_query
            else ""
        )
        print(f"[{i:>2}] {marker} q={record.query[:80]!r}{rewrite}")
        if record.top_title is not None:
            print(
                f"     top: {record.top_title[:70]!r} "
                f"({record.top_relevance})"
            )

    for record in records[: args.answers]:
        answer = algo.answer(record.query, limit=args.limit)
        print(f"\nQ: {record.query[:120]!r}")
        print(f"A: {answer.answer[:300]!r}")

    stats = aggregate(records)
    print("\n=== Results ===")
    print(f"queries              : {int(stats['queries'])}")
    print(f"elapsed              : {elapsed:.1f}s")
    print(
        f"query rewrites fired : "
        f"{int(stats['rewrites'])}/{int(stats['queries'])}"
    )
    print(
        f"CRAG    Hit@{args.limit}={stats['crag_hit']:.3f} "
        f"Recall@{args.limit}={stats['crag_recall']:.3f} "
        f"MRR={stats['crag_mrr']:.3f}"
    )
    print(
        f"Lexical Hit@{args.limit}={stats['lexical_hit']:.3f} "
        f"Recall@{args.limit}={stats['lexical_recall']:.3f} "
        f"MRR={stats['lexical_mrr']:.3f}"
    )


if __name__ == "__main__":
    main()
