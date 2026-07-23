"""Opt-in, live-LLM evaluation of Corrective RAG on the task dataset.

This is marked ``evaluation`` and excluded from the default ``pytest`` run
(see ``addopts`` in ``pyproject.toml``). Run it explicitly with::

    pytest -m evaluation

It calls a real LLM, so it skips cleanly when no credentials are available
(no API key and no usable Azure CLI login). Assertions check a metric floor
rather than exact values, since LLM outputs are non-deterministic.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import eval_corrective_rag as harness  # noqa: E402

pytestmark = pytest.mark.evaluation

NUM_QUERIES = 6
LIMIT = 5
HIT_FLOOR = 0.8


@pytest.fixture(scope="module")
def evaluation_records():
    try:
        model = harness.ensure_llm_ready()
    except (
        RuntimeError,
        FileNotFoundError,
        subprocess.SubprocessError,
    ) as exc:
        pytest.skip(f"No usable LLM credentials for evaluation: {exc}")

    from fulltext_search.algorithms.corrective_rag import (
        CorrectiveRAGAlgorithm,
    )
    from fulltext_search.common.llms import configure_default_lm

    lm = configure_default_lm(model)
    algo = CorrectiveRAGAlgorithm(
        lm=lm,
        batch_size=256,
        max_workers=8,
        candidate_pool=8,
        grade_workers=8,
    )
    algo.index(harness.load_corpus())

    queries = harness.load_queries(num_queries=NUM_QUERIES, seed=13)
    return harness.evaluate(algo, queries, limit=LIMIT)


def test_evaluation_runs_all_queries(evaluation_records) -> None:
    assert len(evaluation_records) == NUM_QUERIES


def test_crag_meets_hit_floor(evaluation_records) -> None:
    stats = harness.aggregate(evaluation_records)
    assert stats["crag_hit"] >= HIT_FLOOR, stats
    assert stats["crag_mrr"] > 0.0, stats


def test_graded_results_have_relevance_metadata(evaluation_records) -> None:
    # Every returned hit should carry a normalized relevance grade.
    valid = {"relevant", "ambiguous"}
    for record in evaluation_records:
        if record.top_relevance:
            assert record.top_relevance in valid, record.top_relevance
