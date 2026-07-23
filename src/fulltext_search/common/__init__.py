"""Shared utilities: LLM configuration, environment, and batching helpers."""

from fulltext_search.common.batching import batched, iter_document_pages
from fulltext_search.common.env import ensure_llm_ready, load_environment
from fulltext_search.common.llms import (
    DEFAULT_MODEL,
    build_lm,
    configure_default_lm,
)

__all__ = [
    "DEFAULT_MODEL",
    "batched",
    "build_lm",
    "configure_default_lm",
    "ensure_llm_ready",
    "iter_document_pages",
    "load_environment",
]
