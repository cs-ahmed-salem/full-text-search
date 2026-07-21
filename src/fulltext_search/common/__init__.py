"""Shared utilities: LLM configuration and batching helpers."""

from fulltext_search.common.batching import batched, iter_document_pages
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
    "iter_document_pages",
]
