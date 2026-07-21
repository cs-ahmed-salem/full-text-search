"""LLM client configuration for DSPy-based algorithms.

This is the single place algorithms pull language-model clients from. LMs are
configured from the environment so no secrets live in code:

* ``FULLTEXT_SEARCH_LM`` -- model id passed to :class:`dspy.LM`
  (default ``openai/gpt-4o-mini``).
* ``FULLTEXT_SEARCH_LM_API_KEY`` -- optional explicit API key. When unset the
  underlying provider key (e.g. ``OPENAI_API_KEY``) is used by ``dspy.LM``.
* ``FULLTEXT_SEARCH_LM_API_BASE`` -- optional base URL for self-hosted or
  proxied providers.
* ``FULLTEXT_SEARCH_LM_MAX_TOKENS`` -- optional max output tokens.
* ``FULLTEXT_SEARCH_LM_TEMPERATURE`` -- optional sampling temperature.
"""

from __future__ import annotations

import os
from typing import Any

import dspy

DEFAULT_MODEL = "openai/gpt-4o-mini"

ENV_MODEL = "FULLTEXT_SEARCH_LM"
ENV_API_KEY = "FULLTEXT_SEARCH_LM_API_KEY"
ENV_API_BASE = "FULLTEXT_SEARCH_LM_API_BASE"
ENV_MAX_TOKENS = "FULLTEXT_SEARCH_LM_MAX_TOKENS"
ENV_TEMPERATURE = "FULLTEXT_SEARCH_LM_TEMPERATURE"


def build_lm(
    model: str | None = None,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> dspy.LM:
    """Build a :class:`dspy.LM` from explicit args, falling back to env vars.

    Arguments always take precedence over the corresponding environment
    variables. Unspecified optional settings are simply omitted so ``dspy.LM``
    applies its own defaults.
    """

    model = model or os.getenv(ENV_MODEL, DEFAULT_MODEL)

    lm_kwargs: dict[str, Any] = {}

    api_key = api_key if api_key is not None else os.getenv(ENV_API_KEY)
    if api_key is not None:
        lm_kwargs["api_key"] = api_key

    api_base = api_base if api_base is not None else os.getenv(ENV_API_BASE)
    if api_base is not None:
        lm_kwargs["api_base"] = api_base

    resolved_max_tokens = _resolve_int(max_tokens, ENV_MAX_TOKENS)
    if resolved_max_tokens is not None:
        lm_kwargs["max_tokens"] = resolved_max_tokens

    resolved_temperature = _resolve_float(temperature, ENV_TEMPERATURE)
    if resolved_temperature is not None:
        lm_kwargs["temperature"] = resolved_temperature

    lm_kwargs.update(kwargs)

    return dspy.LM(model, **lm_kwargs)


def configure_default_lm(
    model: str | None = None,
    **kwargs: Any,
) -> dspy.LM:
    """Build an LM, install it as the global DSPy LM, and return it."""

    lm = build_lm(model, **kwargs)
    dspy.configure(lm=lm)
    return lm


def _resolve_int(value: int | None, env_var: str) -> int | None:
    if value is not None:
        return value
    raw = os.getenv(env_var)
    if raw is None or raw == "":
        return None
    return int(raw)


def _resolve_float(value: float | None, env_var: str) -> float | None:
    if value is not None:
        return value
    raw = os.getenv(env_var)
    if raw is None or raw == "":
        return None
    return float(raw)
