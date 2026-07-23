"""LLM client configuration for DSPy-based algorithms.

This is the single place algorithms pull language-model clients from. LMs are
configured from the environment so no secrets live in code:

* ``FULLTEXT_SEARCH_LM`` -- model id passed to :class:`dspy.LM`. Defaults to
  ``azure/<AZURE_OPENAI_DEPLOYMENT>`` when Azure OpenAI env vars are present,
  otherwise ``openai/gpt-4o-mini``.
* ``FULLTEXT_SEARCH_LM_API_KEY`` -- optional explicit API key. When unset the
  underlying provider key (e.g. ``OPENAI_API_KEY``) is used by ``dspy.LM``.
* ``FULLTEXT_SEARCH_LM_API_BASE`` -- optional base URL for self-hosted or
  proxied providers.
* ``FULLTEXT_SEARCH_LM_API_VERSION`` -- optional API version (Azure).
* ``FULLTEXT_SEARCH_LM_MAX_TOKENS`` -- optional max output tokens.
* ``FULLTEXT_SEARCH_LM_TEMPERATURE`` -- optional sampling temperature.

Azure OpenAI is supported out of the box. When the model id starts with
``azure/`` the following are pulled from the environment unless overridden:

* ``AZURE_ENDPOINT`` -- API base.
* ``AZURE_API_VERSION`` -- API version.
* ``AZURE_API_KEY`` / ``AZURE_OPENAI_API_KEY`` -- API key, or
* ``AZURE_AD_TOKEN`` -- an Azure AD bearer token (used when no key is set).
"""

from __future__ import annotations

import os
from typing import Any

import dspy

DEFAULT_MODEL = "openai/gpt-4o-mini"

ENV_MODEL = "FULLTEXT_SEARCH_LM"
ENV_API_KEY = "FULLTEXT_SEARCH_LM_API_KEY"
ENV_API_BASE = "FULLTEXT_SEARCH_LM_API_BASE"
ENV_API_VERSION = "FULLTEXT_SEARCH_LM_API_VERSION"
ENV_MAX_TOKENS = "FULLTEXT_SEARCH_LM_MAX_TOKENS"
ENV_TEMPERATURE = "FULLTEXT_SEARCH_LM_TEMPERATURE"

# Azure-specific environment fallbacks.
ENV_AZURE_ENDPOINT = "AZURE_ENDPOINT"
ENV_AZURE_DEPLOYMENT = "AZURE_OPENAI_DEPLOYMENT"
ENV_AZURE_API_VERSION = "AZURE_API_VERSION"
ENV_AZURE_AD_TOKEN = "AZURE_AD_TOKEN"
_AZURE_API_KEY_VARS = ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY")


def default_model() -> str:
    """Resolve the default model id from the environment.

    Prefers an explicit ``FULLTEXT_SEARCH_LM``; otherwise falls back to an
    Azure deployment when Azure env vars are configured, and finally to
    :data:`DEFAULT_MODEL`.
    """

    explicit = os.getenv(ENV_MODEL)
    if explicit:
        return explicit

    deployment = os.getenv(ENV_AZURE_DEPLOYMENT)
    if deployment and os.getenv(ENV_AZURE_ENDPOINT):
        return f"azure/{deployment}"

    return DEFAULT_MODEL


def build_lm(
    model: str | None = None,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    api_version: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> dspy.LM:
    """Build a :class:`dspy.LM` from explicit args, falling back to env vars.

    Arguments always take precedence over the corresponding environment
    variables. Unspecified optional settings are simply omitted so ``dspy.LM``
    applies its own defaults.
    """

    model = model or default_model()

    lm_kwargs: dict[str, Any] = {}

    api_key = api_key if api_key is not None else os.getenv(ENV_API_KEY)

    api_base = api_base if api_base is not None else os.getenv(ENV_API_BASE)

    api_version = (
        api_version if api_version is not None else os.getenv(ENV_API_VERSION)
    )

    if model.startswith("azure/"):
        _apply_azure_defaults(lm_kwargs, api_key, api_base, api_version)
    else:
        if api_key is not None:
            lm_kwargs["api_key"] = api_key
        if api_base is not None:
            lm_kwargs["api_base"] = api_base
        if api_version is not None:
            lm_kwargs["api_version"] = api_version

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


def _apply_azure_defaults(
    lm_kwargs: dict[str, Any],
    api_key: str | None,
    api_base: str | None,
    api_version: str | None,
) -> None:
    api_base = api_base or os.getenv(ENV_AZURE_ENDPOINT)
    if api_base is not None:
        lm_kwargs["api_base"] = api_base

    api_version = api_version or os.getenv(ENV_AZURE_API_VERSION)
    if api_version is not None:
        lm_kwargs["api_version"] = api_version

    if api_key is None:
        for var in _AZURE_API_KEY_VARS:
            api_key = os.getenv(var)
            if api_key:
                break

    if api_key:
        lm_kwargs["api_key"] = api_key
        return

    ad_token = os.getenv(ENV_AZURE_AD_TOKEN)
    if ad_token:
        lm_kwargs["azure_ad_token"] = ad_token


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
