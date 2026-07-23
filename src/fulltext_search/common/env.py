"""Environment loading and LLM credential helpers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

AZURE_TOKEN_RESOURCE = "https://cognitiveservices.azure.com"

_API_KEY_VARS = (
    "FULLTEXT_SEARCH_LM_API_KEY",
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_KEY",
)


def load_environment(env_path: str | Path | None = None) -> Path | None:
    """Load a ``.env`` file into ``os.environ`` if it exists.

    When ``env_path`` is omitted, looks for ``.env`` in the current working
    directory and then walks parents. Returns the path that was loaded, or
    ``None`` when no file was found.
    """

    path = _resolve_env_path(env_path)
    if path is None:
        return None

    try:
        from dotenv import load_dotenv
    except ImportError:
        _parse_dotenv(path)
    else:
        load_dotenv(path)
    return path


def _resolve_env_path(env_path: str | Path | None) -> Path | None:
    if env_path is not None:
        path = Path(env_path).expanduser()
        return path if path.is_file() else None

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        path = candidate / ".env"
        if path.is_file():
            return path
    return None


def _parse_dotenv(env_path: Path) -> None:
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(
            key.strip(), value.strip().strip('"').strip("'")
        )


def ensure_llm_ready(*, mint_azure_token: bool = True) -> str:
    """Ensure an LM can be configured; return the resolved model id.

    Loads ``.env`` first. For Azure models without an API key, optionally mints
    an Azure AD token via the Azure CLI (``az``) and sets ``AZURE_AD_TOKEN``.
    """

    load_environment()
    from fulltext_search.common.llms import default_model

    model = default_model()
    has_key = any(os.getenv(var) for var in _API_KEY_VARS)

    if model.startswith("azure/"):
        if has_key or os.getenv("AZURE_AD_TOKEN"):
            return model
        if not mint_azure_token:
            raise RuntimeError(
                f"No API key or AZURE_AD_TOKEN found for model {model!r}."
            )
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
        os.environ.setdefault("FULLTEXT_SEARCH_LM_MAX_TOKENS", "16000")
        return model

    if not has_key:
        raise RuntimeError(
            f"No API key found for model {model!r}; set one of "
            f"{', '.join(_API_KEY_VARS)}."
        )
    return model
