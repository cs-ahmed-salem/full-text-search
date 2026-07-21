"""Tests for the JSONL file data source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fulltext_search.datasources.base import Document
from fulltext_search.datasources.jsonl import JsonlFileSource


def test_jsonl_reads_single_file(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "1", "content": "alpha", "tag": "a"}),
                json.dumps({"id": "2", "content": "beta"}),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with JsonlFileSource(path) as source:
        docs = list(source.iter_documents())

    assert docs == [
        Document(
            id="1",
            content="alpha",
            metadata={"tag": "a", "path": str(path.resolve()), "line": 1},
        ),
        Document(
            id="2",
            content="beta",
            metadata={"path": str(path.resolve()), "line": 2},
        ),
    ]


def test_jsonl_reads_directory(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"id": "a", "content": "one"}) + "\n",
        encoding="utf-8",
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.jsonl").write_text(
        json.dumps({"id": "b", "content": "two"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "ignore.txt").write_text("nope\n", encoding="utf-8")

    with JsonlFileSource(tmp_path) as source:
        docs = list(source.iter_documents())

    assert {d.id for d in docs} == {"a", "b"}
    assert {d.content for d in docs} == {"one", "two"}


def test_jsonl_generates_id_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text(
        json.dumps({"content": "no id here"}) + "\n",
        encoding="utf-8",
    )

    with JsonlFileSource(path) as source:
        docs = list(source.iter_documents())

    assert docs[0].id == f"{path.resolve()}:1"
    assert docs[0].content == "no id here"


def test_jsonl_custom_field_names(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text(
        json.dumps({"doc_id": "x", "body": "hello", "lang": "en"}) + "\n",
        encoding="utf-8",
    )

    with JsonlFileSource(path, id_field="doc_id", content_field="body") as source:
        docs = list(source.iter_documents())

    assert docs[0].id == "x"
    assert docs[0].content == "hello"
    assert docs[0].metadata["lang"] == "en"


def test_jsonl_requires_connect(tmp_path: Path) -> None:
    source = JsonlFileSource(tmp_path)
    with pytest.raises(RuntimeError, match="not connected"):
        list(source.iter_documents())


def test_jsonl_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        JsonlFileSource(tmp_path / "missing.jsonl").connect()


def test_jsonl_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not-json\n", encoding="utf-8")

    with JsonlFileSource(path) as source:
        with pytest.raises(ValueError, match="Invalid JSON"):
            list(source.iter_documents())


def test_jsonl_skip_invalid(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            [
                "{bad",
                json.dumps({"id": "ok", "content": "good"}),
                '"string-not-object"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with JsonlFileSource(path, skip_invalid=True) as source:
        docs = list(source.iter_documents())

    assert docs == [
        Document(
            id="ok",
            content="good",
            metadata={"path": str(path.resolve()), "line": 2},
        )
    ]


def test_jsonl_missing_content_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps({"id": "1", "text": "x"}) + "\n", encoding="utf-8")

    with JsonlFileSource(path) as source:
        with pytest.raises(ValueError, match="Missing content field"):
            list(source.iter_documents())
