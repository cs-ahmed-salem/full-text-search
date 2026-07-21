"""JSON Lines (JSONL) file data source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from fulltext_search.datasources.base import DataSource, Document


class JsonlFileSource(DataSource):
    """Read documents from a JSONL file or directory of JSONL files.

    Each non-empty line must be a JSON object. By default ``id`` and
    ``content`` map to :class:`Document` fields; remaining keys become
    metadata. When ``id`` is missing, the document id is
    ``{path}:{line_number}``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        pattern: str = "**/*.jsonl",
        encoding: str = "utf-8",
        errors: str = "strict",
        id_field: str = "id",
        content_field: str = "content",
        skip_invalid: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.pattern = pattern
        self.encoding = encoding
        self.errors = errors
        self.id_field = id_field
        self.content_field = content_field
        self.skip_invalid = skip_invalid
        self._connected = False

    def connect(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def iter_documents(self) -> Iterator[Document]:
        if not self._connected:
            raise RuntimeError("JsonlFileSource is not connected; call connect()")

        for file_path in self._iter_files():
            yield from self._iter_file(file_path)

    def _iter_files(self) -> Iterator[Path]:
        if self.path.is_file():
            yield self.path
            return

        for file_path in sorted(
            p for p in self.path.glob(self.pattern) if p.is_file()
        ):
            yield file_path

    def _iter_file(self, file_path: Path) -> Iterator[Document]:
        with file_path.open(encoding=self.encoding, errors=self.errors) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if self.skip_invalid:
                        continue
                    raise ValueError(
                        f"Invalid JSON in {file_path} at line {line_number}: {exc}"
                    ) from exc

                if not isinstance(record, dict):
                    if self.skip_invalid:
                        continue
                    raise ValueError(
                        f"Expected a JSON object in {file_path} at line "
                        f"{line_number}, got {type(record).__name__}"
                    )

                yield self._record_to_document(record, file_path, line_number)

    def _record_to_document(
        self,
        record: dict[str, Any],
        file_path: Path,
        line_number: int,
    ) -> Document:
        if self.content_field not in record:
            raise ValueError(
                f"Missing content field '{self.content_field}' in {file_path} "
                f"at line {line_number}"
            )

        if self.id_field in record:
            doc_id = str(record[self.id_field])
        else:
            doc_id = f"{file_path}:{line_number}"

        metadata = {
            key: value
            for key, value in record.items()
            if key not in {self.id_field, self.content_field}
        }
        metadata["path"] = str(file_path)
        metadata["line"] = line_number

        return Document(
            id=doc_id,
            content=str(record[self.content_field]),
            metadata=metadata,
        )
