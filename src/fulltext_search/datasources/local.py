"""Local filesystem data source."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from fulltext_search.datasources.base import DataSource, Document


class LocalFileSource(DataSource):
    """Read documents from local files or directories.

    Each matching file becomes a :class:`Document` whose ``id`` is the
    absolute path and whose ``content`` is the file text.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        pattern: str = "**/*",
        encoding: str = "utf-8",
        errors: str = "replace",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.pattern = pattern
        self.encoding = encoding
        self.errors = errors
        self._connected = False

    def connect(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Path does not exist: {self.root}")
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def iter_documents(self) -> Iterator[Document]:
        if not self._connected:
            raise RuntimeError("LocalFileSource is not connected; call connect()")

        paths: list[Path]
        if self.root.is_file():
            paths = [self.root]
        else:
            paths = sorted(p for p in self.root.glob(self.pattern) if p.is_file())

        for path in paths:
            content = path.read_text(encoding=self.encoding, errors=self.errors)
            yield Document(
                id=str(path),
                content=content,
                metadata={
                    "path": str(path),
                    "name": path.name,
                    "suffix": path.suffix,
                    "size": path.stat().st_size,
                },
            )
