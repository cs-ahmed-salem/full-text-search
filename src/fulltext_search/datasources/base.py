"""Base types for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class Document:
    """A unit of content that can be indexed and searched."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """Abstract base class for document providers.

    Subclasses connect to a backing store and yield :class:`Document`
    instances for indexing.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection or open resources."""

    @abstractmethod
    def close(self) -> None:
        """Release resources opened by :meth:`connect`."""

    @abstractmethod
    def iter_documents(self) -> Iterator[Document]:
        """Yield documents from the source."""

    def __enter__(self) -> DataSource:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
