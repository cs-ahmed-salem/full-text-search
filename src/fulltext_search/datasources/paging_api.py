"""Paging HTTP GET API data source."""

from __future__ import annotations

from typing import Any, Iterator

import httpx

from fulltext_search.datasources.base import DataSource, Document


class PagingApiSource(DataSource):
    """Fetch documents from a page-oriented HTTP GET API.

    Accepts arbitrary query ``params`` and request ``headers`` in the
    datasource configuration. On each request the source overlays
    ``page_param`` / ``size_param`` (defaults: ``page`` / ``size``) and walks
    pages until an empty items list or a truthy ``last`` flag.

    Designed for Spring Data-style responses::

        {
          "content": [{"id": "...", "description": "...", ...}, ...],
          "last": false,
          "number": 0,
          ...
        }

    Also implements native :meth:`iter_pages` so map-reduce retrieval can page
    at the HTTP layer instead of buffering the full stream.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        page_size: int = 100,
        start_page: int = 0,
        page_param: str = "page",
        size_param: str = "size",
        items_key: str = "content",
        last_key: str = "last",
        id_field: str = "id",
        content_field: str = "description",
        timeout: float = 30.0,
    ) -> None:
        if page_size < 1:
            raise ValueError("page_size must be >= 1")
        if start_page < 0:
            raise ValueError("start_page must be >= 0")

        self.url = url
        self.headers = dict(headers or {})
        self.params = dict(params or {})
        self.page_size = page_size
        self.start_page = start_page
        self.page_param = page_param
        self.size_param = size_param
        self.items_key = items_key
        self.last_key = last_key
        self.id_field = id_field
        self.content_field = content_field
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def connect(self) -> None:
        if self._client is None:
            self._client = httpx.Client(
                headers=self.headers,
                timeout=self.timeout,
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def iter_documents(self) -> Iterator[Document]:
        for page in self.iter_pages(self.page_size):
            yield from page

    def iter_pages(self, page_size: int) -> Iterator[list[Document]]:
        """Yield API pages as lists of :class:`Document`.

        ``page_size`` is sent as the HTTP size parameter so paging happens on
        the server.
        """
        if self._client is None:
            raise RuntimeError(
                "PagingApiSource is not connected; call connect()"
            )
        if page_size < 1:
            raise ValueError("page_size must be >= 1")

        page_number = self.start_page
        while True:
            payload = self._fetch_page(page_number, page_size)
            records = self._extract_records(payload)
            documents = [self._to_document(record) for record in records]
            if documents:
                yield documents

            if not records or self._is_last_page(payload):
                break
            page_number += 1

    def _fetch_page(self, page_number: int, page_size: int) -> Any:
        assert self._client is not None
        request_params = {
            **self.params,
            self.page_param: page_number,
            self.size_param: page_size,
        }
        response = self._client.get(
            self.url,
            params=request_params,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def _extract_records(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if self.items_key not in payload:
                raise ValueError(
                    f"Expected paging payload to contain '{self.items_key}', "
                    f"got keys {sorted(payload)}"
                )
            records = payload[self.items_key]
            if not isinstance(records, list):
                raise ValueError(
                    f"Expected '{self.items_key}' to be a list, "
                    f"got {type(records).__name__}"
                )
            return records
        raise ValueError(
            "Paging payload must be a list of documents or an object with an "
            f"'{self.items_key}' list"
        )

    def _is_last_page(self, payload: Any) -> bool:
        if isinstance(payload, dict) and self.last_key in payload:
            return bool(payload[self.last_key])
        return False

    def _to_document(self, record: dict[str, Any]) -> Document:
        if self.id_field not in record:
            raise KeyError(
                f"Record missing id field {self.id_field!r}: {sorted(record)}"
            )
        if self.content_field not in record:
            raise KeyError(
                f"Record missing content field {self.content_field!r}: "
                f"{sorted(record)}"
            )

        doc_id = str(record[self.id_field])
        content = str(record[self.content_field])
        metadata = {
            key: value
            for key, value in record.items()
            if key not in {self.id_field, self.content_field}
        }
        metadata["source_url"] = self.url
        return Document(id=doc_id, content=content, metadata=metadata)
