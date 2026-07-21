"""Remote HTTP endpoint data source."""

from __future__ import annotations

from typing import Any, Iterator

import httpx

from fulltext_search.datasources.base import DataSource, Document


class RemoteEndpointSource(DataSource):
    """Fetch documents from a remote HTTP(S) endpoint.

    Expects a JSON response that is either:

    * a list of objects with ``id`` and ``content`` keys, or
    * an object with a ``documents`` list of the same shape.
    """

    def __init__(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        timeout: float = 30.0,
        id_field: str = "id",
        content_field: str = "content",
        documents_key: str = "documents",
    ) -> None:
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
        self.params = params
        self.json_body = json_body
        self.timeout = timeout
        self.id_field = id_field
        self.content_field = content_field
        self.documents_key = documents_key
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
        if self._client is None:
            raise RuntimeError(
                "RemoteEndpointSource is not connected; call connect()"
            )

        response = self._client.request(
            self.method,
            self.url,
            params=self.params,
            json=self.json_body,
        )
        response.raise_for_status()
        payload = response.json()
        records = self._extract_records(payload)

        for record in records:
            doc_id = str(record[self.id_field])
            content = str(record[self.content_field])
            metadata = {
                key: value
                for key, value in record.items()
                if key not in {self.id_field, self.content_field}
            }
            metadata["source_url"] = self.url
            yield Document(id=doc_id, content=content, metadata=metadata)

    def _extract_records(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if self.documents_key in payload:
                records = payload[self.documents_key]
                if not isinstance(records, list):
                    raise ValueError(
                        f"Expected '{self.documents_key}' to be a list, "
                        f"got {type(records).__name__}"
                    )
                return records
            if self.id_field in payload and self.content_field in payload:
                return [payload]
        raise ValueError(
            "Remote payload must be a list of documents, an object with a "
            f"'{self.documents_key}' list, or a single document object"
        )
