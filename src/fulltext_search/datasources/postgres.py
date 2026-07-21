"""PostgreSQL table data source."""

from __future__ import annotations

from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from fulltext_search.datasources.base import DataSource, Document


class PostgresTableSource(DataSource):
    """Read documents from a PostgreSQL table or SQL query.

    Provide either ``table`` (with optional ``schema``) or a full ``query``.
    Row values from ``id_column`` and ``content_column`` map to
    :class:`Document` fields; remaining columns become metadata.
    """

    def __init__(
        self,
        connection_string: str,
        *,
        table: str | None = None,
        schema: str = "public",
        query: str | None = None,
        id_column: str = "id",
        content_column: str = "content",
        query_params: dict[str, Any] | None = None,
    ) -> None:
        if (table is None) == (query is None):
            raise ValueError("Provide exactly one of 'table' or 'query'")

        self.connection_string = connection_string
        self.table = table
        self.schema = schema
        self.query = query
        self.id_column = id_column
        self.content_column = content_column
        self.query_params = query_params or {}
        self._conn: psycopg.Connection[Any] | None = None

    def connect(self) -> None:
        if self._conn is None:
            self._conn = psycopg.connect(
                self.connection_string,
                row_factory=dict_row,
            )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def iter_documents(self) -> Iterator[Document]:
        if self._conn is None:
            raise RuntimeError(
                "PostgresTableSource is not connected; call connect()"
            )

        sql = self.query or self._table_query()
        with self._conn.cursor() as cursor:
            cursor.execute(sql, self.query_params)
            for row in cursor:
                doc_id = str(row[self.id_column])
                content = str(row[self.content_column])
                metadata = {
                    key: value
                    for key, value in row.items()
                    if key not in {self.id_column, self.content_column}
                }
                if self.table is not None:
                    metadata.setdefault("table", self.table)
                    metadata.setdefault("schema", self.schema)
                yield Document(id=doc_id, content=content, metadata=metadata)

    def _table_query(self) -> str:
        # Identifiers are validated to avoid SQL injection via table/schema names.
        schema = _quote_ident(self.schema)
        table = _quote_ident(self.table or "")
        id_col = _quote_ident(self.id_column)
        content_col = _quote_ident(self.content_column)
        return f"SELECT * FROM {schema}.{table} ORDER BY {id_col}, {content_col}"


def _quote_ident(name: str) -> str:
    if not name.isidentifier():
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'
