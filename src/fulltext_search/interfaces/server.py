"""gRPC server helpers for the full-text search interface."""

from __future__ import annotations

from concurrent import futures
from typing import Any

import grpc

from fulltext_search.algorithms.base import SearchAlgorithm
from fulltext_search.interfaces.servicer import FullTextSearchServicer
from fulltext_search.interfaces.v1 import fulltext_search_pb2_grpc as pb2_grpc


def create_server(
    algorithm: SearchAlgorithm,
    *,
    host: str = "[::]",
    port: int = 50051,
    max_workers: int = 10,
    engine: Any | None = None,
) -> grpc.Server:
    """Build a gRPC server bound to ``host:port``.

    Pass ``engine`` (a :class:`~fulltext_search.engine.SearchEngine` or any
    object with ``search_all`` / ``get_search_page``) to enable the pageable
    brute-force RPCs.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc.add_FullTextSearchServicer_to_server(
        FullTextSearchServicer(algorithm, engine=engine),
        server,
    )
    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind gRPC server on {host}:{port}")
    return server


def serve(
    algorithm: SearchAlgorithm,
    *,
    host: str = "[::]",
    port: int = 50051,
    max_workers: int = 10,
    engine: Any | None = None,
) -> None:
    """Start a blocking gRPC server until interrupted."""
    server = create_server(
        algorithm,
        host=host,
        port=port,
        max_workers=max_workers,
        engine=engine,
    )
    server.start()
    print(f"fulltext-search gRPC listening on {host}:{port}")
    server.wait_for_termination()
