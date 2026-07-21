"""CLI entrypoint: ``fulltext-search-serve``."""

from __future__ import annotations

import argparse

from fulltext_search.algorithms.stub import StubSearchAlgorithm
from fulltext_search.interfaces.server import serve


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Serve full-text search over gRPC",
    )
    parser.add_argument("--host", default="[::]", help="Bind host")
    parser.add_argument("--port", type=int, default=50051, help="Bind port")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Thread pool size",
    )
    args = parser.parse_args(argv)

    serve(
        StubSearchAlgorithm(),
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
