"""Command-line entry points for fulltext-search."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Ask a question against a paging-API corpus with Corrective RAG."""

    parser = argparse.ArgumentParser(
        prog="fulltext-search-ask",
        description="Query a paging-API corpus with Corrective RAG.",
    )
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        dest="top_k",
        help="Number of top-relevance records to return (must be > 0)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Override FULLTEXT_SEARCH_API_URL",
    )
    args = parser.parse_args(argv)

    from fulltext_search.engine import SearchEngine

    engine = SearchEngine.from_env(url=args.url, default_top_k=args.top_k)
    print(f"LM: {engine.model}")
    print(f"API: {getattr(engine.source, 'url', engine.source)}")
    print(f"Indexed documents: {engine.document_count}")
    print(f"Q: {args.query}\n")

    result = engine.ask(args.query, top_k=args.top_k)
    if result.rewritten_query:
        print(f"Rewritten query: {result.rewritten_query}\n")
    print(f"Top-{args.top_k} records:")
    for i, (hit, record) in enumerate(
        zip(result.results, result.records), start=1
    ):
        title = record.metadata.get("title", record.id)
        relevance = (hit.metadata or {}).get("relevance")
        rationale = (hit.metadata or {}).get("rationale")
        print(f"  [{i}] {title} ({relevance})")
        if rationale:
            print(f"      {rationale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
