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
    parser.add_argument("query", help="Natural-language question to answer")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--url",
        default=None,
        help="Override FULLTEXT_SEARCH_API_URL",
    )
    args = parser.parse_args(argv)

    from fulltext_search.engine import SearchEngine

    engine = SearchEngine.from_env(url=args.url, default_limit=args.limit)
    print(f"LM: {engine.model}")
    print(f"API: {getattr(engine.source, 'url', engine.source)}")
    print(f"Indexed documents: {engine.document_count}")
    print(f"Q: {args.query}\n")

    result = engine.ask(args.query, limit=args.limit)
    if result.rewritten_query:
        print(f"Rewritten query: {result.rewritten_query}\n")
    print(f"A: {result.answer}\n")
    print("Supporting hits:")
    for i, hit in enumerate(result.results, start=1):
        title = (hit.document.metadata or {}).get("title", hit.document_id)
        relevance = (hit.metadata or {}).get("relevance")
        rationale = (hit.metadata or {}).get("rationale")
        print(f"  [{i}] {title} ({relevance})")
        if rationale:
            print(f"      {rationale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
