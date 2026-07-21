# fulltext-search

Clearsky's Python package for full-text search over arbitrary data sources, with a gRPC API.

## Install from GitHub

```bash
pip install git+https://github.com/cs-ahmed-salem/full-text-search.git
```

Editable / development install:

```bash
git clone https://github.com/cs-ahmed-salem/full-text-search.git
cd full-text-search
pip install -e ".[dev]"
```

Pin a branch, tag, or commit:

```bash
pip install git+https://github.com/cs-ahmed-salem/full-text-search.git@main
```

## Layout

```
src/fulltext_search/
  algorithms/     # search algorithm implementations
  datasources/    # local files, remote endpoints, Postgres
  interfaces/     # gRPC API (proto, server, client)
tests/
  algorithms/
  datasources/
  interfaces/
  performance/
```

## gRPC interface

Start a server (uses the temporary stub algorithm until real ones land):

```bash
fulltext-search-serve --port 50051
```

Or programmatically:

```python
from fulltext_search.algorithms.stub import StubSearchAlgorithm
from fulltext_search.interfaces import FullTextSearchClient, serve

# server process
serve(StubSearchAlgorithm(), port=50051)

# client process
with FullTextSearchClient("localhost:50051") as client:
    client.index([])  # documents
    hits = client.search("query", limit=10)
```

Regenerate stubs after editing the proto:

```bash
python scripts/generate_protos.py
```

## Corrective RAG (DSPy)

`CorrectiveRAGAlgorithm` implements a scalable [Corrective RAG](https://www.meilisearch.com/blog/corrective-rag)
workflow on top of DSPy: map-reduce lexical recall over document pages, LLM
relevance grading, a corrective query rewrite (re-searching the same corpus when
retrieval is weak), and grounded answer generation.

Configure the LLM via environment variables (no secrets in code):

```bash
export FULLTEXT_SEARCH_LM="openai/gpt-4o-mini"   # default
export OPENAI_API_KEY="sk-..."                    # provider key used by dspy.LM
```

Use it programmatically:

```python
from fulltext_search.algorithms import CorrectiveRAGAlgorithm
from fulltext_search.datasources import Document

algo = CorrectiveRAGAlgorithm(batch_size=256, max_workers=8)
algo.index([Document(id="1", content="...")])

# Graded/corrected hits (implements SearchAlgorithm.search)
hits = algo.search("my question", limit=10)

# Full CRAG: grounded answer + supporting hits
result = algo.answer("my question", limit=10)
print(result.answer, result.rewritten_query)
```

It implements `SearchAlgorithm`, so it can be served over gRPC just like the
stub: `serve(CorrectiveRAGAlgorithm(), port=50051)` (the `Search` RPC returns
graded hits; answer generation is a Python API on the algorithm).

## Test

```bash
pytest
pytest -m performance
```
