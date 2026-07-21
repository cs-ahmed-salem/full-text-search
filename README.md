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

## Test

```bash
pytest
pytest -m performance
```
