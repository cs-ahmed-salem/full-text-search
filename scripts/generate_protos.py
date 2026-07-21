#!/usr/bin/env python3
"""Generate gRPC Python stubs from the interfaces proto definitions."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / "src" / "fulltext_search" / "interfaces" / "proto"
OUT_DIR = ROOT / "src" / "fulltext_search" / "interfaces" / "v1"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "__init__.py").touch()

    proto_file = PROTO_DIR / "fulltext_search.proto"
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={OUT_DIR}",
            f"--grpc_python_out={OUT_DIR}",
            str(proto_file),
        ]
    )
    if result != 0:
        return result

    _fix_grpc_imports(OUT_DIR / "fulltext_search_pb2_grpc.py")
    return 0


def _fix_grpc_imports(path: Path) -> None:
    """Rewrite absolute pb2 imports to package-relative imports."""
    text = path.read_text(encoding="utf-8")
    fixed = re.sub(
        r"^import (fulltext_search_pb2) as ",
        r"from fulltext_search.interfaces.v1 import \1 as ",
        text,
        flags=re.MULTILINE,
    )
    if fixed == text:
        fixed = re.sub(
            r"^import (fulltext_search_pb2)$",
            r"from fulltext_search.interfaces.v1 import \1",
            text,
            flags=re.MULTILINE,
        )
    path.write_text(fixed, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
