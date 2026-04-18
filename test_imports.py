#!/usr/bin/env python3
"""Import 스모크 테스트.

프로젝트가 의존하는 라이브러리와 내부 모듈이 정상적으로 임포트되는지 확인한다.
`python test_imports.py` 또는 `pytest test_imports.py`로 실행 가능.
"""

import importlib
import sys
import traceback

EXTERNAL_MODULES = [
    "torch",
    "flask",
    "werkzeug",
    "fitz",
    "faiss",
    "sentence_transformers",
    "transformers",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_text_splitters",
    "duckduckgo_search",
    "bs4",
    "requests",
]

INTERNAL_MODULES = [
    "config",
    "logging_setup",
    "pdf_processor",
    "vector_store",
    "search_engine",
    "llm_model",
    "rag_system",
]


def _try_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        print(f"OK  {name}")
        return True
    except Exception as e:  # pragma: no cover - 스모크 테스트
        print(f"FAIL {name}: {e}")
        traceback.print_exc()
        return False


def test_external_imports():
    failures = [m for m in EXTERNAL_MODULES if not _try_import(m)]
    assert not failures, f"외부 모듈 import 실패: {failures}"


def test_internal_imports():
    failures = [m for m in INTERNAL_MODULES if not _try_import(m)]
    assert not failures, f"내부 모듈 import 실패: {failures}"


def main() -> int:
    print("외부 의존성:")
    ext_failed = [m for m in EXTERNAL_MODULES if not _try_import(m)]
    print("\n내부 모듈:")
    int_failed = [m for m in INTERNAL_MODULES if not _try_import(m)]

    if ext_failed or int_failed:
        print(f"\n실패: 외부={ext_failed}, 내부={int_failed}")
        return 1
    print("\n모든 import 성공")
    return 0


if __name__ == "__main__":
    sys.exit(main())
