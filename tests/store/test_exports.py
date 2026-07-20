from __future__ import annotations

import importlib.util

import iris.session as session_package


def test_sqlite_store_is_exported_only_from_store_package() -> None:
    assert importlib.util.find_spec("iris.store") is not None

    from iris.store import SQLiteStore

    assert SQLiteStore.__name__ == "SQLiteStore"
    assert "SQLiteSessionStore" not in session_package.__all__
    assert not hasattr(session_package, "SQLiteSessionStore")
