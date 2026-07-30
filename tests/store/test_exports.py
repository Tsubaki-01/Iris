from __future__ import annotations

from iris.store import InMemoryLifecycleStore, SQLiteStore


def test_store_package_exports_only_lifecycle_concrete_stores() -> None:
    import iris.store as store_package

    assert store_package.__all__ == ["InMemoryLifecycleStore", "SQLiteStore"]
    assert InMemoryLifecycleStore.__name__ == "InMemoryLifecycleStore"
    assert SQLiteStore.__name__ == "SQLiteStore"
    assert not hasattr(store_package, "SQLiteSessionStore")
