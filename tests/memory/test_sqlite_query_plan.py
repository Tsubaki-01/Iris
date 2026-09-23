from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    SQLiteMemoryStore,
)


@pytest.mark.parametrize("limit", [0, 101])
@pytest.mark.parametrize(
    "operation",
    [
        lambda store, namespace, limit: store.list_items([namespace], limit=limit),
        lambda store, namespace, limit: store.list_events(namespace, limit=limit),
        lambda store, namespace, limit: store.list_observations(namespace, limit=limit),
    ],
)
def test_list_methods_reject_out_of_range_limits(
    tmp_path: Path,
    operation: Callable[[SQLiteMemoryStore, str, int], object],
    limit: int,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "invalid-limit.db")
    namespace = "project"

    with pytest.raises(IrisMemoryError, match="limit 必须在 1 到 100 之间"):
        operation(store, namespace, limit)
