from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    MemoryScope,
    SQLiteMemoryStore,
)


@pytest.mark.parametrize("limit", [0, 101])
@pytest.mark.parametrize(
    "operation",
    [
        lambda store, scope, limit: store.list_items(scope, limit=limit),
        lambda store, scope, limit: store.list_events(scope, limit=limit),
        lambda store, scope, limit: store.list_candidates(scope, limit=limit),
    ],
)
def test_list_methods_reject_out_of_range_limits(
    tmp_path: Path,
    operation: Callable[[SQLiteMemoryStore, MemoryScope, int], object],
    limit: int,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "invalid-limit.db", use_fts=False)
    scope = MemoryScope(workspace_id="workspace", agent_id="agent")

    with pytest.raises(IrisMemoryError, match="limit 必须在 1 到 100 之间"):
        operation(store, scope, limit)


def _scope_params(scope: MemoryScope) -> list[str]:
    return [
        scope.workspace_id,
        scope.agent_id,
        scope.collection,
        scope.visibility.value,
        scope.session_id or "",
    ]
