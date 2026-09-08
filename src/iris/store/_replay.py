"""两个 store 共用的精确命令 key 与最小重试描述。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..lifecycle.store import RunCommit
from ._serialization import jsonable


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    """只记录重载当前事实所需的 identity，不保留旧 aggregate。"""

    run_id: str
    includes_session_revision: bool
    interaction_id: str | None

    @classmethod
    def from_commit(cls, commit: RunCommit) -> ReplayRecord:
        """从已提交回执投影最小重试描述。"""
        return cls(
            run_id=commit.run.run_id,
            includes_session_revision=commit.session_revision is not None,
            interaction_id=(
                commit.interaction.interaction_id if commit.interaction is not None else None
            ),
        )


def replay_key(operation: str, command: object) -> str:
    """编码完整命令，保留原有 exact-command 重试语义。"""
    return f"{operation}:{json.dumps(jsonable(command), allow_nan=False, sort_keys=True)}"
