"""基于 lifecycle store 的会话历史查询与独立分支。

Example:
    history = SessionHistory(store)
    branch = history.fork(source_run_id)
    result = await runner.start(
        AgentRunRequest(input="换一个方向继续", session_id=branch.session_id)
    )
"""

from datetime import UTC, datetime
from uuid import uuid4

from ..lifecycle import (
    ForkPointCursor,
    ForkPointPage,
    ForkSession,
    LifecycleStore,
    RunHistorySnapshot,
    SessionSnapshot,
)


class SessionHistory:
    """绑定 host 管理的 store，提供跨 run 的历史查询与 session 创建。"""

    def __init__(self, store: LifecycleStore) -> None:
        """复用与 runner 相同的 store，不接管其资源生命周期。"""
        self._store = store

    def list_fork_points(
        self,
        session_id: str,
        *,
        after: ForkPointCursor | None = None,
        limit: int = 50,
    ) -> ForkPointPage:
        """按创建时间与 run ID 升序分页查询终态顶层 run。

        Args:
            session_id: 要浏览的会话。
            after: 上一页返回的游标，省略时从头开始。
            limit: 本页条数，正数约束由 store 统一检查。

        Returns:
            分支点列表与下一页游标；不存在或没有合格节点时返回空页。
        """
        return self._store.list_fork_points(session_id, after=after, limit=limit)

    def get_at_run(self, source_run_id: str) -> RunHistorySnapshot:
        """读取指定分支点末尾的已提交历史，不提供当前会话的 CAS revision。"""
        return self._store.load_session_at_run(source_run_id)

    def fork(self, source_run_id: str) -> SessionSnapshot:
        """自动生成新 session ID 并原子复制指定 run 的历史前缀。

        Args:
            source_run_id: 已结束的顶层来源 run。

        Returns:
            独立的新会话，包含自动生成的 ID 和直接来源；尚未创建新 run。

        Notes:
            每次成功调用都创建不同分支；后续运行由 host 使用现有 runner 启动。
            来源资格与持久化错误由 store 原样传递。
        """
        return self._store.fork_session(
            ForkSession(
                source_run_id=source_run_id,
                target_session_id=f"session_{uuid4().hex}",
                now=datetime.now(UTC),
            )
        )
