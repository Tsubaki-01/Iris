"""记忆 scope 的内部展示 helper。"""

from .models import MemoryScope


def scope_summary(scope: MemoryScope) -> str:
    """生成稳定可读的 scope 摘要。"""
    parts = [
        f"workspace={scope.workspace_id}",
        f"agent={scope.agent_id}",
        f"collection={scope.collection}",
        f"visibility={scope.visibility.value}",
    ]
    if scope.session_id:
        parts.append(f"session={scope.session_id}")
    return ", ".join(parts)
