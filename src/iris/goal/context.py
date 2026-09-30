"""组合宿主动态上下文与本 Run 对应的当前 Goal 投影。"""

from pathlib import Path
from typing import Any

from ..context import ContextBuildScope, ContextContribution, ContextSnapshot, ContextSource
from ..exceptions import IrisConfigError, IrisGoalError, IrisTemplateError
from ..utils.templating import TemplateRenderer
from .models import GoalSnapshot
from .service import GoalService

_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
_RENDERER = TemplateRenderer()


def _render(name: str, variables: dict[str, Any]) -> str:
    """在 Goal 边界一次归一化模板来源错误。"""
    try:
        return _RENDERER.render_file(_PROMPTS / name, variables)
    except IrisTemplateError as exc:
        raise IrisGoalError("Goal 提示模板渲染失败", template=name, error=str(exc)) from exc


def render_continuation(goal: GoalSnapshot) -> str:
    """生成交给 Runner 的自动输入，具体来源标记由提交端完成。"""
    return _render("goal_continuation.j2", {"goal": goal})


class GoalContextSource:
    """每模型步采集宿主一次，再追加 required Goal contribution。"""

    def __init__(self, service: GoalService, host_source: ContextSource | None = None) -> None:
        """保留原宿主 source，组合时不改变其贡献或调用频率。"""
        self.service = service
        self.host_source = host_source

    async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
        """依据 Run 绑定读取最新目标；旧 Run 不接收替换目标的正文。"""
        host = (
            ContextSnapshot() if self.host_source is None else await self.host_source.collect(scope)
        )
        if any(item.key == "iris.goal" for item in host.contributions):
            raise IrisConfigError("宿主 context source 使用了 Goal 保留 key: iris.goal")
        binding = self.service.store.get_goal_run(scope.run_id)
        goal = self.service.get_current(scope.session_id)
        text = _render(
            "goal_context.j2",
            {
                "goal": goal
                if binding is not None and goal is not None and binding.goal_id == goal.goal_id
                else None,
                "bound": binding is not None,
            },
        )
        return ContextSnapshot(host.contributions + (ContextContribution("iris.goal", text),))


__all__ = ["GoalContextSource", "render_continuation"]
