"""Goal 在 harness 边界的运行配置约束。"""

from ..agents import AgentConfig
from ..exceptions import IrisConfigError
from ..lifecycle import AgentRunOptions


def validate_goal_options(config: AgentConfig, options: AgentRunOptions) -> None:
    """在创建、编辑或模型恢复入口检查最终工具策略。"""
    choice = options.runtime.request_options.get("tool_choice", config.model.tool_choice)
    if not options.runtime.include_tools or choice not in (None, "auto"):
        raise IrisConfigError(
            "Goal 自动运行需要 include_tools=true 且 tool_choice 为 auto 或未指定"
        )
