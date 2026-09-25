"""从静态目录与会话原始发现事实派生本步骤工具集合。"""

from dataclasses import dataclass
from typing import Any

from ..exceptions import IrisConfigError, IrisToolNotFoundError
from ..message import Msg, Role
from ..tools import ToolRegistryView


@dataclass(frozen=True, slots=True)
class ToolContextSelection:
    """注册顺序的工具集合与允许按逆序撤下的候选。"""

    names: tuple[str, ...]
    optional_names: tuple[str, ...]
    tool_choice: str | dict[str, Any] | None


def select_tool_context(
    view: ToolRegistryView,
    messages: list[Msg],
    *,
    deferred_tools: bool,
    include_tools: bool,
    tool_choice: str | dict[str, Any] | None,
) -> ToolContextSelection:
    """保持 base 过滤，按已提交 search/使用顺序选择完整 schema 候选。"""
    if not include_tools or tool_choice == "none":
        return ToolContextSelection((), (), None)
    available = {tool.name: tool for tool in view.available_tools}
    required = {tool.name for tool in view.active_tools}
    if isinstance(tool_choice, dict):
        target = tool_choice["function"]["name"]
        try:
            canonical = view.get(target).name
        except IrisToolNotFoundError as exc:
            raise IrisConfigError("tool_choice 指定的工具不存在", tool_name=target) from exc
        if canonical not in available:
            raise IrisConfigError("tool_choice 指定的工具被 base view 排除", tool_name=target)
        required.add(canonical)
        tool_choice = {**tool_choice, "function": {**tool_choice["function"], "name": canonical}}
    discovered: dict[str, int] = {}
    used: dict[str, int] = {}
    latest: tuple[str, ...] = ()
    protected: set[str] = set()
    if deferred_tools:
        for index, message in enumerate(messages):
            if message.role is Role.ASSISTANT:
                protected.clear()
            for result in message.tool_results:
                if result.is_error:
                    continue
                facts = result.metadata.get("extra", {})
                name = facts.get("context_tool_name")
                if name is not None:
                    used[name] = index
                if "context_revealed_tools" in facts:
                    latest = tuple(facts["context_revealed_tools"])
                    discovered.update((name, index) for name in latest)
                    if latest:
                        protected.add(latest[0])
    ranked = sorted(
        (name for name in discovered if name in available and name not in required),
        key=lambda name: (
            0 if name in latest else 1,
            latest.index(name) if name in latest else 0,
            -used.get(name, -1),
            -discovered[name],
            name,
        ),
    )
    selected = required | set(ranked)
    return ToolContextSelection(
        tuple(name for name in available if name in selected),
        tuple(name for name in ranked if name not in protected),
        tool_choice,
    )
