"""从静态目录与会话原始发现事实派生本步骤工具集合。"""

from dataclasses import dataclass

from ..exceptions import IrisConfigError, IrisToolNotFoundError
from ..lifecycle import SessionToolDiscovery
from ..message import ToolChoice
from ..tools import ToolRegistryView


@dataclass(frozen=True, slots=True)
class ToolContextSelection:
    """注册顺序的工具集合与允许按逆序撤下的候选。"""

    names: tuple[str, ...]
    optional_names: tuple[str, ...]
    tool_choice: ToolChoice | None


def select_tool_context(
    view: ToolRegistryView,
    discovery: SessionToolDiscovery | None,
    *,
    include_tools: bool,
    tool_choice: ToolChoice | None,
) -> ToolContextSelection:
    """保持 base 过滤，按已提交 search/使用顺序选择完整 schema 候选。"""
    if not include_tools or tool_choice == "none":
        return ToolContextSelection((), (), None)
    available = {tool.name: tool for tool in view.available_tools}
    required = {tool.name for tool in view.active_tools}
    if isinstance(tool_choice, dict):
        target = tool_choice["name"]
        try:
            canonical = view.get(target).name
        except IrisToolNotFoundError as exc:
            raise IrisConfigError("tool_choice 指定的工具不存在", tool_name=target) from exc
        if canonical not in available:
            raise IrisConfigError("tool_choice 指定的工具被 base view 排除", tool_name=target)
        required.add(canonical)
        tool_choice = {"name": canonical}
    discovered = discovery.discovered_at if discovery is not None else {}
    used = discovery.used_at if discovery is not None else {}
    latest = discovery.latest_search_names if discovery is not None else ()
    protected = discovery.protected_first_names if discovery is not None else ()
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
