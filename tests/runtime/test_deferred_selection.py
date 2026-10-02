"""从已提交原始工具事实派生本步骤schema，不修改共享registry。"""

import pytest
from fakes import history_snapshot

from iris.exceptions import IrisConfigError
from iris.lifecycle import SessionToolDiscovery
from iris.message import Msg, ToolUseBlock
from iris.runtime._tool_context import select_tool_context
from iris.tools import CallableTool, ToolRegistry


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="eager", description="eager")
    for name in ("a", "b", "c"):
        tool = CallableTool(lambda: "ok", name=name, description=name, deferred=True)
        if name == "b":
            tool.definition.aliases = ("alias_b",)
        registry.register(tool)
    return registry


def _search(*names: str) -> list[Msg]:
    return [
        Msg.assistant([ToolUseBlock(id="search", name="tool_search")]),
        Msg.tool_result(
            tool_use_id="search",
            name="tool_search",
            content="summary",
            metadata={"extra": {"context_revealed_tools": list(names)}},
        ),
    ]


def test_search_first_candidate_protection_consumed_by_final_and_fork_prefix() -> None:
    view = _registry().view()
    raw = _search("b", "a", "c")
    current = select_tool_context(
        view, history_snapshot(raw).tool_discovery, include_tools=True, tool_choice=None
    )
    assert current.names == ("eager", "a", "b", "c")
    assert current.optional_names == ("a", "c")
    consumed = select_tool_context(
        view,
        history_snapshot([*raw, Msg.assistant("done")]).tool_discovery,
        include_tools=True,
        tool_choice=None,
    )
    assert consumed.optional_names == ("b", "a", "c")
    assert select_tool_context(
        view, history_snapshot(raw[:1]).tool_discovery, include_tools=True, tool_choice=None
    ).names == ("eager",)
    assert not view.allow


def test_forced_alias_is_required_and_denied_target_is_configuration_error() -> None:
    registry = _registry()
    choice = {"name": "alias_b"}
    selected = select_tool_context(
        registry.view(), SessionToolDiscovery(), include_tools=True, tool_choice=choice
    )
    assert selected.names == ("eager", "b") and not selected.optional_names
    assert selected.tool_choice["name"] == "b"
    for view in (registry.view(deny={"b"}), registry.view(include_groups={"outside"})):
        with pytest.raises(IrisConfigError):
            select_tool_context(
                view, SessionToolDiscovery(), include_tools=True, tool_choice=choice
            )
    with pytest.raises(IrisConfigError):
        select_tool_context(
            registry.view(),
            SessionToolDiscovery(),
            include_tools=True,
            tool_choice={"name": "missing"},
        )


def test_disabled_tools_and_disabled_disclosure_keep_base_behavior() -> None:
    view = _registry().view()
    for included, choice in ((False, "auto"), (True, "none")):
        selection = select_tool_context(
            view,
            history_snapshot(_search("a")).tool_discovery,
            include_tools=included,
            tool_choice=choice,
        )
        assert selection.names == () and selection.tool_choice is None
    assert select_tool_context(view, None, include_tools=True, tool_choice=None).names == ("eager",)


def test_parallel_searches_protect_each_first_candidate_and_usage_ranks_older_hits() -> None:
    view = _registry().view()
    raw = [
        *_search("a", "c"),
        Msg.tool_result(
            tool_use_id="second",
            name="tool_search",
            content="summary",
            metadata={"extra": {"context_revealed_tools": ["b"]}},
        ),
    ]
    selected = select_tool_context(
        view, history_snapshot(raw).tool_discovery, include_tools=True, tool_choice=None
    )
    assert selected.optional_names == ("c",)
    raw.extend(
        [
            Msg.assistant("done"),
            Msg.tool_result(
                tool_use_id="use",
                name="c",
                content="used",
                metadata={"extra": {"context_tool_name": "c"}},
            ),
        ]
    )
    assert select_tool_context(
        view, history_snapshot(raw).tool_discovery, include_tools=True, tool_choice=None
    ).optional_names == ("b", "c", "a")


def test_persisted_discovery_ranking_uses_absolute_positions_and_filters_base_view() -> None:
    """选择器直接消费投影，以最新search、使用、发现和名称排序。"""
    registry = _registry()
    for name in ("d", "e", "f"):
        registry.register_function(lambda: "ok", name=name, description=name, deferred=True)
    discovery = SessionToolDiscovery(
        discovered_at={"a": 100, "b": 50, "c": 100, "d": 200, "e": 100, "f": 100, "missing": 999},
        used_at={"b": 300},
        latest_search_names=("c", "a"),
        protected_first_names=("c",),
    )
    selected = select_tool_context(
        registry.view(deny={"a"}), discovery, include_tools=True, tool_choice=None
    )
    assert selected.names == ("eager", "b", "c", "d", "e", "f")
    assert selected.optional_names == ("b", "d", "e", "f")
    assert discovery.latest_search_names == ("c", "a")


def test_forced_target_still_works_when_discovery_was_not_requested() -> None:
    """None仅表示未读取发现状态，显式强制目标仍进入必需集合。"""
    selected = select_tool_context(
        _registry().view(),
        None,
        include_tools=True,
        tool_choice={"name": "alias_b"},
    )
    assert selected.names == ("eager", "b")
    assert selected.optional_names == ()
