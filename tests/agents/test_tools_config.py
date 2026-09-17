from __future__ import annotations

from pathlib import Path

import pytest

import iris.config as iris_config
from iris.agents import PythonToolsConfig, ToolsConfig, build_tool_registry, load_agent_config
from iris.exceptions import IrisConfigError


def test_build_tool_registry_registers_python_function_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "user_tools.py"
    module_path.write_text(
        '''
def search_notes(query: str) -> str:
    """搜索本地笔记。"""
    return f"search: {query}"
''',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = build_tool_registry(
        ToolsConfig(python=PythonToolsConfig(functions=["user_tools:search_notes"]))
    )

    assert registry.get("search_notes").definition.description == "搜索本地笔记。"


def test_subagent_path_is_parent_relative_and_ordinary_registry_ignores_it(tmp_path: Path) -> None:
    assert ToolsConfig().subagent is None
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "name: parent\nmodel: openai/test\nsystem: parent\n"
        "tools:\n  builtin: [file.read]\n  subagent: catalogs/agents.yaml\n",
        encoding="utf-8",
    )
    config = load_agent_config(config_path)
    assert config.tools.subagent == tmp_path / "catalogs" / "agents.yaml"
    registry = build_tool_registry(config.tools)
    assert registry.get("read_file").name == "read_file"


@pytest.mark.parametrize("names", [["web.search"], ["web.fetch"], ["web.search", "web.fetch"]])
def test_web_builtins_use_configured_key_and_register_only_selected_tools(
    monkeypatch: pytest.MonkeyPatch, names: list[str]
) -> None:
    """Web 凭据沿用集中配置，启用名单不隐式增加其他工具。"""
    monkeypatch.setattr(iris_config, "_config", None)
    monkeypatch.setenv("IRIS_PROVIDER_API_KEYS__TAVILY", "tvly-config-test")
    iris_config.init_config()

    registry = build_tool_registry(ToolsConfig(builtin=names))

    assert {tool.name for tool in registry.view().active_tools} == {
        name.replace(".", "_") for name in names
    }


@pytest.mark.parametrize("name", ["web.search", "web.fetch"])
def test_web_builtins_require_their_own_key(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """没有 Tavily 凭据时，不得使用聊天模型通用 key 代替。"""
    monkeypatch.setattr(iris_config, "_config", None)
    monkeypatch.delenv("IRIS_PROVIDER_API_KEYS__TAVILY", raising=False)
    iris_config.init_config(api_key="llm-only", provider_api_keys={})

    with pytest.raises(IrisConfigError, match="IRIS_PROVIDER_API_KEYS__TAVILY"):
        build_tool_registry(ToolsConfig(builtin=[name]))


def test_file_builtin_does_not_require_initialized_global_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未启用 Web 时，普通 builtin 构造不读取全局配置。"""
    monkeypatch.setattr(iris_config, "_config", None)

    registry = build_tool_registry(ToolsConfig(builtin=["file.read"]))

    assert registry.get("read_file").name == "read_file"
