from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris.memory import (
    MemoryCategory,
    MemoryConfig,
    MemoryItemKind,
    MemoryService,
    MemoryVisibility,
    MemoryWriteInput,
    SQLiteMemoryStore,
    default_memory_scope_factory,
    register_memory_tools,
)
from iris.message import ToolUseBlock
from iris.tools import ToolCapability, ToolExecutionContext, ToolExecutor


def test_register_memory_tools_exposes_read_tools_only(tmp_path: Path) -> None:
    service = _service(tmp_path)
    registry = register_memory_tools(
        service=service,
        scope_factory=default_memory_scope_factory(MemoryConfig()),
    )

    tools = registry.view(include_groups={"memory"}).active_tools

    assert [tool.name for tool in tools] == ["memory_search", "memory_list", "memory_get"]
    assert all(tool.definition.capabilities == {ToolCapability.READ} for tool in tools)
    assert "memory_remember" not in {tool.name for tool in tools}
    assert "memory_forget" not in {tool.name for tool in tools}


def test_default_memory_scope_factory_uses_context_and_config(tmp_path: Path) -> None:
    config = MemoryConfig(scope={"collection": "notes", "visibility": "session"})
    factory = default_memory_scope_factory(config)

    scope = factory(
        ToolExecutionContext(
            workspace_root=tmp_path,
            agent_id="agent",
            session_id="session",
        )
    )

    assert scope.workspace_id == str(tmp_path.resolve())
    assert scope.agent_id == "agent"
    assert scope.collection == "notes"
    assert scope.visibility == MemoryVisibility.SESSION
    assert scope.session_id == "session"


@pytest.mark.asyncio
async def test_memory_search_executes_through_tool_executor(tmp_path: Path) -> None:
    service = _service(tmp_path)
    scope_factory = default_memory_scope_factory(MemoryConfig())
    context = ToolExecutionContext(workspace_root=tmp_path, agent_id="agent")
    item = service.remember(
        MemoryWriteInput(
            scope=scope_factory(context),
            text="用户偏好简洁中文回答",
            reason="test seed",
            kind=MemoryItemKind.PREFERENCE,
        )
    )

    result = await ToolExecutor(
        register_memory_tools(service=service, scope_factory=scope_factory)
    ).execute_one(
        ToolUseBlock(id="call_1", name="memory_search", input={"query": "简洁"}),
        context,
    )

    payload = json.loads(result.model_content)
    assert result.is_error is False
    assert payload["results"][0]["id"] == item.id
    assert payload["results"][0]["text"] == "用户偏好简洁中文回答"


@pytest.mark.asyncio
async def test_memory_list_and_get_are_scope_isolated(tmp_path: Path) -> None:
    service = _service(tmp_path)
    scope_factory = default_memory_scope_factory(MemoryConfig())
    owner_context = ToolExecutionContext(workspace_root=tmp_path, agent_id="agent-a")
    other_context = ToolExecutionContext(workspace_root=tmp_path, agent_id="agent-b")
    item = service.remember(
        MemoryWriteInput(
            scope=scope_factory(owner_context),
            text="只属于 agent-a 的记忆",
            reason="test seed",
            category=MemoryCategory.USER,
        )
    )
    executor = ToolExecutor(register_memory_tools(service=service, scope_factory=scope_factory))

    listed = await executor.execute_one(
        ToolUseBlock(id="list_1", name="memory_list", input={}),
        owner_context,
    )
    found = await executor.execute_one(
        ToolUseBlock(id="get_1", name="memory_get", input={"item_id": item.id}),
        owner_context,
    )
    hidden = await executor.execute_one(
        ToolUseBlock(id="get_2", name="memory_get", input={"item_id": item.id}),
        other_context,
    )

    assert json.loads(listed.model_content)["items"][0]["id"] == item.id
    assert json.loads(found.model_content)["item"]["id"] == item.id
    assert json.loads(hidden.model_content) == {"found": False}


@pytest.mark.asyncio
async def test_memory_tool_input_cannot_override_scope(tmp_path: Path) -> None:
    service = _service(tmp_path)
    result = await ToolExecutor(
        register_memory_tools(
            service=service,
            scope_factory=default_memory_scope_factory(MemoryConfig()),
        )
    ).execute_one(
        ToolUseBlock(
            id="call_1",
            name="memory_search",
            input={"query": "简洁", "workspace_id": "escape"},
        ),
        ToolExecutionContext(workspace_root=tmp_path, agent_id="agent"),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_memory_get_rejects_empty_item_id(tmp_path: Path) -> None:
    service = _service(tmp_path)
    result = await ToolExecutor(
        register_memory_tools(
            service=service,
            scope_factory=default_memory_scope_factory(MemoryConfig()),
        )
    ).execute_one(
        ToolUseBlock(id="call_1", name="memory_get", input={"item_id": " "}),
        ToolExecutionContext(workspace_root=tmp_path, agent_id="agent"),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "VALIDATION_ERROR"


def _service(tmp_path: Path) -> MemoryService:
    return MemoryService(SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False))
