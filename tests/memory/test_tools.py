from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from iris.memory import (
    MemoryAccessPolicy,
    MemoryCategory,
    MemoryConfig,
    MemoryItemKind,
    MemoryScope,
    MemoryService,
    MemoryVisibility,
    MemoryWriteInput,
    SQLiteMemoryStore,
    default_memory_scope_factory,
    register_memory_tools,
)
from iris.message import ToolUseBlock
from iris.tools import ToolCapability, ToolExecutionContext, ToolExecutor, register_file_tools


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


def test_register_memory_tools_extends_file_registry(tmp_path: Path) -> None:
    service = _service(tmp_path)
    registry = register_file_tools()

    returned = register_memory_tools(
        service=service,
        scope_factory=default_memory_scope_factory(MemoryConfig()),
        registry=registry,
    )

    assert returned is registry
    assert [
        tool.definition.name for tool in registry.view(include_groups={"file"}).active_tools
    ] == [
        "read_file",
        "list_files",
        "grep_search",
        "write_file",
        "edit_file",
    ]
    assert [
        tool.definition.name for tool in registry.view(include_groups={"memory"}).active_tools
    ] == [
        "memory_search",
        "memory_list",
        "memory_get",
    ]


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


def test_default_memory_scope_factory_ignores_session_for_agent_scope(
    tmp_path: Path,
) -> None:
    factory = default_memory_scope_factory(MemoryConfig())

    scope = factory(
        ToolExecutionContext(
            workspace_root=tmp_path,
            agent_id="agent",
            session_id="session",
        )
    )

    assert scope.visibility == MemoryVisibility.AGENT
    assert scope.session_id is None


def test_memory_access_policy_exposes_effective_scopes(tmp_path: Path) -> None:
    context = ToolExecutionContext(workspace_root=tmp_path, agent_id="subagent")
    write_scope = _scope(agent_id="subagent", collection="scratch")
    shared_scope = _scope(agent_id="__workspace__", collection="shared")
    policy = MemoryAccessPolicy(
        actor_agent_id="subagent",
        write_scope=write_scope,
        read_scopes=[write_scope, shared_scope],
    )

    assert policy.effective_write_scope(context) == write_scope
    assert policy.effective_read_scopes(context) == [write_scope, shared_scope]


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
async def test_memory_search_reads_only_policy_read_scopes(tmp_path: Path) -> None:
    service = _service(tmp_path)
    own_scope = _scope(agent_id="parent/researcher-1", collection="scratch")
    shared_scope = _scope(agent_id="__workspace__", collection="shared")
    hidden_scope = _scope(agent_id="parent", collection="default")
    own_item = service.remember(
        MemoryWriteInput(scope=own_scope, text="可读的 subagent scratch", reason="seed")
    )
    shared_item = service.remember(
        MemoryWriteInput(scope=shared_scope, text="可读的 workspace shared", reason="seed")
    )
    hidden_item = service.remember(
        MemoryWriteInput(scope=hidden_scope, text="不可读的 parent memory", reason="seed")
    )

    result = await ToolExecutor(
        register_memory_tools(
            service=service,
            access_policy_factory=_policy_factory(own_scope, [own_scope, shared_scope]),
        )
    ).execute_one(
        ToolUseBlock(id="call_1", name="memory_search", input={"query": "可读"}),
        ToolExecutionContext(workspace_root=tmp_path, agent_id="parent/researcher-1"),
    )

    ids = {item["id"] for item in json.loads(result.model_content)["results"]}
    assert {own_item.id, shared_item.id} <= ids
    assert hidden_item.id not in ids


@pytest.mark.asyncio
async def test_agent_memory_is_visible_across_runtime_sessions(tmp_path: Path) -> None:
    service = _service(tmp_path)
    scope_factory = default_memory_scope_factory(MemoryConfig())
    write_context = ToolExecutionContext(
        workspace_root=tmp_path,
        agent_id="agent",
        session_id="session-a",
    )
    read_context = ToolExecutionContext(
        workspace_root=tmp_path,
        agent_id="agent",
        session_id="session-b",
    )
    item = service.remember(
        MemoryWriteInput(
            scope=scope_factory(write_context),
            text="跨会话保留的 agent 记忆",
            reason="test seed",
        )
    )

    result = await ToolExecutor(
        register_memory_tools(service=service, scope_factory=scope_factory)
    ).execute_one(
        ToolUseBlock(id="call_1", name="memory_search", input={"query": "跨会话"}),
        read_context,
    )

    payload = json.loads(result.model_content)
    assert payload["results"][0]["id"] == item.id


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
async def test_memory_list_reads_only_policy_read_scopes(tmp_path: Path) -> None:
    service = _service(tmp_path)
    own_scope = _scope(agent_id="subagent", collection="scratch")
    shared_scope = _scope(agent_id="__workspace__", collection="shared")
    hidden_scope = _scope(agent_id="parent", collection="default")
    own_item = service.remember(
        MemoryWriteInput(
            scope=own_scope,
            text="subagent scratch user memory",
            reason="seed",
            category=MemoryCategory.USER,
        )
    )
    shared_item = service.remember(
        MemoryWriteInput(
            scope=shared_scope,
            text="workspace shared user memory",
            reason="seed",
            category=MemoryCategory.USER,
        )
    )
    hidden_item = service.remember(
        MemoryWriteInput(
            scope=hidden_scope,
            text="parent user memory",
            reason="seed",
            category=MemoryCategory.USER,
        )
    )

    result = await ToolExecutor(
        register_memory_tools(
            service=service,
            access_policy_factory=_policy_factory(own_scope, [own_scope, shared_scope]),
        )
    ).execute_one(
        ToolUseBlock(
            id="list_1",
            name="memory_list",
            input={"category": "user", "limit": 10},
        ),
        ToolExecutionContext(workspace_root=tmp_path, agent_id="subagent"),
    )

    ids = {item["id"] for item in json.loads(result.model_content)["items"]}
    assert ids == {own_item.id, shared_item.id}
    assert hidden_item.id not in ids


@pytest.mark.asyncio
async def test_memory_get_reads_policy_read_scopes(tmp_path: Path) -> None:
    service = _service(tmp_path)
    own_scope = _scope(agent_id="subagent", collection="scratch")
    shared_scope = _scope(agent_id="__workspace__", collection="shared")
    hidden_scope = _scope(agent_id="parent", collection="default")
    shared_item = service.remember(
        MemoryWriteInput(scope=shared_scope, text="workspace shared note", reason="seed")
    )
    hidden_item = service.remember(
        MemoryWriteInput(scope=hidden_scope, text="parent hidden note", reason="seed")
    )
    executor = ToolExecutor(
        register_memory_tools(
            service=service,
            access_policy_factory=_policy_factory(own_scope, [own_scope, shared_scope]),
        )
    )
    context = ToolExecutionContext(workspace_root=tmp_path, agent_id="subagent")

    shared_result = await executor.execute_one(
        ToolUseBlock(id="get_1", name="memory_get", input={"item_id": shared_item.id}),
        context,
    )
    hidden_result = await executor.execute_one(
        ToolUseBlock(id="get_2", name="memory_get", input={"item_id": hidden_item.id}),
        context,
    )

    assert json.loads(shared_result.model_content)["item"]["id"] == shared_item.id
    assert json.loads(hidden_result.model_content) == {"found": False}


@pytest.mark.asyncio
async def test_memory_list_applies_category_before_limit(tmp_path: Path) -> None:
    service = _service(tmp_path)
    scope_factory = default_memory_scope_factory(MemoryConfig())
    context = ToolExecutionContext(workspace_root=tmp_path, agent_id="agent")
    scope = scope_factory(context)
    user_item = service.remember(
        MemoryWriteInput(
            scope=scope,
            text="较早的用户记忆",
            reason="test seed",
            category=MemoryCategory.USER,
        )
    )
    service.remember(
        MemoryWriteInput(
            scope=scope,
            text="较新的任务记忆",
            reason="test seed",
            category=MemoryCategory.TASK,
        )
    )

    result = await ToolExecutor(
        register_memory_tools(service=service, scope_factory=scope_factory)
    ).execute_one(
        ToolUseBlock(
            id="list_1",
            name="memory_list",
            input={"category": "user", "limit": 1},
        ),
        context,
    )

    payload = json.loads(result.model_content)
    assert [item["id"] for item in payload["items"]] == [user_item.id]


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


def _scope(
    *,
    agent_id: str = "agent",
    collection: str = "default",
) -> MemoryScope:
    return MemoryScope(workspace_id="workspace", agent_id=agent_id, collection=collection)


def _policy_factory(
    write_scope: MemoryScope,
    read_scopes: list[MemoryScope],
) -> Callable[[ToolExecutionContext], MemoryAccessPolicy]:
    def _factory(context: ToolExecutionContext) -> MemoryAccessPolicy:
        return MemoryAccessPolicy(
            actor_agent_id=context.agent_id,
            write_scope=write_scope,
            read_scopes=read_scopes,
        )

    return _factory
