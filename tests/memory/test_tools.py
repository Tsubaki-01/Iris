"""记忆工具的项目共享、联合读取、显式写入和真实执行结果。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    MEMORY_TOOL_CLASSES,
    FileMemoryMirror,
    MemoryAccessPolicy,
    MemoryActor,
    MemoryConfig,
    MemoryGetTool,
    MemoryGetToolInput,
    MemoryListTool,
    MemoryListToolInput,
    MemorySearchTool,
    MemorySearchToolInput,
    MemoryService,
    MemorySourceType,
    MemoryWriteInput,
    SQLiteMemoryStore,
    default_memory_access_policy_factory,
    register_memory_tools,
)
from iris.message import ToolUseBlock
from iris.tools import ToolCapability, ToolExecutionContext, ToolExecutor
from iris.tools.permissions import DefaultPermissionPolicy


def _context(tmp_path: Path, agent_id: str = "agent") -> ToolExecutionContext:
    return ToolExecutionContext(workspace_root=tmp_path, agent_id=agent_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("namespaces", [("private", "project"), ("project", "private")])
async def test_search_ranks_all_read_namespaces_before_limiting(
    tmp_path: Path, namespaces: tuple[str, str]
) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "rank.db"))
    service.remember(
        MemoryWriteInput(namespace="private", text="needle " + "background " * 40, reason="test")
    )
    relevant = service.remember(
        MemoryWriteInput(namespace="project", text="needle needle needle", reason="test")
    )
    tool = MemorySearchTool(
        service=service,
        access_policy_factory=lambda _: MemoryAccessPolicy(read_namespaces=namespaces),
    )

    result = await tool.arun(MemorySearchToolInput(query="needle", limit=1), _context(tmp_path))

    payload = json.loads(result.content[0].text)
    assert [item["id"] for item in payload["results"]] == [relevant.id]
    assert payload["results"][0]["namespace"] == "project"


@pytest.mark.asyncio
async def test_get_and_list_use_the_bound_namespaces(tmp_path: Path) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "read.db"))
    shared = service.remember(MemoryWriteInput(text="shared note", reason="test"))
    hidden = service.remember(
        MemoryWriteInput(namespace="private", text="private note", reason="test")
    )
    policy = default_memory_access_policy_factory(MemoryConfig())
    get = MemoryGetTool(service=service, access_policy_factory=policy)
    listing = MemoryListTool(service=service, access_policy_factory=policy)

    found = await get.arun(MemoryGetToolInput(item_id=shared.id), _context(tmp_path, "another"))
    missing = await get.arun(MemoryGetToolInput(item_id=hidden.id), _context(tmp_path))
    listed = await listing.arun(MemoryListToolInput(), _context(tmp_path))

    assert json.loads(found.content[0].text)["item"]["id"] == shared.id
    assert json.loads(missing.content[0].text) == {"found": False}
    assert [item["id"] for item in json.loads(listed.content[0].text)["items"]] == [shared.id]


@pytest.mark.asyncio
async def test_empty_read_range_returns_no_memory(tmp_path: Path) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "empty.db"))
    item = service.remember(MemoryWriteInput(text="remembered", reason="test"))

    def policy(_: ToolExecutionContext) -> MemoryAccessPolicy:
        return MemoryAccessPolicy(read_namespaces=())

    context = _context(tmp_path)

    search = await MemorySearchTool(service=service, access_policy_factory=policy).arun(
        MemorySearchToolInput(query="remembered"), context
    )
    listed = await MemoryListTool(service=service, access_policy_factory=policy).arun(
        MemoryListToolInput(), context
    )
    found = await MemoryGetTool(service=service, access_policy_factory=policy).arun(
        MemoryGetToolInput(item_id=item.id), context
    )

    assert json.loads(search.content[0].text) == {"results": []}
    assert json.loads(listed.content[0].text) == {"items": []}
    assert json.loads(found.content[0].text) == {"found": False}


def test_memory_registry_defaults_to_reads_and_can_select_write_tools(tmp_path: Path) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "registry.db"))
    policy = default_memory_access_policy_factory(MemoryConfig())
    registry = register_memory_tools(service=service, access_policy_factory=policy)
    assert {tool.definition.name for tool in registry.view().active_tools} == {
        "memory_search",
        "memory_list",
        "memory_get",
    }
    writes = register_memory_tools(
        service=service,
        access_policy_factory=policy,
        tool_names=("memory.remember", "memory.update", "memory.forget"),
    )
    for tool in writes.view().active_tools:
        definition = tool.definition
        assert definition.capabilities == {ToolCapability.WRITE}
        assert (
            not {"namespace", "actor", "permission_mode"}
            & definition.input_schema["properties"].keys()
        )


def _executor(service: MemoryService, *, write_namespace: str = "project") -> ToolExecutor:
    registry = register_memory_tools(
        service=service,
        access_policy_factory=lambda _: MemoryAccessPolicy(
            read_namespaces=("project", "private"), write_namespace=write_namespace
        ),
        tool_names=tuple(MEMORY_TOOL_CLASSES),
    )
    return ToolExecutor(registry, permission_policy=DefaultPermissionPolicy(write_mode="allow"))


def test_update_schema_distinguishes_optional_fields_from_nullable_scores(tmp_path: Path) -> None:
    """提供给模型的 schema 允许省略更新字段，不把非 nullable 字段声明为可传 null。"""
    service = MemoryService(SQLiteMemoryStore(tmp_path / "patch-schema.db"))
    schema = _executor(service).registry.get("memory_update").definition.input_schema
    patch = schema["$defs"]["MemoryItemPatch"]
    properties = patch["properties"]
    for field in ("text", "category", "kind", "status", "artifacts", "metadata"):
        assert field not in patch.get("required", [])
        assert {"type": "null"} not in properties[field].get("anyOf", [])
        assert "default" not in properties[field]
    for field in ("confidence", "importance"):
        assert {"type": "null"} in properties[field]["anyOf"]


@pytest.mark.asyncio
async def test_write_tools_crud_uses_same_service_and_reports_actual_delete(tmp_path: Path) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "crud.db"))
    executor = _executor(service, write_namespace="private")
    context = _context(tmp_path)
    remembered = await executor.execute_one(
        ToolUseBlock(
            id="remember",
            name="memory_remember",
            input={
                "text": "prefers bananas",
                "reason": "user asked to remember",
                "kind": "preference",
            },
        ),
        context,
    )
    assert not remembered.is_error
    item = json.loads(remembered.content[0].text)["item"]
    assert item["namespace"] == "private"
    stored = service.get_item(item["id"], ["private"])
    assert stored is not None
    assert stored.source_type is MemorySourceType.TOOL_EVENT
    assert stored.source_id == "remember"
    assert service.list_events("private")[0].actor is MemoryActor.AGENT
    found = await executor.execute_one(
        ToolUseBlock(id="get", name="memory_get", input={"item_id": item["id"]}),
        context,
    )
    assert json.loads(found.content[0].text)["item"]["text"] == "prefers bananas"
    updated = await executor.execute_one(
        ToolUseBlock(
            id="update",
            name="memory_update",
            input={
                "item_id": item["id"],
                "patch": {"text": "prefers oranges"},
                "reason": "user correction",
            },
        ),
        context,
    )
    assert not updated.is_error
    assert json.loads(updated.content[0].text)["item"]["id"] == item["id"]
    search = await executor.execute_one(
        ToolUseBlock(id="search", name="memory_search", input={"query": "oranges"}),
        context,
    )
    assert [entry["id"] for entry in json.loads(search.content[0].text)["results"]] == [item["id"]]
    for call_id, expected in [("forget", True), ("forget-again", False)]:
        forgotten = await executor.execute_one(
            ToolUseBlock(
                id=call_id,
                name="memory_forget",
                input={"item_id": item["id"], "reason": "no longer needed"},
            ),
            context,
        )
        assert not forgotten.is_error
        assert json.loads(forgotten.content[0].text) == {"deleted": expected}
    assert service.get_item(item["id"], ["private"]) is None


@pytest.mark.asyncio
async def test_write_namespace_is_bound_even_when_another_namespace_is_readable(
    tmp_path: Path,
) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "bound.db"))
    other = service.remember(
        MemoryWriteInput(namespace="private", text="other fact", reason="seed")
    )
    executor = _executor(service)
    context = _context(tmp_path)
    updated = await executor.execute_one(
        ToolUseBlock(
            id="update",
            name="memory_update",
            input={"item_id": other.id, "patch": {"text": "changed"}, "reason": "attempt"},
        ),
        context,
    )
    assert updated.is_error
    forgotten = await executor.execute_one(
        ToolUseBlock(
            id="forget", name="memory_forget", input={"item_id": other.id, "reason": "attempt"}
        ),
        context,
    )
    assert json.loads(forgotten.content[0].text) == {"deleted": False}
    injected = await executor.execute_one(
        ToolUseBlock(
            id="remember",
            name="memory_remember",
            input={"text": "new fact", "reason": "attempt", "namespace": "private"},
        ),
        context,
    )
    assert injected.is_error
    assert service.get_item(other.id, ["private"]) == other
    assert service.list_items(["project"]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["store", "mirror"])
async def test_write_tool_distinguishes_store_failure_from_mirror_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "fail.db"), mirror=mirror)

    def fail(*args: object, **kwargs: object) -> None:
        raise IrisMemoryError(f"{failure} unavailable")

    if failure == "store":
        monkeypatch.setattr(service.store, "add_item", fail)
    else:
        monkeypatch.setattr(mirror, "project_batch", fail)
    result = await _executor(service).execute_one(
        ToolUseBlock(id="write", name="memory_remember", input={"text": "fact", "reason": "seed"}),
        _context(tmp_path),
    )
    assert result.is_error is (failure == "store")
    assert len(service.list_items(["project"])) == (0 if failure == "store" else 1)


@pytest.mark.asyncio
async def test_update_rejects_null_metadata_without_corrupting_saved_item(tmp_path: Path) -> None:
    """正常模型调用传入 metadata:null 时应在预检失败，条目保持可读取。"""
    service = MemoryService(SQLiteMemoryStore(tmp_path / "null-metadata.db"))
    item = service.remember(
        MemoryWriteInput(text="unchanged fact", metadata={"origin": "user"}, reason="seed")
    )
    result = await _executor(service).execute_one(
        ToolUseBlock(
            id="update-null",
            name="memory_update",
            input={"item_id": item.id, "patch": {"metadata": None}, "reason": "clear metadata"},
        ),
        _context(tmp_path),
    )

    assert service.get_item(item.id, ["project"]) == item
    assert result.is_error
    assert result.error is not None and result.error.code == "VALIDATION_ERROR"
    assert len(service.list_events("project", item_id=item.id)) == 1


@pytest.mark.asyncio
async def test_update_can_clear_nullable_scores_without_changing_omitted_fields(
    tmp_path: Path,
) -> None:
    """允许明确清空 nullable 评分，省略的正文与元数据不随之改变。"""
    service = MemoryService(SQLiteMemoryStore(tmp_path / "clear-scores.db"))
    item = service.remember(
        MemoryWriteInput(
            text="keep fact",
            metadata={"origin": "user"},
            confidence=0.8,
            importance=0.7,
            reason="seed",
        )
    )
    result = await _executor(service).execute_one(
        ToolUseBlock(
            id="clear-scores",
            name="memory_update",
            input={
                "item_id": item.id,
                "patch": {"confidence": None, "importance": None},
                "reason": "clear scores",
            },
        ),
        _context(tmp_path),
    )

    assert not result.is_error
    updated = service.get_item(item.id, ["project"])
    assert updated is not None
    assert updated.confidence is None and updated.importance is None
    assert updated.text == item.text and updated.metadata == item.metadata
