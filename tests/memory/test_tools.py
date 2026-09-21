"""记忆工具的项目共享、联合读取、显式写入和真实执行结果。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from iris.exceptions import IrisMemoryError
from iris.memory import (
    MEMORY_TOOL_CLASSES,
    FileMemoryMirror,
    MemoryAccessPolicy,
    MemoryActor,
    MemoryArtifactRef,
    MemoryConfig,
    MemoryFetchTool,
    MemoryFetchToolInput,
    MemoryItemPatch,
    MemorySearchQuery,
    MemorySearchTool,
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


@pytest.mark.parametrize(
    "data", [{}, {"item_id": " \n"}, {"item_id": "id", "namespace": "private"}]
)
def test_fetch_input_requires_only_a_nonblank_item_id(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MemoryFetchToolInput.model_validate(data)
    assert MemoryFetchToolInput(item_id="arbitrary-id").item_id == "arbitrary-id"


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

    result = await tool.arun(MemorySearchQuery(query="needle", limit=1), _context(tmp_path))

    payload = json.loads(result.content[0].text)
    assert [item["item_id"] for item in payload["items"]] == [relevant.id]
    assert payload["items"][0]["namespace"] == "project"
    assert payload["has_more"] is True
    assert "不要求继续查询" in payload["hint"]


@pytest.mark.asyncio
async def test_fetch_returns_the_current_full_item_and_refreshes_policy(tmp_path: Path) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "read.db"))
    item = service.remember(
        MemoryWriteInput(
            text="original needle",
            reason="seed",
            source_id="known-source",
            confidence=0,
            artifacts=[MemoryArtifactRef(path="absent-attachment.txt", metadata={"name": "资料"})],
            metadata={"nested": {"names": ["完整", "内容"]}},
        )
    )
    namespace = "project"
    calls = 0

    def policy(_: ToolExecutionContext) -> MemoryAccessPolicy:
        nonlocal calls
        calls += 1
        return MemoryAccessPolicy(read_namespaces=(namespace,))

    search = MemorySearchTool(service=service, access_policy_factory=policy)
    fetch = MemoryFetchTool(service=service, access_policy_factory=policy)
    searched = await search.arun(MemorySearchQuery(query="needle"), _context(tmp_path))
    assert json.loads(searched.content[0].text)["items"][0]["item_id"] == item.id
    updated = service.update(
        item.id, "project", MemoryItemPatch(text="updated body"), reason="edit"
    )
    for _ in range(2):
        found = await fetch.arun(MemoryFetchToolInput(item_id=item.id), _context(tmp_path))
        assert json.loads(found.content[0].text) == {"item": updated.model_dump(mode="json")}
    namespace = "private"
    assert json.loads(
        (await search.arun(MemorySearchQuery(query="updated"), _context(tmp_path))).content[0].text
    ) == {"items": [], "has_more": False}
    with pytest.raises(IrisMemoryError, match="允许读取范围内未找到有效记忆"):
        await fetch.arun(MemoryFetchToolInput(item_id=item.id), _context(tmp_path))
    assert calls == 5


@pytest.mark.asyncio
async def test_fetch_by_known_id_requires_no_previous_search(tmp_path: Path) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "known.db"))
    item = service.remember(MemoryWriteInput(text="direct body", reason="seed"))
    fetch = MemoryFetchTool(
        service=service, access_policy_factory=default_memory_access_policy_factory(MemoryConfig())
    )
    result = await fetch.arun(MemoryFetchToolInput(item_id=item.id), _context(tmp_path))
    assert json.loads(result.content[0].text) == {"item": item.model_dump(mode="json")}


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["missing", "deleted", "superseded", "outside"])
async def test_fetch_reports_unavailable_items_through_the_executor(
    tmp_path: Path, state: str
) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "unavailable.db"))
    item = service.remember(
        MemoryWriteInput(
            namespace="excluded" if state == "outside" else "project", text="body", reason="seed"
        )
    )
    if state in {"deleted", "superseded"}:
        service.update(item.id, "project", MemoryItemPatch(status=state), reason="inactive")
    result = await _executor(service).execute_one(
        ToolUseBlock(
            id="fetch",
            name="memory_fetch",
            input={"item_id": "missing" if state == "missing" else item.id},
        ),
        _context(tmp_path),
    )
    assert result.is_error
    assert result.error is not None
    assert "允许读取范围内未找到有效记忆" in result.error.message


@pytest.mark.asyncio
async def test_empty_read_range_returns_no_search_hits_and_fetch_error(tmp_path: Path) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "empty.db"))
    item = service.remember(MemoryWriteInput(text="remembered", reason="test"))

    def policy(_: ToolExecutionContext) -> MemoryAccessPolicy:
        return MemoryAccessPolicy(read_namespaces=())

    result = await MemorySearchTool(service=service, access_policy_factory=policy).arun(
        MemorySearchQuery(query="remembered"), _context(tmp_path)
    )
    assert json.loads(result.content[0].text) == {"items": [], "has_more": False}
    with pytest.raises(IrisMemoryError, match="允许读取范围内未找到有效记忆"):
        await MemoryFetchTool(service=service, access_policy_factory=policy).arun(
            MemoryFetchToolInput(item_id=item.id), _context(tmp_path)
        )


def test_memory_registry_defaults_empty_and_uses_shared_query_schema(tmp_path: Path) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "registry.db"))
    policy = default_memory_access_policy_factory(MemoryConfig())
    registry = register_memory_tools(service=service, access_policy_factory=policy)
    assert registry.view().active_tools == []
    reads = register_memory_tools(
        service=service, access_policy_factory=policy, tool_names=("memory.search", "memory.fetch")
    )
    assert {tool.definition.name for tool in reads.view().active_tools} == {
        "memory_search",
        "memory_fetch",
    }
    assert MemorySearchTool.input_type is MemorySearchQuery
    for tool in reads.view().active_tools:
        assert tool.definition.capabilities == {ToolCapability.READ}
        assert tool.definition.max_result_chars == 50000
        assert "namespaces" not in tool.definition.input_schema["properties"]
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


@pytest.mark.asyncio
async def test_fetch_uses_the_ordinary_result_cap_and_keeps_full_json_in_artifact(
    tmp_path: Path,
) -> None:
    service = MemoryService(SQLiteMemoryStore(tmp_path / "artifact.db"))
    item = service.remember(
        MemoryWriteInput(text="完整正文 " * 200, reason="seed", metadata={"tail": "保留"})
    )
    executor = ToolExecutor(
        register_memory_tools(
            service=service,
            access_policy_factory=default_memory_access_policy_factory(MemoryConfig()),
            tool_names=("memory.fetch",),
            max_result_chars=500,
        )
    )
    result = await executor.execute_one(
        ToolUseBlock(id="fetch", name="memory_fetch", input={"item_id": item.id}),
        _context(tmp_path),
    )
    assert not result.is_error
    assert result.artifact is not None
    assert len(result.content[0].text) <= 500
    payload = json.loads(result.artifact.path.read_text(encoding="utf-8"))
    assert payload == {"item": item.model_dump(mode="json")}


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
        ToolUseBlock(id="get", name="memory_fetch", input={"item_id": item["id"]}),
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
    assert [entry["item_id"] for entry in json.loads(search.content[0].text)["items"]] == [
        item["id"]
    ]
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
        monkeypatch.setattr(mirror, "rebuild_from_store", fail)
    result = await _executor(service).execute_one(
        ToolUseBlock(id="write", name="memory_remember", input={"text": "fact", "reason": "seed"}),
        _context(tmp_path),
    )
    assert result.is_error is (failure == "store")
    assert len(service.list_items(["project"])) == (0 if failure == "store" else 1)
    if failure == "mirror":
        assert "未同步" in json.loads(result.content[0].text)["warning"]


@pytest.mark.asyncio
async def test_all_write_tools_report_committed_projection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """每个写工具都保留成功结果，并告知正文投影尚未同步。"""
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "write-warning.db"), mirror=mirror)

    def fail(*args: object, **kwargs: object) -> None:
        raise IrisMemoryError("projection unavailable")

    monkeypatch.setattr(mirror, "rebuild_from_store", fail)
    executor = _executor(service)
    context = _context(tmp_path)
    remembered = await executor.execute_one(
        ToolUseBlock(
            id="remember", name="memory_remember", input={"text": "fact", "reason": "seed"}
        ),
        context,
    )
    item_id = json.loads(remembered.content[0].text)["item"]["id"]
    updated = await executor.execute_one(
        ToolUseBlock(
            id="update",
            name="memory_update",
            input={"item_id": item_id, "patch": {"text": "current fact"}, "reason": "edit"},
        ),
        context,
    )
    forgotten = await executor.execute_one(
        ToolUseBlock(
            id="forget", name="memory_forget", input={"item_id": item_id, "reason": "done"}
        ),
        context,
    )
    for result in (remembered, updated, forgotten):
        assert not result.is_error
        assert "未同步" in json.loads(result.content[0].text)["warning"]
    assert json.loads(forgotten.content[0].text)["deleted"] is True
    assert service.get_item(item_id, ["project"]) is None


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
