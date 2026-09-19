"""记忆读取工具的项目共享、联合排序和显式范围。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris.memory import (
    MemoryAccessPolicy,
    MemoryConfig,
    MemoryGetTool,
    MemoryGetToolInput,
    MemoryListTool,
    MemoryListToolInput,
    MemorySearchTool,
    MemorySearchToolInput,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
    default_memory_access_policy_factory,
)
from iris.tools import ToolExecutionContext


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
