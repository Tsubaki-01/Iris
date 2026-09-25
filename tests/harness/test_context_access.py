"""当前 session 原文与产物的有限回读。"""

from pathlib import Path

import pytest

from iris.exceptions import IrisToolExecutionError
from iris.harness._context_access import ContextAccess
from iris.lifecycle import SessionSnapshot
from iris.message import Msg, TextBlock, ToolResultBlock, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import (
    ToolArtifactStore,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)
from iris.tools.context_access import ContextReadInput, ContextReadTool, ContextSearchInput


def _access(messages: list[Msg]) -> ContextAccess:
    """准备专属于会话 one 的已提交原文 fixture。"""
    store = InMemoryLifecycleStore()
    store._sessions["one"] = SessionSnapshot(session_id="one", messages=messages)
    return ContextAccess(store)


def test_read_text_and_raw_pages_keep_original_result(tmp_path: Path) -> None:
    """多块消息的 result ref 取回完整最终文本，并可另读 MCP raw。"""
    artifacts = ToolArtifactStore(tmp_path, preview_chars=5)
    raw = artifacts.persist_json("call", {"raw": "source"}, preview="source")
    text = "新文本\n" * 300
    result = artifacts.persist_if_large(
        ToolResult(
            tool_use_id="call", tool_name="mcp", artifact=raw, content=[TextBlock(text=text)]
        ),
        max_chars=500,
    )
    block = result.to_msg().tool_results[0]
    access = _access([Msg.user("原始问题"), Msg.user([TextBlock(text="note"), block])])
    content = ""
    offset = 0
    while True:
        page = access.read(
            "one", ContextReadInput(ref="result:1:1", offset=offset, limit=73), tmp_path
        )
        content += page.content
        if not page.has_more:
            break
        offset = page.next_offset
    assert content == text
    raw_page = access.read(
        "one", ContextReadInput(ref="result:1:1", representation="raw"), tmp_path
    )
    assert raw_page.content == '{"raw": "source"}'
    assert access.read("one", ContextReadInput(ref="message:0"), tmp_path).content.endswith(
        "原始问题"
    )


def test_search_continues_after_empty_scan_and_returns_original_ref(tmp_path: Path) -> None:
    """一页无命中不代表完整会话无匹配，且匹配使用 Unicode casefold。"""
    messages = [Msg.user("nothing") for _ in range(200)]
    messages.append(Msg.user([ToolResultBlock(tool_use_id="call", name="read", content="Straße")]))
    access = _access(messages)
    first = access.search("one", ContextSearchInput(query="STRASSE"))
    assert first.matches == () and first.has_more and first.next_after == 200
    second = access.search("one", ContextSearchInput(query="STRASSE", after=first.next_after))
    assert second.matches[0].ref == "result:200:0"
    assert not second.has_more


def test_inline_result_raw_unavailable_and_deleted_artifact_fails(tmp_path: Path) -> None:
    """小结果只提供 text；产物失效时明确失败而不返回预览冒充正文。"""
    access = _access([Msg.tool_result(tool_use_id="small", content="inline")])
    assert access.read("one", ContextReadInput(ref="result:0:0"), tmp_path).content == "inline"
    with pytest.raises(IrisToolExecutionError) as unavailable:
        access.read("one", ContextReadInput(ref="result:0:0", representation="raw"), tmp_path)
    assert unavailable.value.context["code"] == "CONTEXT_REPRESENTATION_UNAVAILABLE"
    missing = _access(
        [
            Msg.tool_result(
                tool_use_id="old",
                content="preview",
                metadata={
                    "artifact": {
                        "path": str(tmp_path / "gone.txt"),
                        "text_path": str(tmp_path / "gone.txt"),
                    }
                },
            )
        ]
    )
    with pytest.raises(IrisToolExecutionError) as deleted:
        missing.read("one", ContextReadInput(ref="result:0:0"), tmp_path)
    assert deleted.value.context["code"] == "CONTEXT_SOURCE_UNAVAILABLE"


def test_search_does_not_open_artifact_body(tmp_path: Path) -> None:
    """搜索只覆盖原文消息中的预览，正文通过命中 ref 单独读取。"""
    path = tmp_path / "body.txt"
    path.write_text("hidden needle", encoding="utf-8")
    access = _access(
        [
            Msg.tool_result(
                tool_use_id="large",
                content="preview",
                metadata={"artifact": {"path": str(path), "text_path": str(path)}},
            )
        ]
    )
    assert access.search("one", ContextSearchInput(query="needle")).matches == ()
    assert access.search("one", ContextSearchInput(query="preview")).matches[0].ref == "result:0:0"


@pytest.mark.asyncio
async def test_read_failure_is_tool_result_without_cross_session_lookup(tmp_path: Path) -> None:
    """不能把另一 session 的同位置消息当作本 session 原文。"""
    registry = ToolRegistry()
    registry.register(ContextReadTool(_access([Msg.user("private-to-one")])))
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="read", name="context_read", input={"ref": "message:0"}),
        ToolExecutionContext(workspace_root=tmp_path, session_id="two"),
    )
    assert result.is_error and result.error.code == "CONTEXT_SOURCE_UNAVAILABLE"
