"""当前 session 原文与产物的有限回读。"""

from pathlib import Path

import pytest

from iris.exceptions import IrisToolExecutionError
from iris.harness._context_access import ContextAccess
from iris.lifecycle import SessionReadState, SessionSnapshot
from iris.message import (
    ImageBlock,
    ImageFileRef,
    Msg,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    image_reference_text,
)
from iris.store import InMemoryLifecycleStore
from iris.store._session_projection import advance_session_read_state
from iris.store.in_memory import _MemorySession
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
    snapshot = SessionSnapshot(session_id="one", messages=messages)
    store._sessions["one"] = _MemorySession(
        snapshot, advance_session_read_state(SessionReadState(), 0, messages)
    )
    return ContextAccess(store)


def _image(root: Path, key: str, name: str) -> ImageBlock:
    """引用无需存在的图片，回读和搜索只消费持久化文件信息。"""
    return ImageBlock(
        original=ImageFileRef(
            path=root / f"original-{key}.png", mime_type="image/png", width=3000, height=2000
        ),
        model=ImageFileRef(
            path=root / f"model-{key}.webp", mime_type="image/webp", width=1500, height=1000
        ),
        name=name,
    )


def _read_all(access: ContextAccess, ref: str, workspace: Path, *, limit: int) -> str:
    """根据真实 next_offset 拼接 Unicode 字符页。"""
    content = ""
    offset = 0
    while True:
        page = access.read("one", ContextReadInput(ref=ref, offset=offset, limit=limit), workspace)
        assert page.offset == offset
        assert page.next_offset == offset + len(page.content)
        content += page.content
        if not page.has_more:
            return content
        offset = page.next_offset


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
    messages.append(
        Msg.user(
            [ToolResultBlock(tool_use_id="call", name="read", content=[TextBlock(text="Straße")])]
        )
    )
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


def test_multiple_text_blocks_are_read_and_searched_as_text(tmp_path: Path) -> None:
    """分页与检索消费有序文字投影，保持跨文本块的字符偏移。"""
    access = _access(
        [
            Msg.tool_result(
                tool_use_id="call", content=[TextBlock(text="first"), TextBlock(text="last")]
            )
        ]
    )
    page = access.read("one", ContextReadInput(ref="result:0:0", offset=4, limit=4), tmp_path)
    assert page.content == "t\nla"
    assert page.next_offset == 8 and page.has_more
    message = access.read("one", ContextReadInput(ref="message:0"), tmp_path)
    assert message.content.endswith("first\nlast")
    hit = access.search("one", ContextSearchInput(query="last")).matches[0]
    assert hit.ref == "result:0:0" and hit.snippet == "first\nlast"


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


@pytest.mark.parametrize("image_only", [False, True])
def test_message_and_inline_result_render_images_without_opening_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image_only: bool
) -> None:
    top = _image(tmp_path, "top", "用户截图 🌌")
    nested = _image(tmp_path, "tool", "工具图 中文")
    parts = [nested] if image_only else [TextBlock(text="前文 🖼️"), nested, TextBlock(text="后文")]
    result = ToolResultBlock(tool_use_id="call", name="camera", content=parts)
    message = Msg.user(
        [TextBlock(text="说明"), top, ToolUseBlock(id="call", name="camera", input={}), result]
    )
    access = _access([*[Msg.user("earlier") for _ in range(8)], message])

    def no_file_read(*args: object, **kwargs: object) -> None:
        pytest.fail("内联图片回读不应打开图片文件")

    monkeypatch.setattr(Path, "open", no_file_read)
    text = _read_all(access, "message:8", tmp_path, limit=19)
    assert image_reference_text(top) in text and image_reference_text(nested) in text
    assert "camera call_id=call" in text and "[block 3 tool_result]" in text
    expected = image_reference_text(nested)
    if not image_only:
        expected = "前文 🖼️\n" + expected + "\n后文"
    assert _read_all(access, "result:8:3", tmp_path, limit=17) == expected
    assert result.content == parts


def test_search_matches_image_names_and_paths_with_absolute_refs_without_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    top = _image(tmp_path, "top", "Straße 星空")
    nested = _image(tmp_path, "nested", "图表 needle")
    access = _access(
        [
            *[Msg.user("earlier") for _ in range(5)],
            Msg.user([top]),
            Msg.user(
                [
                    TextBlock(text="note"),
                    ToolResultBlock(
                        tool_use_id="call",
                        name="camera",
                        content=[nested],
                        metadata={
                            "artifact": {
                                "path": str(tmp_path / "raw.json"),
                                "text_path": str(tmp_path / "full.txt"),
                            }
                        },
                    ),
                ]
            ),
        ]
    )

    def no_file_read(*args: object, **kwargs: object) -> None:
        pytest.fail("检索不能打开图片、raw artifact 或外置正文")

    monkeypatch.setattr(Path, "open", no_file_read)
    top_hit = access.search("one", ContextSearchInput(query="STRASSE")).matches[0]
    assert top_hit.ref == "message:5" and "Straße 星空" in top_hit.snippet
    nested_hit = access.search("one", ContextSearchInput(query="needle")).matches[0]
    assert nested_hit.ref == "result:6:1" and nested_hit.tool_name == "camera"
    path_hit = access.search("one", ContextSearchInput(query="model-nested.webp")).matches[0]
    assert path_hit.ref == "result:6:1"


def test_image_text_path_pages_use_saved_references_once_and_preserve_raw(tmp_path: Path) -> None:
    first = _image(tmp_path, "first", "首图")
    second = _image(tmp_path, "second", "尾图")
    artifacts = ToolArtifactStore(tmp_path / "artifacts", preview_chars=8)
    raw = artifacts.persist_json("call", {"raw": "完整 MCP JSON"}, preview="raw")
    body = "很长的中文 🖼️\n" * 300
    result = artifacts.persist_if_large(
        ToolResult(
            tool_use_id="call",
            tool_name="camera",
            artifact=raw,
            content=[first, TextBlock(text=body), second],
        ),
        max_chars=1000,
    )
    access = _access([result.to_msg()])
    expected = f"{image_reference_text(first)}\n{image_reference_text(second)}\n\n{body}"
    assert result.artifact.text_path.read_text(encoding="utf-8") == expected
    text = _read_all(access, "result:0:0", tmp_path, limit=31)
    assert text == expected
    assert text.count(image_reference_text(first)) == text.count(image_reference_text(second)) == 1
    raw_page = access.read(
        "one", ContextReadInput(ref="result:0:0", representation="raw"), tmp_path
    )
    assert raw_page.content == '{"raw": "完整 MCP JSON"}'


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
