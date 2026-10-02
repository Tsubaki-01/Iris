"""压缩后的旧工具图片通过 context_read 和 read_file 再次进入活动请求。"""

from __future__ import annotations

import base64
import re
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from iris.agents import AgentConfig
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.message import (
    ImageBlock,
    LLMRequest,
    LLMResponse,
    Msg,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from iris.providers.chat_completions import ChatCompletionsMapper
from iris.providers.responses import ResponsesMapper
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolResult

from .fakes import StaticProvider, text_response, tool_response
from .test_context_compaction import CompactionProvider

_CONCLUSION = "此前已确认图片是蓝色色块。"


def _images(messages: list[Msg]) -> list[ImageBlock]:
    """只检查当前消息契约的顶层与工具正文两处图片。"""
    images = []
    for message in messages:
        for block in message.blocks:
            if isinstance(block, ImageBlock):
                images.append(block)
            elif isinstance(block, ToolResultBlock):
                images.extend(part for part in block.content if isinstance(part, ImageBlock))
    return images


def _config(workspace: Path) -> AgentConfig:
    """使用普通装配注册上下文工具和已有 file.read。"""
    return AgentConfig.model_validate(
        {
            "name": "image-context",
            "model": "openai/test",
            "system": "仅根据实际读取到的资料回答。",
            "permissions": {"workspace": str(workspace)},
            "tools": {"builtin": ["file.read"]},
            "compaction": {"input_budget_tokens": 3000},
        }
    )


class ImageRecallProvider(CompactionProvider):
    """沿已收到的摘要引用和工具结果决定下一次读取，不预先知道缓存路径。"""

    def __init__(self) -> None:
        super().__init__()
        self.recalled_path: Path | None = None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """摘要只消费文字；主请求依次定位原文、读取缓存图、完成回答。"""
        if request.provider_options.get("num_retries") == 0:
            self.summary_requests.append(request)
            material = "\n".join(message.text for message in request.messages)
            assert _CONCLUSION in material
            assert "[image:" in material and "model=" in material
            assert not _images(request.messages)
            assert "data:image" not in material and "base64" not in material
            ref = re.search(r"ref=(result:\d+:\d+)", material)
            assert ref is not None
            return text_response(f"{_CONCLUSION}图片原文引用：{ref.group(1)}。")

        if not self.requests:
            assert not _images(request.messages)
            summary = next(
                message.text for message in request.messages if "<summary>" in message.text
            )
            ref = re.search(r"result:\d+:\d+", summary)
            assert ref is not None
            response = tool_response(
                ToolUseBlock(
                    id="locate-image",
                    name="context_read",
                    input={"ref": ref.group(0), "limit": 8000},
                )
            )
        elif len(self.requests) == 1:
            result = next(
                block
                for message in request.messages
                for block in message.tool_results
                if block.tool_use_id == "locate-image"
            )
            match = re.search(r"model=(.+?) \(image/", result.text)
            assert match is not None
            self.recalled_path = Path(match.group(1))
            response = tool_response(
                ToolUseBlock(
                    id="reload-image",
                    name="read_file",
                    input={"file_path": str(self.recalled_path)},
                )
            )
        else:
            assert len(_images(request.messages)) == 1
            response = text_response("已经通过保存的模型版重新查看图片。")
        self.responses.append(response)
        return await super().complete(request)


@pytest.mark.asyncio
async def test_compacted_tool_image_is_recalled_from_saved_model_copy(tmp_path: Path) -> None:
    """真实工具/store/压缩/回读闭环保留原历史，活动历史只重新带回读到的模型版。"""
    output = BytesIO()
    with Image.new("RGB", (2400, 1200), "blue") as pixels:
        pixels.save(output, format="PNG")
    source = tmp_path / "source.png"
    source.write_bytes(output.getvalue())
    store = InMemoryLifecycleStore()
    seed = AgentRunner.from_config(
        _config(tmp_path),
        store=store,
        provider=StaticProvider(
            tool_response(ToolUseBlock(id="old-image", name="capture_image")),
            text_response(_CONCLUSION + "已有背景材料。" * 800),
        ),
    )
    image = await seed.import_image(source, session_id="main", name="blue-chart.png")

    def capture_image() -> ToolResult:
        """普通图片工具无需自己添加用于摘要或 context_read 的引用文字。"""
        return ToolResult(
            tool_use_id="", tool_name="capture_image", content=[TextBlock(text=_CONCLUSION), image]
        )

    seed.runtime.environment.tool_bridge.tool_view.registry.register_function(capture_image)
    try:
        seeded = await seed.start(
            AgentRunRequest(input="读取图片", session_id="main", run_id="seed")
        )
        assert seeded.run.stop_reason is RunStopReason.COMPLETED
    finally:
        await seed.aclose()
    before = store.load_session("main")
    assert before.messages[2].tool_results[0].tool_use_id == "old-image"
    assert image.model.path != image.original.path
    source.write_bytes(b"original external source is no longer an image")

    provider = ImageRecallProvider()
    runner = AgentRunner.from_config(_config(tmp_path), store=store, provider=provider)
    try:
        result = await runner.start(
            AgentRunRequest(input="回看刚才图片的细节", session_id="main", run_id="recall")
        )
        assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    finally:
        await runner.aclose()

    after = store.load_session("main")
    assert after.compaction is not None and after.compaction.covered_message_count >= 4
    assert after.messages[: len(before.messages)] == before.messages
    assert _images(before.messages) == [image]
    assert len(provider.summary_requests) == 1 and len(provider.requests) == 3
    assert _CONCLUSION in after.compaction.summary
    assert "result:2:0" in after.compaction.summary
    assert not _images(provider.requests[0].messages)
    assert all(
        block.tool_use_id != "old-image"
        for request in provider.requests
        for message in request.messages
        for block in message.tool_results
    )
    assert provider.recalled_path == image.model.path
    recalled = _images(provider.requests[-1].messages)[0]
    assert recalled.original.path == recalled.model.path == image.model.path
    assert (recalled.model.width, recalled.model.height) == (2000, 1000)
    assert image.original.path.read_bytes() == output.getvalue()
    assert [call.tool_name for call in store.list_tool_calls("recall")] == [
        "context_read",
        "read_file",
    ]

    effective = provider.requests[-1].messages
    saved_effective = [message.model_dump() for message in effective]
    responses_wire = ResponsesMapper().format_messages(effective)
    chat_wire = ChatCompletionsMapper().format_messages(effective)
    responses_output = next(
        item
        for item in responses_wire
        if item.get("call_id") == "reload-image" and item["type"] == "function_call_output"
    )
    responses_image = next(
        part for part in responses_output["output"] if part["type"] == "input_image"
    )
    chat_image = next(
        part
        for item in chat_wire
        if isinstance(item.get("content"), list)
        for part in item["content"]
        if part["type"] == "image_url"
    )
    expected_url = "data:image/png;base64," + base64.b64encode(
        image.model.path.read_bytes()
    ).decode("ascii")
    assert responses_image["image_url"] == chat_image["image_url"]["url"] == expected_url
    assert [message.model_dump() for message in effective] == saved_effective
