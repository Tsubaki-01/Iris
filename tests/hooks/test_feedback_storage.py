"""真实工具 Hook 反馈的提交、恢复、压缩与原文回读。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from iris.agents import CompactionConfig
from iris.harness import AgentRunner
from iris.harness._context_access import ContextAccess
from iris.hooks import HookEvent, HookHandler, HookRegistration, ToolAfterEvent, ToolAfterResult
from iris.hooks.dispatcher import HookDispatcher
from iris.lifecycle import AgentRunRequest, LifecycleStore, RunStopReason
from iris.message import (
    ImageBlock,
    ImageFileRef,
    LLMRequest,
    LLMResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    image_reference_text,
)
from iris.runtime import AgentRuntime
from iris.store import SQLiteStore
from iris.tools import ToolErrorInfo, ToolRegistry, ToolResult
from iris.tools.context_access import ContextReadInput

from ..harness.fakes import StaticProvider, build_runtime, text_response, tool_response
from ..harness.test_context_compaction import CompactionProvider
from ..store.test_lifecycle_store_contract import lifecycle_store as lifecycle_store

_FEEDBACK = ("检查反馈：保留原始输出。", "后续建议：核对第二项。")
_SESSION = "feedback-session"
_CALL = "feedback-call"


@pytest.fixture
def tool_image(tmp_path: Path) -> ImageBlock:
    """提供路径和尺寸不同的真实原图与模型副本。"""
    original_path = tmp_path / "original.png"
    model_path = tmp_path / "model.png"
    Image.new("RGB", (32, 24), "red").save(original_path)
    Image.new("RGB", (16, 12), "red").save(model_path)
    return ImageBlock(
        original=ImageFileRef(path=original_path, mime_type="image/png", width=32, height=24),
        model=ImageFileRef(path=model_path, mime_type="image/png", width=16, height=12),
        name="feedback-image",
    )


def _with_feedback(
    runtime: AgentRuntime, feedback: tuple[str, ...], effects: list[str], image: ImageBlock
) -> AgentRuntime:
    """通过环境构造期真实接线，将两段反馈各自交给独立 after 处理器。"""

    async def before(event: HookEvent) -> None:
        effects.append("before")

    def make_after(index: int, text: str) -> HookHandler:
        async def after(event: HookEvent) -> ToolAfterResult:
            assert isinstance(event, ToolAfterEvent)
            assert event.call_id == _CALL
            assert event.result.hook_feedback == ()
            assert [part for part in event.result.content if isinstance(part, ImageBlock)] == [
                image
            ]
            effects.append(f"after:{index}")
            return ToolAfterResult(feedback=text)

        return after

    dispatcher = HookDispatcher(
        [
            HookRegistration(event="tool.before", name="before", handler=before),
            *(
                HookRegistration(
                    event="tool.after", name=f"after:{index}", handler=make_after(index, text)
                )
                for index, text in enumerate(feedback)
            ),
        ]
    )
    return AgentRuntime(replace(runtime.environment, hook_dispatcher=dispatcher))


def _reopen(store: LifecycleStore) -> LifecycleStore:
    """SQLite 从新实例重读，进程内 store 使用其真实读取投影。"""
    return SQLiteStore(store.path) if isinstance(store, SQLiteStore) else store


def _read_full(access: ContextAccess, ref: str, workspace: Path) -> str:
    """跨多个真实回读页重建最终正文，覆盖 Unicode 分页。"""
    pieces: list[str] = []
    offset = 0
    while True:
        page = access.read(_SESSION, ContextReadInput(ref=ref, offset=offset, limit=113), workspace)
        pieces.append(page.content)
        if not page.has_more:
            return "".join(pieces)
        offset = page.next_offset


def _result_block(store: LifecycleStore) -> tuple[int, int, ToolResultBlock]:
    """从已提交会话定位唯一工具结果，不手造历史消息。"""
    session = store.load_session(_SESSION)
    assert session is not None
    found = [
        (message_index, block_index, block)
        for message_index, message in enumerate(session.messages)
        for block_index, block in enumerate(message.blocks)
        if isinstance(block, ToolResultBlock) and block.tool_use_id == _CALL
    ]
    assert len(found) == 1
    return found[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("large", [False, True])
async def test_feedback_survives_runner_commit_and_context_reads(
    tmp_path: Path,
    lifecycle_store: LifecycleStore,
    failed: bool,
    large: bool,
    tool_image: ImageBlock,
) -> None:
    """图文成功/错误及超额正文都经过真实提交，图片与反馈持久化后仍可回读。"""
    feedback = (_FEEDBACK[0] + ("资料" * 700 if large else ""), _FEEDBACK[1])
    raw = ToolResult(
        tool_use_id="",
        tool_name="feedback_source",
        content=[TextBlock(text="原始工具正文"), tool_image],
        is_error=failed,
        error=ToolErrorInfo(code="SOURCE_FAILED", message="原始工具错误") if failed else None,
        hook_feedback=feedback,
    )
    effects: list[str] = []

    def feedback_source() -> ToolResult:
        """body 只返回业务结果；反馈由真实 after 处理器追加。"""
        effects.append("executed")
        return raw.model_copy(update={"hook_feedback": ()})

    registry = ToolRegistry()
    tool = registry.register_function(feedback_source, description="提供带反馈的结果")
    tool.definition.max_result_chars = 1000
    tool.definition.preview_chars = 64
    provider = StaticProvider(
        tool_response(ToolUseBlock(id=_CALL, name="feedback_source", input={})),
        text_response("完成"),
    )
    runner = AgentRunner(
        runtime=_with_feedback(
            build_runtime(tmp_path, registry=registry, provider=provider),
            feedback,
            effects,
            tool_image,
        ),
        store=lifecycle_store,
    )
    try:
        result = await runner.start(
            AgentRunRequest(input="检查反馈", run_id="feedback-run", session_id=_SESSION)
        )
    finally:
        await runner.aclose()
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert effects == ["before", "executed", "after:0", "after:1"]

    store = _reopen(lifecycle_store)
    record = store.load_tool_call("feedback-run", _CALL)
    assert record is not None and record.result is not None
    saved = record.result
    assert [part for part in saved.content if isinstance(part, ImageBlock)] == [tool_image]
    assert saved.is_error is failed
    assert saved.hook_feedback == (() if large else feedback)
    assert saved.model_content.count("Error[SOURCE_FAILED]") == int(failed)
    assert saved.model_content.count("[Hook feedback]") <= 2
    message_index, block_index, block = _result_block(store)
    # SQLite 的 metadata 路径经过 JSON 编码；比较同一个序列化消息契约。
    assert block.model_dump(mode="json") == saved.to_msg().tool_results[0].model_dump(mode="json")
    delivered = [
        item
        for message in provider.requests[1].messages
        for item in message.tool_results
        if item.tool_use_id == _CALL
    ]
    assert len(delivered) == 1
    assert [part for part in delivered[0].content if isinstance(part, ImageBlock)] == [tool_image]
    assert delivered[0].is_error is failed
    assert delivered[0].name == block.name
    if large:
        # 模型历史投影为 artifact 结果追加回读 ref，已存消息仍保持原预览。
        assert delivered[0].text.startswith(block.text)
        assert f"result:{message_index}:{block_index}" in delivered[0].text
    else:
        assert delivered[0].model_dump(mode="json") == block.model_dump(mode="json")

    if large:
        assert saved.artifact is not None and saved.artifact.text_path is not None
        expected_full = f"{image_reference_text(tool_image)}\n\n{raw.model_content}"
        assert saved.artifact.text_path.read_text(encoding="utf-8") == expected_full
        assert len(saved.model_content) <= 1000
    else:
        assert saved.artifact is None
        assert saved.model_content == raw.model_content
        expected_full = "\n".join(
            [
                "Error[SOURCE_FAILED]: 原始工具错误" if failed else "原始工具正文",
                image_reference_text(tool_image),
                *(f"[Hook feedback]\n{text}" for text in feedback),
            ]
        )
    access = ContextAccess(store)
    full_text = _read_full(access, f"result:{message_index}:{block_index}", tmp_path)
    assert full_text == expected_full
    assert full_text.count("[Hook feedback]") == 2
    assert all(full_text.count(item) == 1 for item in feedback)
    message_text = _read_full(access, f"message:{message_index}", tmp_path)
    assert image_reference_text(tool_image) in message_text
    if large:
        assert saved.model_content in message_text
    else:
        assert message_text.endswith(expected_full)
    assert effects == ["before", "executed", "after:0", "after:1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_compaction_receives_feedback_and_preserves_original_reads(
    tmp_path: Path, lifecycle_store: LifecycleStore, failed: bool, tool_image: ImageBlock
) -> None:
    """摘要接收图片引用及反馈，原始图片块和正文在压缩后仍可回读。"""
    body = "原始资料" * 230
    raw = ToolResult(
        tool_use_id="",
        tool_name="feedback_source",
        content=[TextBlock(text=body), tool_image],
        is_error=failed,
        error=ToolErrorInfo(code="SOURCE_FAILED", message=body) if failed else None,
        hook_feedback=_FEEDBACK,
    )
    effects: list[str] = []

    def feedback_source() -> ToolResult:
        """用超过上下文预算、未超过 artifact 阈值的结果触发真实压缩。"""
        effects.append("executed")
        return raw.model_copy(update={"hook_feedback": ()})

    registry = ToolRegistry()
    registry.register_function(feedback_source, description="提供需要压缩的反馈结果")
    provider = CompactionProvider(
        tool_response(ToolUseBlock(id=_CALL, name="feedback_source", input={})),
        text_response("压缩后完成"),
    )
    runtime = _with_feedback(
        build_runtime(tmp_path, registry=registry, provider=provider),
        _FEEDBACK,
        effects,
        tool_image,
    )
    runtime.environment.agent_config = runtime.environment.agent_config.model_copy(
        update={"compaction": CompactionConfig(input_budget_tokens=1000)}
    )
    runner = AgentRunner(runtime=runtime, store=lifecycle_store)
    try:
        result = await runner.start(
            AgentRunRequest(input="读取并总结", run_id="compacted-feedback", session_id=_SESSION)
        )
    finally:
        await runner.aclose()
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert effects == ["before", "executed", "after:0", "after:1"]
    assert len(provider.summary_requests) == 1

    store = _reopen(lifecycle_store)
    message_index, block_index, block = _result_block(store)
    session = store.load_session(_SESSION)
    assert session is not None and session.compaction is not None
    assert session.compaction.covered_message_count > message_index
    summary_input = "\n".join(
        message.text for request in provider.summary_requests for message in request.messages
    )
    assert body in summary_input
    assert image_reference_text(tool_image) in summary_input
    assert f"ref=result:{message_index}:{block_index}" in summary_input
    assert all(summary_input.count(item) == 1 for item in _FEEDBACK)
    assert any(message.text.startswith("<summary>") for message in provider.requests[-1].messages)
    assert not any(
        item.tool_use_id == _CALL
        for message in provider.requests[-1].messages
        for item in message.tool_results
    )

    record = store.load_tool_call("compacted-feedback", _CALL)
    assert record is not None and record.result is not None
    assert record.result.hook_feedback == _FEEDBACK
    assert [part for part in record.result.content if isinstance(part, ImageBlock)] == [tool_image]
    assert block.content == record.result.model_blocks
    assert block.text == record.result.model_content == raw.model_content
    expected_full = "\n".join(
        [
            f"Error[SOURCE_FAILED]: {body}" if failed else body,
            image_reference_text(tool_image),
            *(f"[Hook feedback]\n{text}" for text in _FEEDBACK),
        ]
    )
    access = ContextAccess(store)
    assert _read_full(access, f"result:{message_index}:{block_index}", tmp_path) == expected_full
    assert _read_full(access, f"message:{message_index}", tmp_path).endswith(expected_full)
    assert effects == ["before", "executed", "after:0", "after:1"]


@pytest.mark.asyncio
async def test_recovery_reuses_committed_feedback_without_replaying_tool_or_hooks(
    tmp_path: Path, lifecycle_store: LifecycleStore, tool_image: ImageBlock
) -> None:
    """提交工具结果后的 SDK 中断以原 activation 恢复，不重放已完成 before/body/after。"""
    effects: list[str] = []
    entered = asyncio.Event()

    def feedback_source() -> ToolResult:
        """提供可精确计数的真实业务副作用。"""
        effects.append("executed")
        return ToolResult(
            tool_use_id="",
            tool_name="feedback_source",
            content=[TextBlock(text="已提交的原始正文"), tool_image],
        )

    class PauseAfterCommit(StaticProvider):
        """第二步模型入口只在工具结果已提交后到达。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            if not self.requests:
                return await super().complete(request)
            self.requests.append(request)
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("阻塞模型不应自行返回")

    registry = ToolRegistry()
    registry.register_function(feedback_source, description="提供需要恢复的反馈结果")
    provider = PauseAfterCommit(
        tool_response(ToolUseBlock(id=_CALL, name="feedback_source", input={}))
    )
    runner = AgentRunner(
        runtime=_with_feedback(
            build_runtime(tmp_path, registry=registry, provider=provider),
            _FEEDBACK,
            effects,
            tool_image,
        ),
        store=lifecycle_store,
    )
    running = asyncio.create_task(
        runner.start(
            AgentRunRequest(input="恢复反馈", run_id="recover-feedback", session_id=_SESSION)
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
    finally:
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        await runner.aclose()
    expected_effects = ["before", "executed", "after:0", "after:1"]
    assert effects == expected_effects
    store = _reopen(lifecycle_store)
    interrupted = store.load_run("recover-feedback")
    assert interrupted is not None and interrupted.current_activation_id is not None
    before = store.load_tool_call("recover-feedback", _CALL)
    assert before is not None and before.result is not None
    assert before.result.hook_feedback == _FEEDBACK
    assert [part for part in before.result.content if isinstance(part, ImageBlock)] == [tool_image]
    message_index, block_index, block = _result_block(store)
    assert block.content == before.result.model_blocks
    assert block.text == before.result.model_content

    resumed_provider = StaticProvider(text_response("恢复后完成"))
    resumed = AgentRunner(
        runtime=_with_feedback(
            build_runtime(tmp_path, registry=registry, provider=resumed_provider),
            _FEEDBACK,
            effects,
            tool_image,
        ),
        store=store,
    )
    try:
        result = await resumed.recover(
            "recover-feedback", expected_activation_id=interrupted.current_activation_id
        )
    finally:
        await resumed.aclose()
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert effects == expected_effects
    assert store.load_tool_call("recover-feedback", _CALL).result == before.result
    assert len(resumed_provider.requests) == 1
    delivered = [
        item
        for message in resumed_provider.requests[0].messages
        for item in message.tool_results
        if item.tool_use_id == _CALL
    ]
    assert len(delivered) == 1 and delivered[0].text == before.result.model_content
    assert delivered[0].content == block.content
    assert delivered[0].text.count("[Hook feedback]") == 2
    expected_full = "\n".join(
        [
            "已提交的原始正文",
            image_reference_text(tool_image),
            *(f"[Hook feedback]\n{text}" for text in _FEEDBACK),
        ]
    )
    access = ContextAccess(_reopen(store))
    assert _read_full(access, f"result:{message_index}:{block_index}", tmp_path) == expected_full
    assert _read_full(access, f"message:{message_index}", tmp_path).endswith(expected_full)
    assert effects == expected_effects
