"""反馈结果的真实提交与原文回读；本阶段不验证尚未接入的 Hook 派发。"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris.agents import CompactionConfig
from iris.harness import AgentRunner
from iris.harness._context_access import ContextAccess
from iris.lifecycle import AgentRunRequest, LifecycleStore, RunStopReason
from iris.message import TextBlock, ToolResultBlock, ToolUseBlock
from iris.store import SQLiteStore
from iris.tools import ToolErrorInfo, ToolRegistry, ToolResult
from iris.tools.context_access import ContextReadInput

from ..harness.fakes import StaticProvider, build_runtime, text_response, tool_response
from ..harness.test_context_compaction import CompactionProvider
from ..store.test_lifecycle_store_contract import lifecycle_store as lifecycle_store

_FEEDBACK = ("检查反馈：保留原始输出。", "后续建议：核对第二项。")
_SESSION = "feedback-session"
_CALL = "feedback-call"


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
    tmp_path: Path, lifecycle_store: LifecycleStore, failed: bool, large: bool
) -> None:
    """成功/错误及超额正文都经过真实提交，结果、历史和全文回读保持一致。"""
    feedback = (_FEEDBACK[0] + ("资料" * 700 if large else ""), _FEEDBACK[1])
    raw = ToolResult(
        tool_use_id="",
        tool_name="feedback_source",
        content=[TextBlock(text="原始工具正文")],
        is_error=failed,
        error=ToolErrorInfo(code="SOURCE_FAILED", message="原始工具错误") if failed else None,
        hook_feedback=feedback,
    )
    effects: list[str] = []

    def feedback_source() -> ToolResult:
        """直接提供 P2 结果字段，不假装通过了 P4 tool.after。"""
        effects.append("executed")
        return raw

    registry = ToolRegistry()
    tool = registry.register_function(feedback_source, description="提供带反馈的结果")
    tool.definition.max_result_chars = 1000
    tool.definition.preview_chars = 64
    provider = StaticProvider(
        tool_response(ToolUseBlock(id=_CALL, name="feedback_source", input={})),
        text_response("完成"),
    )
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, registry=registry, provider=provider),
        store=lifecycle_store,
    )
    try:
        result = await runner.start(
            AgentRunRequest(input="检查反馈", run_id="feedback-run", session_id=_SESSION)
        )
    finally:
        await runner.aclose()
    assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
    assert effects == ["executed"]

    store = _reopen(lifecycle_store)
    record = store.load_tool_call("feedback-run", _CALL)
    assert record is not None and record.result is not None
    saved = record.result
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
        assert saved.artifact.text_path.read_text(encoding="utf-8") == raw.model_content
        assert len(saved.model_content) <= 1000
    else:
        assert saved.artifact is None
        assert saved.model_content == raw.model_content
    access = ContextAccess(store)
    full_text = _read_full(access, f"result:{message_index}:{block_index}", tmp_path)
    assert full_text == raw.model_content
    assert full_text.count("[Hook feedback]") == 2
    assert all(full_text.count(item) == 1 for item in feedback)
    message_text = _read_full(access, f"message:{message_index}", tmp_path)
    assert message_text.endswith(saved.model_content)
    assert effects == ["executed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_compaction_receives_feedback_and_preserves_original_reads(
    tmp_path: Path, lifecycle_store: LifecycleStore, failed: bool
) -> None:
    """反馈进入真实摘要输入；压缩仅改变模型视图，原结果与原文仍可回读。"""
    body = "原始资料" * 230
    raw = ToolResult(
        tool_use_id="",
        tool_name="feedback_source",
        content=[TextBlock(text=body)],
        is_error=failed,
        error=ToolErrorInfo(code="SOURCE_FAILED", message=body) if failed else None,
        hook_feedback=_FEEDBACK,
    )
    effects: list[str] = []

    def feedback_source() -> ToolResult:
        """用超过上下文预算、未超过 artifact 阈值的结果触发真实压缩。"""
        effects.append("executed")
        return raw

    registry = ToolRegistry()
    registry.register_function(feedback_source, description="提供需要压缩的反馈结果")
    provider = CompactionProvider(
        tool_response(ToolUseBlock(id=_CALL, name="feedback_source", input={})),
        text_response("压缩后完成"),
    )
    runtime = build_runtime(tmp_path, registry=registry, provider=provider)
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
    assert effects == ["executed"]
    assert len(provider.summary_requests) == 1

    store = _reopen(lifecycle_store)
    message_index, block_index, block = _result_block(store)
    session = store.load_session(_SESSION)
    assert session is not None and session.compaction is not None
    assert session.compaction.covered_message_count > message_index
    summary_input = "\n".join(
        message.text for request in provider.summary_requests for message in request.messages
    )
    assert raw.model_content in summary_input
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
    assert block.text == record.result.model_content == raw.model_content
    access = ContextAccess(store)
    assert (
        _read_full(access, f"result:{message_index}:{block_index}", tmp_path) == raw.model_content
    )
    assert _read_full(access, f"message:{message_index}", tmp_path).endswith(raw.model_content)
    assert effects == ["executed"]
