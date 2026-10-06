"""确定性减载与原文摘要、memory 专用额度的集成边界。"""

from math import ceil
from pathlib import Path
from typing import Any

import pytest
from fakes import (
    FakeProvider,
    FakeRuntimeCommitPort,
    MutableCancellationSignal,
    build_runtime,
    start_activation,
)

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.harness._context_access import ContextAccess
from iris.lifecycle import RuntimeExecutionOptions, SessionContextWindow
from iris.memory import MemoryService, SQLiteMemoryStore
from iris.message import (
    ImageBlock,
    ImageFileRef,
    LLMRequest,
    LLMResponse,
    Msg,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from iris.runtime import RuntimeActivationOutcome
from iris.runtime._tool_context import select_tool_context
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolRegistry, ToolResult
from iris.tools.context_access import ContextReadTool


def _batch(text: str, call_id: str) -> list[Msg]:
    return [
        Msg.assistant([ToolUseBlock(id=call_id, name="read", input={"path": "x"})]),
        Msg.tool_result(
            tool_use_id=call_id,
            name="read",
            content=text,
            metadata={"extra": {"context_retention": "observation", "context_tool_name": "read"}},
        ),
    ]


class CountingProvider:
    """每次计量完整工具/schema及正文，摘要与主响应均可控。"""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.estimates: list[LLMRequest] = []

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        self.estimates.append(request)
        return len(str(request.tools)) + sum(
            len(
                block.text
                if isinstance(block, ToolResultBlock)
                else block.text
                if isinstance(block, TextBlock)
                else str(block.input)
            )
            for message in request.messages
            for block in message.blocks
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            provider="test",
            finish_reason="stop",
            content=[
                TextBlock(
                    text="summary" if request.provider_options.get("num_retries") == 0 else "done"
                )
            ],
        )


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ContextReadTool(ContextAccess(InMemoryLifecycleStore())))
    return registry


@pytest.mark.asyncio
async def test_images_remain_visible_across_ordinary_tool_steps(tmp_path: Path) -> None:
    """主请求不会按看图次数移走图片，用户图和已提交工具图持续随历史发送。"""
    ref = ImageFileRef(path=tmp_path / "plot.png", mime_type="image/png", width=40, height=20)
    image = ImageBlock(original=ref, model=ref)
    provider = FakeProvider(
        [
            LLMResponse(
                provider="test",
                finish_reason="tool_calls",
                content=[ToolUseBlock(id=call_id, name="capture")],
            )
            for call_id in ("first", "second")
        ]
        + [LLMResponse(provider="test", finish_reason="stop", content=[TextBlock(text="done")])]
    )

    def capture() -> ToolResult:
        """返回已有图片引用，不执行文件副作用。"""
        return ToolResult(tool_use_id="", tool_name="capture", content=[image])

    registry = ToolRegistry()
    registry.register_function(capture)
    runtime = build_runtime(
        workspace_root=tmp_path,
        agent_config=AgentConfig(name="images", model="openai/test", system="rules"),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="rules", content="rules")])
        ),
        provider=provider,
        tool_registry=registry,
    )
    activation = start_activation().model_copy(update={"run_input": [image]})
    port = FakeRuntimeCommitPort(activation)
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert len(provider.requests) == 3 and port.compaction_commits == []
    for index, request in enumerate(provider.requests):
        assert [
            block
            for message in request.messages
            for block in message.blocks
            if isinstance(block, ImageBlock)
        ] == [image]
        assert [
            part
            for message in request.messages
            for block in message.tool_results
            for part in block.content
            if isinstance(part, ImageBlock)
        ] == [image] * index
    assert len(port.tool_commits) == 2


@pytest.mark.asyncio
async def test_remaining_pressure_summarizes_raw_body_and_accepts_final_projection(
    tmp_path: Path,
) -> None:
    provider = CountingProvider()
    raw = [
        *_batch("result-A-full-prefix" + "A" * 3000, "a"),
        *_batch("result-B-full-prefix" + "B" * 3000, "b"),
        *_batch("result-C-full-prefix" + "C" * 3000, "c"),
    ]
    runtime = build_runtime(
        workspace_root=tmp_path,
        agent_config=AgentConfig.model_validate(
            {
                "name": "summary",
                "model": "openai/test",
                "system": "rules",
                "compaction": {"input_budget_tokens": 10000},
            }
        ),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="rules", content="fixed" * 800)])
        ),
        provider=provider,
        tool_registry=_registry(),
    )
    activation = start_activation(input="continue", initial_session_message_count=len(raw))
    port = FakeRuntimeCommitPort(activation, messages=raw)
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    summaries = [
        request for request in provider.requests if request.provider_options.get("num_retries") == 0
    ]
    assert summaries and len(port.compaction_commits) == 1
    material = "\n".join(request.messages[1].text for request in summaries)
    assert "result-A-full-prefix" in material
    assert "历史正文已移出当前窗口" not in material
    assert any(
        "历史正文已移出当前窗口" in block.text
        for request in provider.estimates
        for message in request.messages
        for block in message.tool_results
    )
    assert (
        provider.estimate_input_tokens(provider.requests[-1])
        <= runtime.environment.agent_config.compaction.trigger_tokens
    )
    assert port.messages[: len(raw)] == raw


@pytest.mark.asyncio
async def test_memory_delta_uses_unpruned_history_then_final_window_is_reduced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.runtime.runtime as runtime_module

    full = SessionContextWindow(memory_overview="FULL" * 1000, mode="full")
    navigation = SessionContextWindow(memory_overview="NAV" * 30, mode="navigation")

    async def windows(**kwargs: Any) -> tuple[SessionContextWindow, SessionContextWindow]:
        return full, navigation

    monkeypatch.setattr(runtime_module, "load_context_windows", windows)
    provider = CountingProvider()
    raw = [*_batch("body" * 750, "a"), *_batch("body" * 750, "b")]
    config = AgentConfig.model_validate(
        {
            "name": "memory-boundary",
            "model": "openai/test",
            "system": "rules",
            "context_policy": {"preserve_recent_tool_groups": 0},
        }
    )
    runtime = build_runtime(
        workspace_root=tmp_path,
        agent_config=config,
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="rules", content="fixed")])
        ),
        provider=provider,
        tool_registry=_registry(),
        memory_service=MemoryService(SQLiteMemoryStore(tmp_path / "memory.db")),
    )
    base, _ = runtime._build_model_request(
        history=[*raw, Msg.user("continue")],
        options=RuntimeExecutionOptions(),
        context_window=SessionContextWindow(),
        tool_selection=select_tool_context(
            runtime.environment.tool_bridge.tool_view,
            None,
            include_tools=True,
            tool_choice=None,
        ),
    )
    baseline = provider.estimate_input_tokens(base)
    runtime.environment.agent_config = config.model_copy(
        update={
            "compaction": config.compaction.model_copy(
                update={"input_budget_tokens": ceil((baseline + 50) / 0.8)}
            )
        }
    )
    activation = start_activation(input="continue", initial_session_message_count=len(raw))
    port = FakeRuntimeCommitPort(activation, messages=raw)
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert port.context_window is navigation
    assert len(provider.requests) == 1 and not port.compaction_commits
    actual = provider.requests[0]
    assert actual.messages[0].text.endswith(navigation.memory_overview)
    assert any(
        "重复正文见" in block.text for message in actual.messages for block in message.tool_results
    )
    assert (
        provider.estimate_input_tokens(actual)
        < runtime.environment.agent_config.compaction.trigger_tokens
    )
    full_trials = [
        request
        for request in provider.estimates
        if full.memory_overview in request.messages[0].text
    ]
    assert full_trials
    assert all(
        sum(len(message.tool_results) for message in request.messages) == 2
        for request in full_trials
    )
    assert all(
        block.text == "body" * 750
        for request in full_trials
        for message in request.messages
        for block in message.tool_results
    )
