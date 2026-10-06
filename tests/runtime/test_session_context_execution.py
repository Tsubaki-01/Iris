"""有效历史接入主请求时保留首次窗口预算与旧发现事实。"""

from pathlib import Path
from typing import Any

import pytest
from fakes import FakeRuntimeCommitPort, MutableCancellationSignal, build_runtime, start_activation

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.lifecycle import RuntimeExecutionOptions, SessionCompaction, SessionContextWindow
from iris.memory import MemoryService, SQLiteMemoryStore
from iris.message import LLMRequest, LLMResponse, Msg, TextBlock, ToolUseBlock
from iris.runtime import RuntimeActivationOutcome
from iris.runtime._context_projection import project_context_request
from iris.runtime._request_measurement import measure_request
from iris.runtime._tool_context import select_tool_context
from iris.tools import ToolRegistry


@pytest.mark.asyncio
async def test_initial_memory_budget_includes_discovery_covered_by_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """继承摘要后首次采用窗口也必须计入已发现工具的完整 schema。"""
    import iris.runtime.runtime as runtime_module

    full = SessionContextWindow(memory_overview="FULL", mode="full")
    navigation = SessionContextWindow(memory_overview="NAV", mode="navigation")

    async def windows(**kwargs: Any) -> tuple[SessionContextWindow, SessionContextWindow]:
        return full, navigation

    monkeypatch.setattr(runtime_module, "load_context_windows", windows)

    class Provider:
        """工具成本使 full 超总额，但 navigation 仍可供后续选材。"""

        def __init__(self) -> None:
            self.estimates: list[LLMRequest] = []

        def estimate_input_tokens(self, request: LLMRequest) -> int:
            self.estimates.append(request)
            base = (
                950 if any(tool.name == "lookup" for tool in request.tools) else 100
            )
            system = request.messages[0].text
            return base + (60 if "FULL" in system else 20 if "NAV" in system else 0)

        async def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(provider="test", content=[TextBlock(text="完成")])

    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="lookup", description="检索", deferred=True)
    provider = Provider()
    runtime = build_runtime(
        workspace_root=tmp_path,
        agent_config=AgentConfig(
            name="window-discovery",
            model="openai/test",
            system="rules",
            context_policy={"deferred_tools": True},
            compaction={"input_budget_tokens": 1000},
            memory={"overview": {"system_budget_ratio": 0.5}},
        ),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="rules", content="rules")])
        ),
        provider=provider,
        tool_registry=registry,
        memory_service=MemoryService(SQLiteMemoryStore(tmp_path / "memory.db")),
    )
    raw = [
        Msg.user("旧任务"),
        Msg.assistant([ToolUseBlock(id="search", name="tool_search")]),
        Msg.tool_result(
            tool_use_id="search",
            name="tool_search",
            content="已发现lookup",
            metadata={"extra": {"context_revealed_tools": ["lookup"]}},
        ),
        Msg.assistant("旧任务完成"),
    ]
    activation = start_activation(input="新任务", initial_session_message_count=len(raw))
    port = FakeRuntimeCommitPort(activation, messages=raw)
    port.compaction = SessionCompaction(summary="旧摘要", covered_message_count=len(raw))
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED, result.error
    assert port.input_commits[0].initial_context_window == navigation
    full_candidates = [
        request for request in provider.estimates if "FULL" in request.messages[0].text
    ]
    assert full_candidates
    assert all(
        any(tool.name == "lookup" for tool in request.tools)
        for request in full_candidates
    )


@pytest.mark.asyncio
async def test_memory_allowance_cannot_use_body_savings_triggered_only_by_full_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """full触发正文短化而base不触发时，释放的历史额度不能抵扣概览成本。"""
    import iris.runtime.runtime as runtime_module

    full = SessionContextWindow(memory_overview="FULL", mode="full")
    navigation = SessionContextWindow(memory_overview="NAV", mode="navigation")

    async def windows(**kwargs: Any) -> tuple[SessionContextWindow, SessionContextWindow]:
        return full, navigation

    monkeypatch.setattr(runtime_module, "load_context_windows", windows)
    body = "original observation " * 100
    history = [
        Msg.assistant([ToolUseBlock(id="read", name="read_file", input={"path": "notes"})]),
        Msg.tool_result(
            tool_use_id="read",
            name="read_file",
            content=body,
            metadata={
                "extra": {"context_retention": "observation", "context_tool_name": "read_file"}
            },
        ),
        Msg.user("current input"),
    ]

    class Provider:
        """完整原请求为700，full增加200；正文短化恰好释放200。"""

        def __init__(self) -> None:
            self.estimates: list[LLMRequest] = []

        def estimate_input_tokens(self, request: LLMRequest) -> int:
            self.estimates.append(request)
            original = any(
                result.text == body
                for message in request.messages
                for result in message.tool_results
            )
            system = request.messages[0].text
            return (700 if original else 500) + (
                200 if system.endswith("FULL") else 20 if system.endswith("NAV") else 0
            )

        async def complete(self, request: LLMRequest) -> LLMResponse:
            raise AssertionError("窗口预算采用不应调用模型")

    provider = Provider()
    registry = ToolRegistry()
    registry.register_function(lambda: "unused", name="context_read", description="read history")
    config = AgentConfig(
        name="raw-memory-budget",
        model="openai/test",
        system="rules",
        compaction={"input_budget_tokens": 1000},
        context_policy={"preserve_recent_tool_groups": 0, "old_result_preview_chars": 0},
        memory={"overview": {"system_budget_ratio": 0.05}},
    )
    runtime = build_runtime(
        workspace_root=tmp_path,
        agent_config=config,
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="rules", content="rules")])
        ),
        provider=provider,
        tool_registry=registry,
        memory_service=MemoryService(SQLiteMemoryStore(tmp_path / "memory.db")),
    )
    window, measured = await runtime._adopt_context_window(
        history=history,
        options=RuntimeExecutionOptions(),
        input_budget_tokens=config.compaction.input_budget_tokens,
        tool_selection=select_tool_context(
            registry.view(), None, include_tools=True, tool_choice=None
        ),
    )
    base_request, full_request, navigation_request = provider.estimates
    assert all(request.messages[1:] == history for request in provider.estimates)
    assert all(request.tools == base_request.tools for request in provider.estimates)
    assert window is navigation and measured.request is navigation_request
    assert measured.input_tokens == 720

    source_indices = {id(message): index for index, message in enumerate(history)}
    base = measure_request(base_request, provider.estimate_input_tokens)
    full_candidate = measure_request(full_request, provider.estimate_input_tokens)
    projected_base, _ = project_context_request(
        base,
        source_indices=source_indices,
        config=config.context_policy,
        trigger_tokens=config.compaction.trigger_tokens,
        estimate_input_tokens=provider.estimate_input_tokens,
    )
    projected_full, _ = project_context_request(
        full_candidate,
        source_indices=source_indices,
        config=config.context_policy,
        trigger_tokens=config.compaction.trigger_tokens,
        estimate_input_tokens=provider.estimate_input_tokens,
    )
    assert base.input_tokens < config.compaction.trigger_tokens <= full_candidate.input_tokens
    assert projected_base is base
    assert projected_full.request.messages[2].tool_results[0].text != body
    memory_allowance = (
        config.compaction.input_budget_tokens * config.memory.overview.system_budget_ratio
    )
    assert full_candidate.input_tokens - base.input_tokens > memory_allowance
    assert projected_full.input_tokens - projected_base.input_tokens <= memory_allowance
