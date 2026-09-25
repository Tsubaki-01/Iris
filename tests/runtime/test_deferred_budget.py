"""延迟schema的完整预算、强制目标与compaction冻结选择。"""

from pathlib import Path
from typing import Any

import pytest
from fakes import FakeRuntimeCommitPort, MutableCancellationSignal, build_runtime, start_activation

from iris.agents import AgentConfig, ContextPolicyConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.harness import AgentRunner
from iris.lifecycle import RuntimeExecutionOptions, SessionContextWindow
from iris.memory import MemoryService, SQLiteMemoryStore
from iris.message import LLMRequest, LLMResponse, Msg, ToolUseBlock
from iris.runtime import RuntimeActivationOutcome
from iris.runtime._context_projection import project_context_request
from iris.runtime._tool_context import select_tool_context
from iris.tools import ToolRegistry
from tests.runtime.test_context_projection import _estimate
from tests.runtime.test_context_projection_execution import CountingProvider
from tests.runtime.test_deferred_selection import _search


def test_hundred_deferred_tools_start_hidden_and_latest_first_fits_complete_schema() -> None:
    registry = ToolRegistry()
    for index in range(100):
        registry.register_function(
            lambda: "ok", name=f"tool_{index}", description="D" * 2000, deferred=True
        )
    view = registry.view()
    assert (
        select_tool_context(
            view, [], deferred_tools=True, include_tools=True, tool_choice=None
        ).names
        == ()
    )
    selected = select_tool_context(
        view,
        _search("tool_3", "tool_2", "tool_1"),
        deferred_tools=True,
        include_tools=True,
        tool_choice=None,
    )
    schemas = view.schemas_for(selected.names)
    request, _ = project_context_request(
        LLMRequest(model="test", messages=[Msg.user("question")], tools=schemas),
        source_indices={},
        config=ContextPolicyConfig(),
        trigger_tokens=3000,
        estimate_input_tokens=_estimate,
        select_optional=True,
        optional_tool_names=selected.optional_names,
    )
    assert [schema["function"]["name"] for schema in request.tools] == ["tool_3"]
    assert request.tools[0] == next(
        schema for schema in schemas if schema["function"]["name"] == "tool_3"
    )


@pytest.mark.asyncio
async def test_frozen_schema_selection_survives_summary_and_new_memory_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris.runtime.runtime as runtime_module

    window = SessionContextWindow(memory_overview="NEW" * 20, mode="full")

    async def windows(**kwargs: Any) -> tuple[SessionContextWindow, SessionContextWindow]:
        return window, window

    monkeypatch.setattr(runtime_module, "load_context_windows", windows)

    class Provider(CountingProvider):
        """摘要后发出一次真实可见工具调用。"""

        issued = False

        async def complete(self, request: LLMRequest) -> LLMResponse:
            if request.provider_options.get("num_retries") != 0 and not self.issued:
                self.issued = True
                self.requests.append(request)
                return LLMResponse(provider="test", content=[ToolUseBlock(id="call", name="eager")])
            return await super().complete(request)

    registry = ToolRegistry()
    registry.register_function(lambda: "ok", name="eager", description="eager")
    for name in ("a", "b", "c"):
        registry.register_function(
            lambda: "ok", name=name, description="schema" * 500, deferred=True
        )
    raw = [Msg.assistant("archived" * 2000), *_search("a", "b", "c"), Msg.assistant("finished")]
    provider = Provider()
    runtime = build_runtime(
        agent_config=AgentConfig(
            name="budget",
            model="openai/test",
            system="rules",
            context_policy={"deferred_tools": True},
            compaction={"input_budget_tokens": 10000},
        ),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="rules", content="stable")])
        ),
        provider=provider,
        tool_registry=registry,
        memory_service=MemoryService(SQLiteMemoryStore(tmp_path / "memory.db")),
    )
    activation = start_activation(input="next", initial_session_message_count=len(raw))
    port = FakeRuntimeCommitPort(activation, messages=raw)
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert port.compaction_commits
    first_main = next(
        request for request in provider.requests if request.provider_options.get("num_retries") != 0
    )
    assert [tool["function"]["name"] for tool in first_main.tools] == ["eager"]
    assert first_main.messages[0].text.endswith(window.memory_overview)
    assert port.model_commits[0].cursor_after.visible_tool_names == ("eager",)
    assert port.tool_commits[0].cursor_after.visible_tool_names == ()
    assert port.messages[: len(raw)] == raw


@pytest.mark.asyncio
async def test_effective_forced_target_is_in_schema_and_cursor(tmp_path: Path) -> None:
    from tests.harness.fakes import StaticProvider

    provider = StaticProvider(
        LLMResponse(provider="test", content=[ToolUseBlock(id="forced", name="deferred")]),
        LLMResponse(provider="test", content=[]),
    )
    runner = AgentRunner.from_config(
        AgentConfig(
            name="forced",
            model={
                "provider": "openai",
                "name": "test",
                "tool_choice": {"type": "function", "function": {"name": "missing"}},
            },
            system="rules",
            context_policy={"deferred_tools": True},
        ),
        provider=provider,
    )
    registry = runner.runtime.environment.tool_bridge.tool_view.registry
    calls = []
    registry.register_function(
        lambda: calls.append("executed") or "ok",
        name="deferred",
        description="forced",
        deferred=True,
    )
    activation = start_activation(
        options=RuntimeExecutionOptions(
            request_options={"tool_choice": {"type": "function", "function": {"name": "deferred"}}}
        )
    )
    port = FakeRuntimeCommitPort(activation)
    result = await runner.runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert calls == ["executed"]
    assert "deferred" in port.model_commits[0].cursor_after.visible_tool_names
    assert all(
        request.tool_choice["function"]["name"] == "deferred"
        and any(tool["function"]["name"] == "deferred" for tool in request.tools)
        for request in provider.requests
    )
