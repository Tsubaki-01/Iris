"""本步快照贯穿摘要重试与新 memory 窗口，选材不回填也不进入摘要原料。"""

from pathlib import Path
from typing import Any

import pytest
from fakes import FakeRuntimeCommitPort, MutableCancellationSignal, build_runtime, start_activation

from iris.agents import AgentConfig
from iris.context import (
    ContextBuildInput,
    ContextBuildScope,
    ContextContribution,
    ContextSection,
    ContextSlot,
    ContextSnapshot,
)
from iris.exceptions import IrisAPIConnectionError
from iris.lifecycle import RuntimeExecutionOptions, SessionContextWindow
from iris.memory import MemoryService, SQLiteMemoryStore
from iris.message import LLMRequest, LLMResponse, Msg
from iris.runtime import RuntimeActivationOutcome, RuntimeCursor
from tests.runtime.test_context_projection_execution import CountingProvider


@pytest.mark.asyncio
async def test_once_collect_across_summary_retry_frozen_selection_and_new_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris.runtime.runtime as runtime_module

    full = SessionContextWindow(memory_overview="NEW-FULL" * 50, mode="full")
    navigation = SessionContextWindow(memory_overview="NEW-NAV" * 20, mode="navigation")

    async def windows(**kwargs: Any) -> tuple[SessionContextWindow, SessionContextWindow]:
        return full, navigation

    monkeypatch.setattr(runtime_module, "load_context_windows", windows)

    class RetryProvider(CountingProvider):
        retried = False

        async def complete(self, request: LLMRequest) -> LLMResponse:
            if request.provider_options.get("num_retries") == 0 and not self.retried:
                self.retried = True
                self.requests.append(request)
                raise IrisAPIConnectionError("one retry")
            return await super().complete(request)

    class Source:
        calls = 0

        async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
            self.calls += 1
            return ContextSnapshot(
                (
                    ContextContribution("required-state", "current-only-evidence" * 20),
                    ContextContribution("optional-not-refilled", "discarded" * 500, required=False),
                )
            )

    provider, source = RetryProvider(), Source()
    runtime = build_runtime(
        agent_config=AgentConfig(
            name="source-budget",
            model="openai/test",
            system="stable",
            compaction={"input_budget_tokens": 10000},
        ),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="rules", content="stable")])
        ),
        provider=provider,
        memory_service=MemoryService(SQLiteMemoryStore(tmp_path / "memory.db")),
    )
    runtime.environment.context_source = source
    activation = start_activation(
        input="original",
        initial_session_message_count=1,
        options=RuntimeExecutionOptions(include_tools=False),
    ).model_copy(
        update={
            "kind": "recover",
            "cursor": RuntimeCursor(position="before_model", step_index=0),
        }
    )
    raw = [Msg.assistant("archived evidence" * 1000), Msg.user("original")]
    port = FakeRuntimeCommitPort(activation, messages=raw)
    port.context_window = SessionContextWindow(memory_overview="OLD", mode="full")
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    assert source.calls == 1 and len(port.compaction_commits) == 1
    summaries = [
        request for request in provider.requests if request.provider_options.get("num_retries") == 0
    ]
    assert len(summaries) >= 3 and summaries[0].messages == summaries[1].messages
    assert all(
        "current-only-evidence" not in message.text and "optional-not-refilled" not in message.text
        for request in summaries
        for message in request.messages
    )
    final = provider.requests[-1]
    assert final.messages[0].text.endswith(navigation.memory_overview)
    assert "current-only-evidence" in final.messages[-1].text
    assert "optional-not-refilled" not in final.messages[-1].text
    assert (
        sum(
            message.metadata.get("context_kind") == "runtime_snapshot" for message in final.messages
        )
        == 1
    )
    assert (
        provider.estimate_input_tokens(final)
        <= runtime.environment.agent_config.compaction.trigger_tokens
    )
    assert port.messages[:2] == raw
    candidates = [
        request
        for request in provider.estimates
        if request.messages[0].text.endswith(full.memory_overview)
        or request.messages[0].text.endswith(navigation.memory_overview)
    ]
    assert candidates
    assert all(request.messages[-1].text == final.messages[-1].text for request in candidates)
    assert all(request.messages[1:-1] == final.messages[1:-1] for request in candidates)


@pytest.mark.asyncio
async def test_required_snapshot_cannot_be_removed_to_fit_hard_budget() -> None:
    class Source:
        async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
            return ContextSnapshot((ContextContribution("required", "must keep" * 1500),))

    provider = CountingProvider()
    runtime = build_runtime(
        agent_config=AgentConfig(
            name="required",
            model="openai/test",
            system="stable",
            compaction={"input_budget_tokens": 10000},
        ),
        context_input=ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="rules", content="stable")])
        ),
        provider=provider,
    )
    runtime.environment.context_source = Source()
    activation = start_activation(input="original")
    port = FakeRuntimeCommitPort(activation)
    result = await runtime.execute(
        activation, commits=port, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert (
        result.error.source == "context" and result.error.code == "CONTEXT_COMPACTION_UNAVAILABLE"
    )
    assert not provider.requests
    assert [message.text for message in port.messages] == ["original"]
