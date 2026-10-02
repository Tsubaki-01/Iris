"""真实 YAML 工厂链的并发工具反馈、取消与 SQLite 回读。"""

from __future__ import annotations

import asyncio
import importlib
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from iris.command import (
    CommandBinding,
    CommandConfig,
    CommandEnvironment,
    CommandMode,
    CommandOutcome,
    CommandOutputStats,
    CommandStatus,
)
from iris.exceptions import IrisCommandCleanupError
from iris.harness import AgentRunner
from iris.hooks import HookEvent, HookRegistration, ToolAfterResult
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunLimits, RunPhase, RunStopReason
from iris.message import ToolUseBlock
from iris.runtime import _assembly
from iris.store import SQLiteStore

from .fakes import FrozenClock, StaticProvider, text_response, tool_batch_response, tool_response
from .test_command_settlement import ControlledService

_EXTENSIONS = '''"""应用自己的配置工厂；每次装配创建一次实例。"""
import asyncio
from typing import cast
from iris.hooks import HookEvent, HookHandler, HookResult, ToolAfterEvent, ToolAfterResult
from iris.message import TextBlock
from iris.tools import ToolCall, ToolCapability, ToolMiddleware, ToolNext, ToolRegistry, ToolResult

feedback = "已核对资料。" * 1000
created: list[str] = []
effects: list[str] = []
wrapped: list[str] = []
arrived: set[str] = set()
drained: set[str] = set()
later: list[str] = []
ready = asyncio.Event()

def body(label: str) -> ToolResult:
    """返回有确定结果的只读工具正文。"""
    effects.append(label)
    return ToolResult(tool_use_id="", tool_name="body", content=[TextBlock(text="body:" + label)])

def register_tools(registry: ToolRegistry) -> None:
    """通过现有registrar接口显式声明并发能力与输出额度。"""
    tool = registry.register_function(
        body, capabilities={ToolCapability.READ}, concurrency_safe=True,
    )
    tool.definition.max_result_chars = 1000
    tool.definition.preview_chars = 64

def create_hook(kind: str) -> HookHandler:
    """工厂只构造处理器，不提前执行事件。"""
    created.append(kind)

    async def handle(event: HookEvent) -> HookResult:
        call = cast(ToolAfterEvent, event)
        if kind == "feedback":
            return ToolAfterResult(feedback=feedback)
        if kind == "block":
            arrived.add(call.call_id)
            if len(arrived) == 2:
                ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.add(call.call_id)
        else:
            later.append(call.call_id)
        return None

    return handle

class Wrapper(ToolMiddleware):
    """每调用仅推进一次下游。"""

    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        wrapped.append(call.tool_use_id)
        return await call_next()

def create_middleware() -> ToolMiddleware:
    """同一装配复用的包装器实例。"""
    created.append("middleware")
    return Wrapper()
'''


@pytest.mark.asyncio
async def test_yaml_parallel_long_feedback_is_committed_before_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """长反馈只落一次artifact，取消停止后项，重建Runner只读取已提交结果。"""
    module_name = "hook_control_" + uuid4().hex
    (tmp_path / f"{module_name}.py").write_text(_EXTENSIONS, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "config-control",
                "model": "openai/fake",
                "system": "test",
                "context_policy": {"enabled": False},
                "permissions": {"workspace": str(tmp_path), "writes": "allow"},
                "tools": {"python": {"registrars": [f"{module_name}:register_tools"]}},
                "hooks": [
                    {
                        "name": kind,
                        "event": "tool.after",
                        "handler": {
                            "type": "python",
                            "factory": f"{module_name}:create_hook",
                            "options": {"kind": kind},
                        },
                    }
                    for kind in ("feedback", "block", "later")
                ],
                "middleware": {"tools": [{"factory": f"{module_name}:create_middleware"}]},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    provider = StaticProvider(
        tool_batch_response(
            *[ToolUseBlock(id=label, name="body", input={"label": label}) for label in ("a", "b")]
        )
    )
    store = SQLiteStore(tmp_path / "lifecycle.db")
    runner = AgentRunner.from_config_path(config_path, provider=provider, store=store)
    extension = importlib.import_module(module_name)
    task = asyncio.create_task(runner.start(AgentRunRequest(input="go", run_id="parallel")))
    try:
        await asyncio.wait_for(extension.ready.wait(), 3)
        runner.request_cancel("parallel")
        result = await asyncio.wait_for(task, 3)
        assert result.run.stop_reason is RunStopReason.CANCELLED
        assert sorted(extension.created) == ["block", "feedback", "later", "middleware"]
        assert sorted(extension.effects) == ["a", "b"]
        assert sorted(extension.wrapped) == ["a", "b"]
        assert extension.drained == {"a", "b"} and extension.later == []
        assert len(provider.requests) == 1
        records = SQLiteStore(store.path).list_tool_calls("parallel")
        assert [record.tool_call_id for record in records] == ["a", "b"]
        for record in records:
            saved = record.result
            assert saved is not None and not saved.is_error
            assert saved.artifact is not None and saved.artifact.text_path is not None
            full = saved.artifact.text_path.read_text(encoding="utf-8")
            assert full.startswith("body:" + record.tool_call_id)
            assert full.count("[Hook feedback]") == 1
            assert full.count(extension.feedback) == 1
            assert saved.hook_feedback == () and len(saved.model_content) <= 1000
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runner.aclose()
    rebuilt = AgentRunner.from_config_path(
        config_path, provider=StaticProvider(), store=SQLiteStore(store.path)
    )
    try:
        assert (await rebuilt.recover("parallel")) == result
        assert sorted(extension.effects) == ["a", "b"]
        assert extension.later == []
    finally:
        await rebuilt.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["cancel", "deadline", "sdk"])
async def test_public_sdk_after_cleanup_retry_preserves_original_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """真实公共配置入口下，after清理失败仍保存结果、原意图，并只重试收口。"""
    (tmp_path / "input.txt").write_text("known file body", encoding="utf-8")
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "sdk-control",
                "model": "openai/fake",
                "system": "test",
                "context_policy": {"enabled": False},
                "permissions": {"workspace": str(tmp_path), "writes": "allow"},
                "tools": {"builtin": ["file.read"]},
            }
        ),
        encoding="utf-8",
    )
    service = ControlledService()
    service.release()

    def binding(config: CommandConfig, workspace_root: Path, *, writable: bool) -> CommandBinding:
        return CommandBinding(
            config,
            service,
            CommandEnvironment("Linux", CommandMode.NATIVE, "Linux", "/bin/sh"),
        )

    monkeypatch.setattr(_assembly, "_create_command_binding", binding)
    entered = asyncio.Event()
    calls: list[str] = []
    cleanup = IrisCommandCleanupError(
        "SDK after cleanup pending",
        command_outcome=CommandOutcome(
            CommandMode.NATIVE,
            CommandStatus.EXITED,
            0,
            "",
            "",
            CommandOutputStats(0, 0, 0, 0, frozenset()),
            0,
            ".",
            service.receipt,
        ),
    )

    async def feedback(event: HookEvent) -> ToolAfterResult:
        calls.append("feedback")
        return ToolAfterResult(feedback="kept SDK feedback")

    async def post(event: HookEvent) -> None:
        calls.append("post")
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise cleanup from None

    provider = StaticProvider(
        tool_response(ToolUseBlock(id="read", name="read_file", input={"file_path": "input.txt"})),
        text_response("recovered"),
    )
    clock = FrozenClock()
    store = SQLiteStore(tmp_path / "lifecycle.db")
    runner = AgentRunner.from_config_path(
        config_path,
        provider=provider,
        store=store,
        clock=clock,
        hooks=[
            HookRegistration(event="tool.after", name="feedback", handler=feedback),
            HookRegistration(event="tool.after", name="post", handler=post),
        ],
    )
    task = asyncio.create_task(
        runner.start(
            AgentRunRequest(input="go", run_id="post"),
            options=AgentRunOptions(
                limits=RunLimits(
                    deadline_at=clock.now() + timedelta(seconds=30)
                    if source == "deadline"
                    else None,
                )
            ),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 3)
        if source == "sdk":
            task.cancel()
        elif source == "deadline":
            clock.advance(seconds=31)
            await runner._command_deadline_due("post")
        else:
            runner.request_cancel("post")
        with pytest.raises(IrisCommandCleanupError) as caught:
            await asyncio.wait_for(task, 3)
        assert caught.value is cleanup
        assert runner.get_run("post").phase is RunPhase.ACTIVE
        pending = runner._command_lifecycle.pending["post"]
        expected = {
            "cancel": RunStopReason.CANCELLED,
            "deadline": RunStopReason.DEADLINE_EXCEEDED,
            "sdk": None,
        }[source]
        assert pending.stop_reason is expected and pending.receipt is service.receipt
        saved = SQLiteStore(store.path).load_tool_call("post", "read").result
        assert saved is not None and "known file body" in saved.model_content
        assert saved.hook_feedback == ("kept SDK feedback",)
        assert len(provider.requests) == 1
        result = await runner.recover(
            "post", expected_activation_id=runner.get_run("post").current_activation_id
        )
        assert result.run.stop_reason is (RunStopReason.COMPLETED if source == "sdk" else expected)
        assert len(provider.requests) == (2 if source == "sdk" else 1)
        assert calls == ["feedback", "post"]
        assert SQLiteStore(store.path).load_tool_call("post", "read").result == saved
        assert not runner._command_lifecycle.pending
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runner.aclose()
