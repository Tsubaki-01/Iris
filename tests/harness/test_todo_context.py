"""真实 Runner 与文件工具下的 Todo 快照、读写约束和恢复边界。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.exceptions import IrisRunPersistenceError
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    CommitToolResult,
    RunCommit,
    RunLimits,
    RunStopReason,
    ToolCallPhase,
)
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import SQLiteStore
from iris.todo import TodoStatus
from iris.tools import WorkspaceFileService

from .fakes import StaticProvider, text_response, tool_response


def _config(workspace: Path) -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "name": "todo-files",
            "model": "openai/test",
            "system": "根据真实文件状态维护当前工作清单。",
            "todo": {"enabled": True},
            "permissions": {"workspace": str(workspace), "writes": "allow"},
            "tools": {"builtin": ["file.read", "file.write"]},
        }
    )


def _call(call_id: str, path: Path, content: str | None = None) -> LLMResponse:
    arguments = {"file_path": str(path)}
    if content is not None:
        arguments["content"] = content
    return tool_response(
        ToolUseBlock(
            id=call_id,
            name="read_file" if content is None else "write_file",
            input=arguments,
        )
    )


def _snapshot_text(request: LLMRequest) -> str:
    [snapshot] = [
        message
        for message in request.messages
        if message.metadata.get("context_kind") == "runtime_snapshot"
    ]
    assert "iris.todo" in snapshot.text
    return snapshot.text


def _options(steps: int) -> AgentRunOptions:
    return AgentRunOptions(limits=RunLimits(max_model_steps=steps))


@pytest.mark.asyncio
@pytest.mark.parametrize("removal", ["empty", "delete"])
async def test_file_tools_refresh_todo_and_human_changes_do_not_hydrate_history(
    tmp_path: Path, removal: str
) -> None:
    """真实读写下步可见，人工替换和清空后不从旧工具参数恢复清单。"""
    provider = StaticProvider()
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider)
    try:
        path = (await runner.get_todo("work")).path
        path.parent.mkdir(parents=True)
        path.write_text("- [ ] 核对集成结果\n", encoding="utf-8")
        provider.responses.extend(
            [
                _call("read-plan", path),
                _call("finish-plan", path, "- [x] 核对集成结果\n"),
                text_response(),
                text_response(),
                text_response(),
            ]
        )
        result = await runner.start(
            AgentRunRequest(input="完成清单", session_id="work", run_id="finish"),
            options=_options(3),
        )
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert all(not record.result.is_error for record in runner.list_tool_calls("finish"))
        assert "- [ ] 核对集成结果" in _snapshot_text(provider.requests[0])
        assert "- [ ] 核对集成结果" in _snapshot_text(provider.requests[1])
        assert "- [x] 核对集成结果" in _snapshot_text(provider.requests[2])
        assert "- [ ] 核对集成结果" not in _snapshot_text(provider.requests[2])
        assert (await runner.get_todo("work")).items[0].status is TodoStatus.COMPLETED

        path.write_text("- [-] 人工重新安排工作\n", encoding="utf-8")
        await runner.start(
            AgentRunRequest(input="查看当前计划", session_id="work"), options=_options(1)
        )
        assert "- [-] 人工重新安排工作" in _snapshot_text(provider.requests[3])
        assert "核对集成结果" not in _snapshot_text(provider.requests[3])
        if removal == "empty":
            path.write_text("", encoding="utf-8")
        else:
            path.unlink()
        await runner.start(
            AgentRunRequest(input="继续普通聊天", session_id="work"), options=_options(1)
        )
        current = _snapshot_text(provider.requests[4])
        assert str(path) in current
        assert "人工重新安排工作" not in current and "核对集成结果" not in current
        assert (await runner.get_todo("work")).items == ()
        assert path.read_text(encoding="utf-8") == "" if removal == "empty" else not path.exists()
        history = runner.get_session("work").messages
        assert all(
            message.metadata.get("context_kind") != "runtime_snapshot" for message in history
        )
        history_content = "\n".join(message.model_dump_json() for message in history)
        assert "核对集成结果" in history_content
        assert "人工重新安排工作" not in history_content
    finally:
        await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("read_previous_run", [False, True])
async def test_snapshot_and_previous_run_do_not_grant_file_read_state(
    tmp_path: Path, read_previous_run: bool
) -> None:
    """动态上下文和上一 Run 的 read_file 都不能冒充当前 Run 的文件读取。"""
    provider = StaticProvider()
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider)
    try:
        path = (await runner.get_todo("work")).path
        path.parent.mkdir(parents=True)
        initial = "- [ ] 尚未完成的任务\n"
        path.write_text(initial, encoding="utf-8")
        if read_previous_run:
            provider.responses.extend([_call("previous-read", path), text_response()])
            await runner.start(
                AgentRunRequest(input="读取文件", session_id="work"), options=_options(2)
            )
        provider.responses.extend(
            [_call("unread-write", path, "- [x] 尚未完成的任务\n"), text_response()]
        )
        await runner.start(
            AgentRunRequest(input="尝试直接修改", session_id="work", run_id="unread"),
            options=_options(2),
        )
        [record] = runner.list_tool_calls("unread")
        assert record.result is not None and record.result.error is not None
        assert record.result.error.code == "FILE_NOT_READ"
        assert record.phase is ToolCallPhase.COMMITTED
        assert path.read_text(encoding="utf-8") == initial
        assert initial.strip() in _snapshot_text(provider.requests[-2])
        assert initial.strip() in _snapshot_text(provider.requests[-1])
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_human_edit_preserves_stale_read_error_until_tool_rereads(tmp_path: Path) -> None:
    """人工改写仍触发普通 stale-read，重新 read_file 后才能真正写成功。"""
    path = tmp_path / ".iris/todos/776f726b.md"
    path.parent.mkdir(parents=True)
    path.write_text("- [ ] 原始工作\n", encoding="utf-8")

    class HumanEditingProvider(StaticProvider):
        """在第二次请求采样完成后模拟人工编辑，保留确定性的时序。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """让旧 read receipt 与磁盘文件发生真实分歧。"""
            if len(self.requests) == 1:
                path.write_text("- [-] 人工修改后的当前工作\n", encoding="utf-8")
            return await super().complete(request)

    provider = HumanEditingProvider(
        _call("initial-read", path),
        _call("stale-write", path, "- [x] 过期的覆盖内容\n"),
        _call("fresh-read", path),
        _call("fresh-write", path, "- [x] 人工修改后的当前工作\n"),
        text_response(),
    )
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider)
    try:
        result = await runner.start(
            AgentRunRequest(input="处理工作清单", session_id="work", run_id="stale"),
            options=_options(5),
        )
        assert result.run.stop_reason is RunStopReason.COMPLETED
        records = runner.list_tool_calls("stale")
        assert [record.result.is_error for record in records] == [False, True, False, False]
        assert records[1].result.error.code == "STALE_FILE_STATE"
        assert "- [ ] 原始工作" in _snapshot_text(provider.requests[1])
        assert "- [-] 人工修改后的当前工作" in _snapshot_text(provider.requests[2])
        assert "- [x] 人工修改后的当前工作" in _snapshot_text(provider.requests[4])
        assert "过期的覆盖内容" not in _snapshot_text(provider.requests[4])
        assert path.read_text(encoding="utf-8") == "- [x] 人工修改后的当前工作\n"
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_file_effect_survives_failed_commit_and_unknown_recovery_without_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """文件已落盘而结果提交失败时，SDK 读当前文件且恢复不重放副作用。"""

    class FailWriteCommitStore(SQLiteStore):
        """只在指定写工具的结果入库前注入故障，留下真实 CLAIMED 记录。"""

        def commit_tool_result(self, command: CommitToolResult) -> RunCommit:
            """读工具正常提交，写工具已经执行完成时模拟进程退出边界。"""
            if command.tool_call_id == "write-before-crash":
                raise IrisRunPersistenceError("write result commit failed")
            return super().commit_tool_result(command)

    writes: list[Path] = []
    original_write = WorkspaceFileService.atomic_write

    def record_write(self: WorkspaceFileService, path: Path, content: str) -> None:
        writes.append(path)
        original_write(self, path, content)

    monkeypatch.setattr(WorkspaceFileService, "atomic_write", record_write)
    database = tmp_path / "lifecycle.db"
    store = FailWriteCommitStore(database)
    provider = StaticProvider()
    first = AgentRunner.from_config(_config(tmp_path), provider=provider, store=store)
    try:
        path = (await first.get_todo("work")).path
        path.parent.mkdir(parents=True)
        path.write_text("- [ ] 原计划\n", encoding="utf-8")
        provider.responses.extend(
            [
                _call("read-before-crash", path),
                _call("write-before-crash", path, "- [x] 已实际完成的新计划\n"),
            ]
        )
        with pytest.raises(IrisRunPersistenceError, match="write result commit failed"):
            await first.start(
                AgentRunRequest(input="更新清单", session_id="work", run_id="crashed"),
                options=_options(3),
            )
        crashed = store.load_run("crashed")
        assert crashed is not None and crashed.current_activation_id is not None
        assert store.load_tool_call("crashed", "write-before-crash").phase is ToolCallPhase.CLAIMED
        assert path.read_text(encoding="utf-8") == "- [x] 已实际完成的新计划\n"
        assert (await first.get_todo("work")).items[0].content == "已实际完成的新计划"
        assert writes == [path]
    finally:
        await first.aclose()

    recovery_provider = StaticProvider(text_response())
    restarted_store = SQLiteStore(database)
    second = AgentRunner.from_config(
        _config(tmp_path), provider=recovery_provider, store=restarted_store
    )
    try:
        recovered = await second.recover(
            "crashed", expected_activation_id=crashed.current_activation_id
        )
        assert recovered.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
        assert recovered.error is not None and recovered.error.code == "TOOL_OUTCOME_UNKNOWN"
        assert recovery_provider.requests == []
        assert writes == [path]
        assert (await second.get_todo("work")).items[0].status is TodoStatus.COMPLETED
        assert (
            restarted_store.load_tool_call("crashed", "write-before-crash").phase
            is ToolCallPhase.OUTCOME_UNKNOWN
        )

        await second.start(
            AgentRunRequest(input="确认当前文件", session_id="work"), options=_options(1)
        )
        assert "- [x] 已实际完成的新计划" in _snapshot_text(recovery_provider.requests[0])
        assert "原计划" not in _snapshot_text(recovery_provider.requests[0])
        assert writes == [path]
    finally:
        await second.aclose()
