"""显式开启的 DeepSeek Todo 文件维护、自查、用户确认与 SQLite 验证。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.config import init_config, reset
from iris.harness import AgentRunner
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunLimits,
    RunResult,
    RunStopReason,
    RunToolCallRecord,
    ToolCallPhase,
)
from iris.store import SQLiteStore
from iris.todo import TodoSnapshot, TodoStatus

pytestmark = [pytest.mark.live_deepseek, pytest.mark.asyncio]

_SESSION = "todo-live"
_SUM_TASK = "计算并核对数字总和"
_CONFIRM_TASK = "等待用户确认结果"
_SYSTEM = """你需要使用实际 read_file/write_file 工具完成文件任务，并维护当前会话 Todo。
不要只输出文件内容或声称已调用工具。按用户要求操作，使用工具返回结果作为完成证据。
需要覆盖现有文件时先在本 Run 用 read_file 读取；当前 Todo 动态快照不能替代文件工具读取。
等待用户确认的事项必须保持 pending，不能因结束自查指令伪造用户确认或标为完成。
收到 Todo 结束自查后，依据当前文件和用户授权核对状态；等待事项如实说明即可结束。
"""


@pytest.fixture
def live_todo_config(request: pytest.FixtureRequest) -> Iterator[None]:
    """显式开关之后才加载凭据，测试与证据输出均不包含配置或密钥。"""
    if not request.config.getoption("--run-live-deepseek"):
        pytest.skip("使用 --run-live-deepseek 显式开启真实 Todo API 调用")
    reset()
    try:
        config = init_config(env_file=".env.local")
        if not (config.provider_api_keys.get("deepseek") or config.api_key):
            pytest.fail(".env.local 中缺少 DeepSeek API 凭据")
        yield
    finally:
        reset()


def _config(workspace: Path) -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "name": "todo-live",
            "model": {
                "provider": "deepseek",
                "name": "deepseek-chat",
                "temperature": 0,
                "max_tokens": 2048,
                "timeout": 60,
            },
            "system": _SYSTEM,
            "todo": {"enabled": True},
            "permissions": {"workspace": str(workspace), "writes": "allow"},
            "tools": {"builtin": ["file.read", "file.write"]},
        }
    )


def _snapshot_evidence(snapshot: TodoSnapshot) -> dict[str, object]:
    return {
        "path": str(snapshot.path),
        "items": [
            {"content": item.content, "status": item.status.value} for item in snapshot.items
        ],
        "error": snapshot.error,
    }


def _successful_file_calls(result: RunResult, store: SQLiteStore) -> list[RunToolCallRecord]:
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert result.error is None
    assert result.run.usage.input_tokens > 0 and result.run.usage.output_tokens > 0
    assert result.run.usage.total_tokens > 0
    assert result.run.usage.model_steps_reserved == result.run.usage.model_steps_committed
    calls = store.list_tool_calls(result.run.run_id)
    assert calls and all(call.phase is ToolCallPhase.COMMITTED for call in calls)
    file_calls = [call for call in calls if call.tool_name in {"read_file", "write_file"}]
    assert {call.tool_name for call in file_calls} == {"read_file", "write_file"}
    assert all(not call.result.is_error for call in file_calls)
    return file_calls


@pytest.mark.usefixtures("live_todo_config")
async def test_real_deepseek_todo_files_reminder_and_user_confirmation(tmp_path: Path) -> None:
    """不注入 provider，联合验证真实模型、普通文件工具、一次自查和次 Run 确认。"""
    store = SQLiteStore(tmp_path / "todo-live.db")
    runner = AgentRunner.from_config(_config(tmp_path), store=store)
    run_evidence: dict[str, object] = {}
    checkpoint_evidence: dict[str, object] = {}
    tool_evidence: dict[str, object] = {}
    todo_evidence: dict[str, object] = {}
    evidence: dict[str, object] = {
        "runs": run_evidence,
        "checkpoints": checkpoint_evidence,
        "tools": tool_evidence,
        "todo": todo_evidence,
    }
    run_ids = ("todo-work", "todo-confirm")
    try:
        initial = await runner.get_todo(_SESSION)
        assert initial.items == () and initial.error is None and not initial.path.exists()
        path = initial.path
        initial_content = f"- [ ] {_SUM_TASK}\n- [ ] {_CONFIRM_TASK}\n"
        waiting_content = f"- [x] {_SUM_TASK}\n- [ ] {_CONFIRM_TASK}\n"
        finished_content = f"- [x] {_SUM_TASK}\n- [x] {_CONFIRM_TASK}\n"
        first_prompt = f"""请实际执行这组文件工作，并维护当前会话 Todo。严格按下面顺序完成：
1. 用 write_file 在 `{path}` 创建清单，完整内容为：
{initial_content}
2. 用 write_file 创建 numbers.json，内容为 JSON 数组 [7, 11, 13]；再用 read_file 读取核对。
3. 根据读取的三个数字计算总和，用 write_file 创建 result.json，内容为 {{"sum": 31}}；
   再用 read_file 读回 result.json，确认 sum 正确。
4. 用 read_file 读取 `{path}`，然后用 write_file 将完整清单替换为：
{waiting_content}
5. 返回简短普通文本说明“已计算 31，等待用户确认”，结束当前回复。

我在这条消息中没有确认结果。若随后收到 Todo 结束自查，重新读取清单和 result.json，
核对已完成事项；第二项确实仍等待用户，不要清空、移除或勾选它，如实说明等待后结束。
只用上述当前 Todo 路径，numbers.json/result.json 使用相对路径。工具路径参数为 file_path，
write_file.content 传完整文本。需要覆盖现有文件时必须先调用 read_file。
"""
        first = await runner.start(
            AgentRunRequest(input=first_prompt, session_id=_SESSION, run_id=run_ids[0]),
            options=AgentRunOptions(
                limits=RunLimits(
                    max_model_steps=16, deadline_at=datetime.now(UTC) + timedelta(seconds=300)
                )
            ),
        )
        first_calls = _successful_file_calls(first, store)
        assert json.loads((tmp_path / "numbers.json").read_text(encoding="utf-8")) == [7, 11, 13]
        assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {"sum": 31}
        waiting = await runner.get_todo(_SESSION)
        todo_evidence["after_work"] = _snapshot_evidence(waiting)
        assert waiting.path == path and waiting.error is None
        assert [(item.content, item.status) for item in waiting.items] == [
            (_SUM_TASK, TodoStatus.COMPLETED),
            (_CONFIRM_TASK, TodoStatus.PENDING),
        ]
        first_checkpoint = store.load_checkpoint(run_ids[0])
        assert first_checkpoint is not None and first_checkpoint.checkpoint_version == 4
        reminder_step = first_checkpoint.engine_cursor["todo_reminder_step"]
        assert reminder_step is not None
        assert first_checkpoint.engine_cursor["step_index"] >= reminder_step
        assert any(call.step_index >= reminder_step for call in first_calls)
        assert any(
            call.tool_name == "write_file"
            and tmp_path.joinpath(str(call.arguments["file_path"])).resolve() == path
            for call in first_calls
        )
        for name in ("numbers.json", "result.json"):
            assert any(
                call.tool_name == "write_file"
                and tmp_path.joinpath(str(call.arguments["file_path"])).resolve() == tmp_path / name
                for call in first_calls
            )
            assert any(
                call.tool_name == "read_file"
                and tmp_path.joinpath(str(call.arguments["file_path"])).resolve() == tmp_path / name
                for call in first_calls
            )

        confirmation = f"""我已确认 result.json 中的 sum=31 正确，现在授权完成待确认事项。
请先用 read_file 读取 result.json 和当前 Todo 文件 `{path}`，然后用 write_file
将该 Todo 文件完整内容替换为：
{finished_content}
再用 read_file 读回清单，确认两项均完成，最后简短回复。
这是同一会话的新 Run，必须重新调用 read_file，不能沿用上一 Run 的文件读取记录。
"""
        second = await runner.start(
            AgentRunRequest(input=confirmation, session_id=_SESSION, run_id=run_ids[1]),
            options=AgentRunOptions(
                limits=RunLimits(
                    max_model_steps=10, deadline_at=datetime.now(UTC) + timedelta(seconds=300)
                )
            ),
        )
        second_calls = _successful_file_calls(second, store)
        finished = await runner.get_todo(_SESSION)
        todo_evidence["after_confirmation"] = _snapshot_evidence(finished)
        assert finished.path == path and finished.error is None
        assert [(item.content, item.status) for item in finished.items] == [
            (_SUM_TASK, TodoStatus.COMPLETED),
            (_CONFIRM_TASK, TodoStatus.COMPLETED),
        ]
        assert any(
            call.tool_name == "write_file"
            and tmp_path.joinpath(str(call.arguments["file_path"])).resolve() == path
            for call in second_calls
        )
        second_checkpoint = store.load_checkpoint(run_ids[1])
        assert second_checkpoint.engine_cursor["todo_reminder_step"] is None
        assert first.run.session_id == second.run.session_id == _SESSION
        assert all(
            message.metadata.get("context_kind") != "runtime_snapshot"
            for message in runner.get_session(_SESSION).messages
        )
        print(
            f"Todo evidence={tmp_path}, runs=2, reminder_step={reminder_step}, "
            f"tokens={first.run.usage.total_tokens + second.run.usage.total_tokens}"
        )
    finally:
        for run_id in run_ids:
            result = store.load_result(run_id)
            checkpoint = store.load_checkpoint(run_id)
            run_evidence[run_id] = None if result is None else result.model_dump(mode="json")
            checkpoint_evidence[run_id] = (
                None if checkpoint is None else checkpoint.model_dump(mode="json")
            )
            tool_evidence[run_id] = [
                call.model_dump(mode="json") for call in store.list_tool_calls(run_id)
            ]
        evidence["artifacts"] = {
            name: (tmp_path / name).read_text(encoding="utf-8")
            for name in ("numbers.json", "result.json")
            if (tmp_path / name).exists()
        }
        (tmp_path / "todo-evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        await runner.aclose()
