"""主请求使用有效历史快照，公开完整历史查询仍保持独立。"""

from pathlib import Path

import pytest

from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest, RunStopReason, SessionSnapshot
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .fakes import StaticProvider, build_runtime, text_response


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
async def test_input_and_model_steps_do_not_load_complete_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persistent: bool
) -> None:
    """首次空窗口与后续已初始化窗口均不能回退完整历史读取。"""
    store = SQLiteStore(tmp_path / "context.db") if persistent else InMemoryLifecycleStore()
    provider = StaticProvider(text_response("第一轮完成"), text_response("第二轮完成"))
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)

    def reject_complete_read(session_id: str) -> SessionSnapshot:
        raise AssertionError(f"主请求读取了完整历史：{session_id}")

    monkeypatch.setattr(store, "load_session", reject_complete_read)
    try:
        for number in (1, 2):
            result = await runner.start(
                AgentRunRequest(input=f"任务{number}", session_id="shared", run_id=f"run-{number}")
            )
            assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
        assert [message.text for message in provider.requests[1].messages][-3:] == [
            "任务1",
            "第一轮完成",
            "任务2",
        ]
    finally:
        await runner.aclose()
