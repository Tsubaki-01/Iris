"""AgentRunner 窄控制读取的委托契约。"""

from pathlib import Path

import pytest

from iris.exceptions import IrisRunNotFoundError
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest
from iris.store import InMemoryLifecycleStore

from .fakes import build_runtime


@pytest.mark.asyncio
async def test_get_run_control_uses_only_narrow_store_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """已有 run 返回控制快照，缺失 run 报错，两者都不读取完整 aggregate。"""
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    await runner.start(AgentRunRequest(input="你好", run_id="run-control"))
    expected = store.load_run_control("run-control")
    assert expected.session_id == "default"

    def reject_full_read(run_id: str) -> None:
        pytest.fail("get_run_control must not load the complete run")

    monkeypatch.setattr(store, "load_run", reject_full_read)
    assert runner.get_run_control("run-control") == expected
    with pytest.raises(IrisRunNotFoundError):
        runner.get_run_control("missing")
