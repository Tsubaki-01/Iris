"""通过真实 SDK 验证容器入口，模型只在测试边界替换。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.agentic_rl.evaluation.harbor import results
from experiments.agentic_rl.evaluation.harbor.runner import run_agent
from iris.agents import AgentConfig
from iris.exceptions import IrisProviderError, IrisRunStateError
from iris.harness import AgentRunner
from iris.message import LLMRequest, LLMResponse, TextBlock, ToolUseBlock


class Provider:
    """返回确定性工具动作或响应。"""

    def __init__(self, *responses: LLMResponse | Exception) -> None:
        self.responses = list(responses)
        self.started = asyncio.Event()

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """保持固定估算。"""
        return 1

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """消费响应；没有响应时等候取消。"""
        self.started.set()
        if not self.responses:
            await asyncio.Event().wait()
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def answer(text: str = "完成", *, tool: ToolUseBlock | None = None) -> LLMResponse:
    """构造非流式响应。"""
    return LLMResponse(
        provider="fake",
        model="fake",
        content=[tool] if tool else [TextBlock(text=text)],
        input_tokens=3,
        output_tokens=2,
        total_tokens=5,
        finish_reason="tool_calls" if tool else "stop",
    )


@pytest.fixture
def profile(tmp_path: Path) -> Path:
    """复制正式终端 profile，使模板写入保持在测试目录。"""
    source = Path("experiments/agentic_rl/evaluation/harbor/profiles/terminal/agent.yaml")
    target = tmp_path / "profile" / "agent.yaml"
    target.parent.mkdir()
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def install(monkeypatch: pytest.MonkeyPatch, provider: Provider) -> list[AgentRunner]:
    """注入 provider，继续使用真实配置与 SQLite。"""
    original = AgentRunner.from_config
    runners = []

    def create(cls: type[AgentRunner], config: AgentConfig, **kwargs: Any) -> AgentRunner:
        runner = original(config, provider=provider, **kwargs)
        runners.append(runner)
        return runner

    monkeypatch.setattr(AgentRunner, "from_config", classmethod(create))
    return runners


async def execute(profile: Path, tmp_path: Path, stop: asyncio.Event, *, steps: int = 20) -> int:
    """在与 profile 不同的工作目录运行。"""
    workspace = tmp_path / "task-workdir"
    workspace.mkdir(exist_ok=True)
    return await run_agent(
        profile=profile,
        model="openai/effective-model",
        instruction="完成文件任务",
        workspace=workspace,
        logs_dir=tmp_path / "logs",
        max_model_steps=steps,
        stop=stop,
    )


@pytest.mark.asyncio
async def test_real_file_and_command_share_task_workspace(
    profile: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = Provider(
        answer(
            tool=ToolUseBlock(
                id="write",
                name="write_file",
                input={
                    "file_path": "answer.txt",
                    "content": "正确产物",
                },
            )
        ),
        answer(
            tool=ToolUseBlock(
                id="exec",
                name="exec_command",
                input={
                    "command": "echo command-output",
                    "cwd": ".",
                },
            )
        ),
        answer(),
    )
    runners = install(monkeypatch, provider)
    assert await execute(profile, tmp_path, asyncio.Event()) == 0
    record = json.loads((tmp_path / "logs/result.json").read_text(encoding="utf-8"))
    assert record["cleanup_completed"]
    assert record["stop_reason"] == "completed"
    assert record["trace_status"] == "complete"
    assert record["model"] == "openai/effective-model"
    assert record["usage"]["total_tokens"] == 15
    assert (tmp_path / "task-workdir/answer.txt").read_text(encoding="utf-8") == "正确产物"
    trace = json.loads((tmp_path / "logs/trace.json").read_text(encoding="utf-8"))
    calls = trace["runs"][record["run_id"]]["tool_calls"]
    assert len(calls) == 2
    assert not any(call["result"]["is_error"] for call in calls)
    assert "command-output" in str(calls[-1]["result"])
    assert (tmp_path / "logs/lifecycle.db").is_file()
    with pytest.raises(IrisRunStateError, match="关闭"):
        await runners[0].aprepare()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,expected,reason",
    [
        ("failed", 1, "failed"),
        ("budget", 0, "budget_exhausted"),
        ("waiting", 1, None),
    ],
)
async def test_failure_budget_and_waiting(
    profile: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected: int,
    reason: str | None,
) -> None:
    if mode == "failed":
        provider = Provider(IrisProviderError("模型失败"))
    elif mode == "waiting":
        profile.write_text(
            profile.read_text(encoding="utf-8").replace("file.read,", "human.ask, file.read,"),
            encoding="utf-8",
        )
        provider = Provider(
            answer(tool=ToolUseBlock(id="ask", name="ask_question", input={"question": "等待回答"}))
        )
    else:
        provider = Provider(answer(tool=ToolUseBlock(id="missing", name="missing", input={})))
    install(monkeypatch, provider)
    assert await execute(profile, tmp_path, asyncio.Event(), steps=1) == expected
    record = json.loads((tmp_path / "logs/result.json").read_text(encoding="utf-8"))
    assert record["cleanup_completed"]
    assert record["stop_reason"] == reason
    if mode == "waiting":
        assert record["phase"] == "waiting"
        trace = json.loads((tmp_path / "logs/trace.json").read_text(encoding="utf-8"))
        assert trace["runs"][record["run_id"]]["run"]["stop_reason"] == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("before_admission", [False, True])
async def test_stop_during_prepare_or_model(
    profile: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    before_admission: bool,
) -> None:
    provider = Provider()
    install(monkeypatch, provider)
    preparing = asyncio.Event()

    async def blocked_prepare(self: AgentRunner) -> None:
        preparing.set()
        await asyncio.Event().wait()

    if before_admission:
        monkeypatch.setattr(AgentRunner, "aprepare", blocked_prepare)
    stop = asyncio.Event()
    task = asyncio.create_task(execute(profile, tmp_path, stop))
    await asyncio.wait_for((preparing if before_admission else provider.started).wait(), 3)
    stop.set()
    assert await asyncio.wait_for(task, 3) == 1
    record = json.loads((tmp_path / "logs/result.json").read_text(encoding="utf-8"))
    assert record["cleanup_completed"]
    assert record["stop_reason"] == (None if before_admission else "cancelled")


@pytest.mark.asyncio
async def test_cleanup_failure_has_own_exit_code(
    profile: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, Provider(answer()))
    close = AgentRunner.aclose

    async def broken_close(self: AgentRunner) -> None:
        await close(self)
        raise IrisRunStateError("收尾失败")

    monkeypatch.setattr(AgentRunner, "aclose", broken_close)
    assert await execute(profile, tmp_path, asyncio.Event()) == 70
    record = json.loads((tmp_path / "logs/result.json").read_text(encoding="utf-8"))
    assert not record["cleanup_completed"]
    assert record["stop_reason"] == "completed"
    assert record["trace_status"] == "partial"


@pytest.mark.asyncio
async def test_start_exception_is_not_cleanup_failure(
    profile: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, Provider())

    async def broken_start(self: AgentRunner, *args: Any, **kwargs: Any) -> None:
        raise IrisRunStateError("准入调用失败")

    monkeypatch.setattr(AgentRunner, "start", broken_start)
    assert await execute(profile, tmp_path, asyncio.Event()) == 1
    record = json.loads((tmp_path / "logs/result.json").read_text(encoding="utf-8"))
    assert record["cleanup_completed"]
    assert not record["admitted"]
    assert "准入调用失败" in record["execution_error"]


@pytest.mark.asyncio
async def test_missing_export_asset_keeps_success_exit(
    profile: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, Provider(answer()))
    collect = results.collect_trajectory

    def with_missing_asset(runner: AgentRunner, session_id: str) -> dict[str, Any]:
        document = collect(runner, session_id)
        document["assets"].append(
            {
                "source_path": str(tmp_path / "deleted.txt"),
                "path": None,
                "error": None,
            }
        )
        return document

    monkeypatch.setattr(results, "collect_trajectory", with_missing_asset)
    assert await execute(profile, tmp_path, asyncio.Event()) == 0
    record = json.loads((tmp_path / "logs/result.json").read_text(encoding="utf-8"))
    assert record["cleanup_completed"]
    assert record["stop_reason"] == "completed"
    assert record["trace_status"] == "partial"


def test_token_usage_counts_compaction_once() -> None:
    assert results.token_usage(
        {
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "compaction": {"input_tokens": 5, "output_tokens": 3},
            }
        }
    ) == (16, 10)
    assert results.token_usage({"usage": None}) == (None, None)
