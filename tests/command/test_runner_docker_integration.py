"""真实本地 Docker 与完整 Runner 的组合验收，provider 为确定性脚本。"""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.exceptions import IrisProviderError
from iris.harness import AgentRunner
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunLimits, RunPhase, RunStopReason
from iris.message import LLMRequest, LLMResponse, Role, ToolUseBlock

from ..harness.fakes import StaticProvider, text_response, tool_response

type Step = LLMResponse | Callable[[], Awaitable[LLMResponse]]


@pytest.fixture(autouse=True)
def explicit_docker(request: pytest.FixtureRequest) -> None:
    """不显式启用则不连接引擎。"""
    if not request.config.getoption("--run-docker"):
        pytest.skip("真实 Docker 需显式 --run-docker")


class ScriptedSessions:
    """仅替代远程推理，所有工具与lifecycle走真实公开入口。"""

    def __init__(self, scripts: dict[str, list[Step]]) -> None:
        self.scripts = scripts

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """本组不测token估算。"""
        return 1

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """按本session最后一条测试用户输入推进。"""
        key = next(
            message.text
            for message in reversed(request.messages)
            if message.role is Role.USER and message.text in self.scripts
        )
        step = self.scripts[key].pop(0)
        return await step() if callable(step) else step


def _config(workspace: Path, **extra: object) -> AgentConfig:
    """完整root配置，只有命令工具进入容器。"""
    return AgentConfig.model_validate(
        {
            "name": "docker-integration",
            "model": "openai/scripted",
            "system": "test",
            "context_policy": {"enabled": False},
            "tools": {"builtin": ["exec.command", "human.ask"]},
            "permissions": {"workspace": str(workspace), "writes": "allow", "execute": "allow"},
            "command": {"mode": "docker", "timeout_seconds": 20},
            **extra,
        }
    )


def _command(code: str, *, call_id: str = "command", timeout: float | None = None) -> LLMResponse:
    """以实际Linux shell语法构造容器内Python命令。"""
    values: dict[str, object] = {"command": shlex.join(["python", "-c", code])}
    if timeout is not None:
        values["timeout_seconds"] = timeout
    return tool_response(ToolUseBlock(id=call_id, name="exec_command", input=values))


async def _file_ready(path: Path) -> None:
    """真实进程通过bind mount告知已经进入前台命令。"""
    async with asyncio.timeout(10):
        while not path.exists():
            await asyncio.sleep(0.02)


def _tool_error(runner: AgentRunner, session: str) -> str | None:
    """读取已提交的真实工具结果错误，不从退出码猜测原因。"""
    for message in runner.get_session(session).messages:
        for result in message.tool_results:
            error = result.metadata.get("error")
            if error is not None:
                return error["code"]
    return None


@pytest.mark.asyncio
async def test_run_failure_stops_commands_but_keeps_model_and_hitl(tmp_path: Path) -> None:
    fail_a = asyncio.Event()
    model_started = asyncio.Event()
    model_release = asyncio.Event()

    async def failure() -> LLMResponse:
        await fail_a.wait()
        raise IrisProviderError("controlled A failure")

    async def thinking() -> LLMResponse:
        model_started.set()
        await model_release.wait()
        return text_response("C survived")

    provider = ScriptedSessions(
        {
            "A": [
                _command(
                    "from pathlib import Path; Path('/tmp/retained').write_text('kept'); "
                    "Path('persist.txt').write_text('host')"
                ),
                failure,
            ],
            "B": [
                _command(
                    "import time; from pathlib import Path; Path('b-start').touch(); time.sleep(30)"
                ),
                text_response("B handled interruption"),
            ],
            "C": [thinking],
            "D": [
                tool_response(
                    ToolUseBlock(id="ask", name="ask_question", input={"question": "continue?"})
                ),
                text_response("D resumed"),
            ],
            "E": [
                _command(
                    "from pathlib import Path; print(Path('/tmp/retained').read_text()); "
                    "print(Path('persist.txt').read_text())"
                ),
                text_response("E restarted"),
            ],
        }
    )
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider)
    tasks: list[asyncio.Task] = []
    try:
        a = asyncio.create_task(
            runner.start(AgentRunRequest(input="A", session_id="a", run_id="a"))
        )
        tasks.append(a)
        await _file_ready(tmp_path / "persist.txt")
        b = asyncio.create_task(
            runner.start(AgentRunRequest(input="B", session_id="b", run_id="b"))
        )
        c = asyncio.create_task(
            runner.start(AgentRunRequest(input="C", session_id="c", run_id="c"))
        )
        tasks.extend([b, c])
        await _file_ready(tmp_path / "b-start")
        await model_started.wait()
        d = await runner.start(AgentRunRequest(input="D", session_id="d", run_id="d"))
        assert d.run.phase is RunPhase.WAITING
        fail_a.set()
        assert (await asyncio.wait_for(a, 10)).run.stop_reason is RunStopReason.FAILED
        assert (await asyncio.wait_for(b, 10)).run.stop_reason is RunStopReason.COMPLETED
        assert _tool_error(runner, "b") == "COMMAND_ENVIRONMENT_INTERRUPTED"
        assert not c.done()
        assert runner.get_run("c").phase is RunPhase.ACTIVE
        assert runner.get_run("d").phase is RunPhase.WAITING
        e = await runner.start(AgentRunRequest(input="E", session_id="e", run_id="e"))
        assert e.run.stop_reason is RunStopReason.COMPLETED
        assert _tool_error(runner, "e") is None
        outputs = [
            result for msg in runner.get_session("e").messages for result in msg.tool_results
        ]
        assert "kept" in str(outputs[0].model_dump()) and "host" in str(outputs[0].model_dump())
        model_release.set()
        assert (await c).run.stop_reason is RunStopReason.COMPLETED
        resumed = await runner.resume(
            "d",
            interaction_id=d.pending_interaction.interaction_id,
            response=QuestionInteractionResponse(answer="yes"),
        )
        assert resumed.run.stop_reason is RunStopReason.COMPLETED
    finally:
        fail_a.set()
        model_release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runner.aclose()
    assert (tmp_path / "persist.txt").read_text() == "host"


@pytest.mark.asyncio
async def test_command_timeout_continues_run_and_other_command(tmp_path: Path) -> None:
    provider = ScriptedSessions(
        {
            "short": [
                _command("import time; time.sleep(30)", timeout=0.25),
                text_response("timeout handled"),
            ],
            "other": [
                _command("import time; time.sleep(0.8); print('survived')"),
                text_response("other finished"),
            ],
        }
    )
    runner = AgentRunner.from_config(_config(tmp_path), provider=provider)
    try:
        short, other = await asyncio.gather(
            runner.start(AgentRunRequest(input="short", session_id="short")),
            runner.start(AgentRunRequest(input="other", session_id="other")),
        )
        assert short.run.stop_reason is other.run.stop_reason is RunStopReason.COMPLETED
        assert _tool_error(runner, "short") == "COMMAND_TIMEOUT"
        assert _tool_error(runner, "other") is None
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_independent_roots_do_not_share_files_or_close_each_other(tmp_path: Path) -> None:
    """两个root实例各有容器，一个关闭不影响另一个的后续命令。"""
    a_path, b_path = tmp_path / "a", tmp_path / "b"
    a_path.mkdir()
    b_path.mkdir()
    first = AgentRunner.from_config(
        _config(a_path),
        provider=StaticProvider(
            _command(
                "import socket; from pathlib import Path; "
                "Path('/tmp/owner-only').touch(); Path('hostname').write_text(socket.gethostname())"
            ),
            text_response("a done"),
        ),
    )
    second = AgentRunner.from_config(
        _config(b_path),
        provider=StaticProvider(
            _command(
                "import socket; from pathlib import Path; "
                "assert not Path('/tmp/owner-only').exists(); "
                "Path('hostname').write_text(socket.gethostname())"
            ),
            text_response("b done"),
            _command("print('still running')"),
            text_response("b remains"),
        ),
    )
    try:
        await first.start(AgentRunRequest(input="A"))
        await second.start(AgentRunRequest(input="B"))
        assert _tool_error(first, "default") is _tool_error(second, "default") is None
        assert (a_path / "hostname").read_text() != (b_path / "hostname").read_text()
        await first.aclose()
        assert (
            await second.start(AgentRunRequest(input="again"))
        ).run.stop_reason is RunStopReason.COMPLETED
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
async def test_run_deadline_is_distinct_from_command_timeout(tmp_path: Path) -> None:
    runner = AgentRunner.from_config(
        _config(tmp_path),
        provider=StaticProvider(
            _command(
                "import time; from pathlib import Path; Path('started').touch(); time.sleep(30)"
            )
        ),
    )
    try:
        # 先预热，避免把镜像/容器首次创建时间当作命令已执行的证据。
        await runner.aprepare()
        result = await runner.start(
            AgentRunRequest(input="deadline", run_id="deadline"),
            options=AgentRunOptions(
                limits=RunLimits(deadline_at=datetime.now(UTC) + timedelta(seconds=2))
            ),
        )
        assert (tmp_path / "started").exists()
        assert result.run.stop_reason is RunStopReason.DEADLINE_EXCEEDED
        assert _tool_error(runner, "default") == "COMMAND_CANCELLED"
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_readonly_child_can_write_through_root_docker_mount(tmp_path: Path) -> None:
    (tmp_path / "child").mkdir()
    child_config = {
        "name": "readonly-files-child",
        "model": "openai/scripted",
        "system": "test",
        "tools": {"builtin": ["exec.command"]},
        "permissions": {"workspace": "child", "writes": "deny", "execute": "allow"},
        "context_policy": {"enabled": False},
    }
    (tmp_path / "child.yaml").write_text(json.dumps(child_config), encoding="utf-8")
    (tmp_path / "subagents.yaml").write_text(
        json.dumps(
            {
                "default": "child",
                "agents": {"child": {"path": "child.yaml", "description": "test child"}},
            }
        ),
        encoding="utf-8",
    )
    parent = StaticProvider(
        tool_response(
            ToolUseBlock(id="delegate", name="subagent", input={"prompt": "write proof"})
        ),
        text_response("done"),
    )
    child = StaticProvider(
        _command(
            "from pathlib import Path; "
            "Path('/workspace/outside-child.txt').write_text('root-visible'); print(Path.cwd())"
        ),
        text_response("child wrote"),
    )
    runner = AgentRunner.from_config(
        _config(tmp_path, tools={"builtin": ["exec.command"], "subagent": "subagents.yaml"}),
        config_path=tmp_path / "agent.yaml",
        provider=parent,
        child_provider_factory=lambda config, config_path: child,
    )
    try:
        result = await runner.start(AgentRunRequest(input="delegate"))
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert (tmp_path / "outside-child.txt").read_text() == "root-visible"
    finally:
        await runner.aclose()
