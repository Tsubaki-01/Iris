"""显式开关下真实 Docker、YAML Hook 工厂和完整 Runner 的验收。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from iris.command.docker import DockerCommandService
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.message import TextBlock, ToolUseBlock

from ..harness.fakes import StaticProvider, text_response, tool_response
from ..hooks.config_extensions import command_hook, script_command, write_config, write_hook_script


@pytest.fixture(autouse=True)
def explicit_docker(request: pytest.FixtureRequest) -> None:
    """真实引擎失败会使验收失败，只有未启用开关才跳过。"""
    if not request.config.getoption("--run-docker"):
        pytest.skip("真实 Docker 需显式 --run-docker")


@pytest.mark.asyncio
async def test_yaml_docker_four_events_and_feedback_reach_next_model(tmp_path: Path) -> None:
    """配置构造实际容器服务，脚本与工具均进入容器，模型只接收正文及反馈。"""
    (tmp_path / "body.py").write_text("print('容器正文')\n", encoding="utf-8")
    command = script_command(write_hook_script(tmp_path), docker=True)
    path = write_config(
        tmp_path,
        [
            command_hook(event, command)
            for event in ("run.started", "tool.before", "tool.after", "run.finished")
        ],
        tools={"builtin": ["exec.command"]},
        command={"mode": "docker", "timeout_seconds": 20},
    )
    provider = StaticProvider(
        tool_response(
            ToolUseBlock(id="command", name="exec_command", input={"command": "python body.py"})
        ),
        text_response("done"),
    )
    runner = AgentRunner.from_config_path(path, provider=provider)
    try:
        binding = runner.runtime.environment.command_binding
        assert binding is not None and isinstance(binding.service, DockerCommandService)
        source = tmp_path / "input.png"
        with Image.new("RGB", (40, 20), "red") as picture:
            picture.save(source)
        image = await runner.import_image(source, session_id="s")
        content = [TextBlock(text="运行中文命令"), image]
        result = await runner.start(AgentRunRequest(input=content, run_id="run", session_id="s"))
        assert result.run.stop_reason is RunStopReason.COMPLETED, result.error
        events = [
            json.loads(line)
            for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [event["event"] for event in events] == [
            "run.started",
            "tool.before",
            "tool.after",
            "run.finished",
        ]
        assert events[0]["input"] == [block.model_dump(mode="json") for block in content]
        assert provider.requests[0].messages[1].content == content
        assert events[1]["arguments"]["command"] == "python body.py"
        assert events[2]["body_status"] == "success"
        assert events[3]["result"]["run"]["stop_reason"] == "completed"
        saved = runner.store.load_tool_call("run", "command")
        assert saved is not None and saved.result is not None
        assert saved.result.hook_feedback == ("脚本反馈",)
        delivered = [
            item for message in provider.requests[1].messages for item in message.tool_results
        ]
        assert len(delivered) == 1 and delivered[0].text == saved.result.model_content
        assert "容器正文" in delivered[0].text and "脚本反馈" in delivered[0].text
        assert "diagnostic-only" not in delivered[0].text
        assert not binding.service._calls
        assert not runner._command_lifecycle.pending
    finally:
        await runner.aclose()
