from pathlib import Path

import pytest

from examples.lifecycle import tools
from iris.agents import build_tool_registry, load_agent_config
from iris.context import ContextBuilder, load_context_build_input
from iris.exceptions import IrisToolValidationError
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.runtime import RuntimeFactory
from iris.tools import ToolExecutionContext

ROOT = Path(__file__).resolve().parents[2]


class _NoNetworkProvider:
    """确保示例集成测试不会进入 provider 调用。"""

    def __init__(self) -> None:
        self.called = False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """若测试意外触发 provider，则立即失败。"""
        self.called = True
        raise AssertionError(f"离线示例测试不得调用 provider: {request.model}")


def test_chat_example_config_resolves_context_and_workspace() -> None:
    config = load_agent_config(ROOT / "examples/chat/agent.yaml")
    assert config.context is not None
    context = load_context_build_input(config.context.path)
    rendered = ContextBuilder().build(context)
    assert "Iris Example Agent" in rendered.system.text
    assert config.permissions.workspace == "workspace"
    assert config.session.backend == "sqlite"


@pytest.mark.asyncio
async def test_chat_example_catalogs_and_loads_required_skill_without_network() -> None:
    provider = _NoNetworkProvider()
    runtime = RuntimeFactory.from_config_path(
        ROOT / "examples/chat/agent.yaml",
        provider=provider,
    )
    environment = runtime.environment
    rendered = environment.context_builder.build(environment.context_input)

    assert "review-python" in rendered.system.text
    assert "检查 Python 代码并报告可复现问题" in rendered.system.text
    assert environment.tool_bridge.tool_view.get("load_skill") is not None

    result = await environment.tool_bridge.tool_executor.execute_one(
        ToolUseBlock(
            id="load-example-skill",
            name="load_skill",
            input={"name": "review-python"},
        ),
        ToolExecutionContext(workspace_root=environment.workspace_root),
    )

    assert result.is_error is False
    assert "# Python 代码检查" in result.model_content
    assert "不要修改文件" in result.model_content
    assert provider.called is False


def test_lifecycle_example_registers_wait_tool() -> None:
    config = load_agent_config(ROOT / "examples/lifecycle/agent.yaml")
    registry = build_tool_registry(config.tools)
    assert registry.get("wait_for_seconds").definition.name == "wait_for_seconds"
    assert config.session.path == ".iris/lifecycle.db"


def test_lifecycle_wait_tool_keeps_sync_smoke_semantics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    slept: list[int] = []
    monkeypatch.setattr(tools.time, "sleep", slept.append)

    assert tools.wait_for_seconds(2) == "waited 2 seconds"
    captured = capsys.readouterr()

    assert slept == [2]
    assert captured.out == ""
    assert captured.err == "IRIS_EXAMPLE_TOOL_STARTED seconds=2\n"


@pytest.mark.parametrize("seconds", [True, False, 0, 61, 1.5])
def test_lifecycle_wait_tool_rejects_invalid_seconds(seconds: int | float | bool) -> None:
    with pytest.raises(IrisToolValidationError):
        tools.wait_for_seconds(seconds)
