"""执行环境沿已有 system addendum 路径进入请求，保留记忆与预算语义。"""

from pathlib import Path
from typing import cast

import pytest
from fakes import FakeProvider

from iris.exceptions import IrisContextError
from iris.execution import CommandEnvironment, ExecutionMode
from iris.lifecycle import RuntimeExecutionOptions, SessionContextWindow
from iris.memory import MemoryService
from iris.message import Msg, Role
from iris.runtime._tool_context import ToolContextSelection

from .test_execute import _runtime


@pytest.mark.parametrize("memory_enabled", [False, True])
def test_environment_and_adopted_memory_are_composed_once(
    tmp_path: Path, memory_enabled: bool
) -> None:
    runtime = _runtime(provider=FakeProvider([]), tmp_path=tmp_path)
    runtime.environment.host_os = "Windows"
    runtime.environment.command_environment = CommandEnvironment(
        "Windows", ExecutionMode.DOCKER, "Linux", "/bin/sh"
    )
    # 本例只测试已采用概览的投影；不调用服务或重新读取记忆。
    runtime.environment.memory_service = cast(MemoryService, object()) if memory_enabled else None
    kwargs = {
        "history": [Msg.user("question")],
        "options": RuntimeExecutionOptions(),
        "context_window": SessionContextWindow(memory_overview="adopted memory"),
        "tool_selection": ToolContextSelection((), (), None),
    }
    first, _ = runtime._build_model_request(**kwargs)
    second, _ = runtime._build_model_request(**kwargs)
    system = next(message for message in first.messages if message.role is Role.SYSTEM)
    assert system.text.count("<runtime_environment>") == 1
    assert "host_os: Windows" in system.text
    assert "command_os: Linux" in system.text
    assert "command_shell: /bin/sh" in system.text
    assert ("adopted memory" in system.text) is memory_enabled
    if memory_enabled:
        assert system.text.index("</runtime_environment>") < system.text.index("adopted memory")
    assert len([message for message in first.messages if message.role is Role.SYSTEM]) == 1
    assert first.messages == second.messages
    assert all("runtime_environment" not in message.text for message in first.messages[1:])


def test_agent_without_commands_only_describes_host_and_total_system_limit_applies(
    tmp_path: Path,
) -> None:
    runtime = _runtime(provider=FakeProvider([]), tmp_path=tmp_path)
    runtime.environment.host_os = "Linux"
    request, _ = runtime._build_model_request(
        history=[],
        options=RuntimeExecutionOptions(),
        context_window=SessionContextWindow(),
        tool_selection=ToolContextSelection((), (), None),
    )
    assert "host_os: Linux" in request.messages[0].text
    assert "execution_mode:" not in request.messages[0].text
    assert "command_shell:" not in request.messages[0].text
    section = runtime.environment.context_input.system.model_copy(
        update={"max_chars": len(request.messages[0].text) - 1}
    )
    runtime.environment.context_input = runtime.environment.context_input.model_copy(
        update={"system": section}
    )
    with pytest.raises(IrisContextError):
        runtime._build_model_request(
            history=[],
            options=RuntimeExecutionOptions(),
            context_window=SessionContextWindow(),
            tool_selection=ToolContextSelection((), (), None),
        )
