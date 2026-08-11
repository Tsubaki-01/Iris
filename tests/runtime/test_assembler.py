from __future__ import annotations

from iris.agents import AgentConfig
from iris.context import ContextBuilder, ContextBuildInput, ContextSection, ContextSlot
from iris.message import Msg
from iris.runtime import RuntimeMessageAssembler


def test_structured_context_keeps_memory_history_before_current_input_order() -> None:
    context_output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="instructions", content="系统规则")]),
            memory=ContextSection(slots=[ContextSlot(name="memory", content="用户偏好简洁回答")]),
            before_current_input=ContextSection(
                slots=[ContextSlot(name="environment_state", content={"cwd": "J:/repo"})]
            ),
        )
    )
    history = [Msg.user("历史输入")]
    current_input = Msg.user("当前输入")

    request = RuntimeMessageAssembler().build_request(
        agent_config=AgentConfig(
            name="runtime-agent",
            model={"provider": "openai", "name": "gpt-4o-mini"},
            system="你是本地助手。",
        ),
        context_output=context_output,
        history=history,
        current_input=current_input,
    )

    assert request.messages == [
        context_output.system,
        context_output.memory,
        *history,
        context_output.before_current_input,
        current_input,
    ]
