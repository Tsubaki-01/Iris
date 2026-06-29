from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from iris.agents import AgentConfig, load_agent_config
from iris.context import (
    ContextBuilder,
    ContextBuildInput,
    ContextBuildOutput,
    ContextSection,
    ContextSlot,
    load_context_build_input,
)
from iris.message import Conversation, LLMRequest, Msg, Role
from iris.runtime import RuntimeMessageAssembler


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="runtime-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
    )


def _system_context(content: str = "遵守用户指令") -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(slots=[ContextSlot(name="instructions", content=content)])
    )


def _build_context_output(
    input_data: ContextBuildInput | None = None,
) -> ContextBuildOutput:
    return ContextBuilder().build(input_data or _system_context())


def test_simple_system_context_assembles_system_history_and_current_input() -> None:
    history = [Msg.user("历史问题"), Msg.assistant("历史回答")]
    original_history = list(history)
    current_input = Msg.user("当前问题")
    context_output = _build_context_output()

    request = RuntimeMessageAssembler().build_request(
        agent_config=_agent_config(),
        context_output=context_output,
        history=history,
        current_input=current_input,
    )

    assert [message.role for message in request.messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]
    assert "遵守用户指令" in request.messages[0].text
    assert request.messages[1:] == [*history, current_input]
    assert history == original_history
    assert request.messages is not history


def test_build_conversation_returns_ordered_runtime_conversation() -> None:
    history = [Msg.user("历史问题"), Msg.assistant("历史回答")]
    current_input = Msg.user("当前问题")
    context_output = _build_context_output()

    conversation = RuntimeMessageAssembler().build_conversation(
        context_output=context_output,
        history=history,
        current_input=current_input,
    )

    assert isinstance(conversation, Conversation)
    assert conversation.messages == [
        context_output.system,
        *history,
        current_input,
    ]
    assert conversation.messages is not history


def test_structured_context_keeps_memory_history_before_current_input_order() -> None:
    history = [Msg.user("历史输入")]
    current_input = Msg.user("当前输入")
    context_input = ContextBuildInput(
        system=ContextSection(
            slots=[ContextSlot(name="instructions", content="系统规则")]
        ),
        memory=ContextSection(
            slots=[ContextSlot(name="memory", content="用户偏好简洁回答")]
        ),
        before_current_input=ContextSection(
            slots=[ContextSlot(name="environment_state", content={"cwd": "J:/repo"})]
        ),
    )
    context_output = _build_context_output(context_input)

    request = RuntimeMessageAssembler().build_request(
        agent_config=_agent_config(),
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
    assert context_output.memory is not None
    assert context_output.memory.text.startswith("<memory_context>")
    assert context_output.before_current_input is not None
    assert context_output.before_current_input.text.startswith(
        "<before_current_input_context>"
    )


def test_current_input_none_is_not_appended() -> None:
    history = [Msg.user("历史问题")]
    context_output = _build_context_output()

    request = RuntimeMessageAssembler().build_request(
        agent_config=_agent_config(),
        context_output=context_output,
        history=history,
        current_input=None,
    )

    assert request.messages == [context_output.system, *history]


def test_request_uses_provider_neutral_llm_request() -> None:
    current_input = Msg.user("当前问题")
    context_output = _build_context_output()

    request = RuntimeMessageAssembler().build_request(
        agent_config=_agent_config(),
        context_output=context_output,
        history=[],
        current_input=current_input,
    )

    assert isinstance(request, LLMRequest)
    assert request.model == "gpt-4o-mini"
    assert request.messages == [context_output.system, current_input]
    assert request.tools == []
    assert request.provider_options == {}


def test_request_inherits_llm_request_options_from_model_config() -> None:
    agent_config = AgentConfig(
        name="runtime-agent",
        model={
            "provider": "openai",
            "name": "gpt-4o-mini",
            "api_style": "responses",
            "temperature": 0.2,
            "max_tokens": 512,
            "timeout": 30,
            "provider_options": {"reasoning_effort": "low"},
        },
        system="你是本地助手。",
    )

    request = RuntimeMessageAssembler().build_request(
        agent_config=agent_config,
        context_output=_build_context_output(),
        history=[],
        current_input=Msg.user("当前问题"),
    )

    assert request.temperature == 0.2
    assert request.max_tokens == 512
    assert request.timeout == 30
    assert request.provider_options == {
        "api_style": "responses",
        "reasoning_effort": "low",
    }


def test_loader_resolves_agent_and_context_relative_paths() -> None:
    workspace = Path("tmp") / f"runtime-assembler-{uuid.uuid4().hex}"
    workspace.mkdir(parents=True)
    try:
        template_dir = workspace / "templates"
        template_dir.mkdir()
        template_path = template_dir / "before.j2"
        template_path.write_text(
            (
                "{% for slot in slots %}"
                "<{{ slot.name }}>{{ slot.content }}</{{ slot.name }}>"
                "{% endfor %}"
            ),
            encoding="utf-8",
        )
        context_path = workspace / "context.yaml"
        context_path.write_text(
            """
system:
  slots:
    - name: instructions
      content: 来自 context.yaml
before_current_input:
  template: templates/before.j2
  slots:
    - name: runtime_state
      content: 来自模板
""".strip(),
            encoding="utf-8",
        )
        agent_path = workspace / "agent.yaml"
        agent_path.write_text(
            """
name: runtime-agent
model: openai/gpt-4o-mini
context:
  path: context.yaml
""".strip(),
            encoding="utf-8",
        )

        agent_config = load_agent_config(agent_path)
        context_input = load_context_build_input(agent_config.context.path)

        assert agent_config.context is not None
        assert agent_config.context.path == context_path.resolve()
        assert context_input.before_current_input is not None
        assert context_input.before_current_input.template == template_path.resolve()

        context_output = ContextBuilder().build(context_input)
        request = RuntimeMessageAssembler().build_request(
            agent_config=agent_config,
            context_output=context_output,
            history=[],
            current_input=Msg.user("当前问题"),
        )

        assert "来自 context.yaml" in request.system_prompt()
        assert context_output.before_current_input is not None
        assert "<runtime_state>来自模板</runtime_state>" in (
            context_output.before_current_input.text
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
