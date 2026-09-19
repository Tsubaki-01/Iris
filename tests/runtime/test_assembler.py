from __future__ import annotations

from iris.agents import AgentConfig
from iris.context import ContextBuilder, ContextBuildInput, ContextSection, ContextSlot
from iris.lifecycle import SessionCompaction
from iris.message import Msg
from iris.runtime import RuntimeMessageAssembler
from iris.runtime.compaction import project_history, protected_message_indices


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


def test_compacted_history_keeps_fixed_sections_and_unarchived_turn_in_order() -> None:
    """模型投影不改写归档输入，step0新输入仍只由assembler追加。"""
    context_output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(slots=[ContextSlot(name="instructions", content="规则")]),
            memory=ContextSection(slots=[ContextSlot(name="memory", content="记忆")]),
            before_current_input=ContextSection(
                slots=[ContextSlot(name="environment", content="当前环境")]
            ),
        )
    )
    raw = [Msg.user("旧任务"), Msg.assistant("旧结果"), Msg.user("最近原文")]
    compaction = SessionCompaction(summary="历史摘要", covered_message_count=2)
    protected = protected_message_indices(raw, initial_session_message_count=len(raw))
    history = project_history(raw, compaction, protected)
    current = Msg.user("尚未归档的新任务")
    assembler = RuntimeMessageAssembler()
    conversation = assembler.build_conversation(
        context_output=context_output,
        history=history,
        current_input=current,
    )

    assert [message.text for message in conversation.messages] == [
        context_output.system.text,
        context_output.memory.text,
        "<summary>\n历史摘要\n</summary>",
        "最近原文",
        context_output.before_current_input.text,
        "尚未归档的新任务",
    ]
    assert conversation.messages[2].sender == "context"
    assert assembler.build_turn_messages(
        before_current_input=context_output.before_current_input, current_input=current
    ) == [
        context_output.before_current_input,
        current,
    ]
    assert [message.text for message in raw] == ["旧任务", "旧结果", "最近原文"]
