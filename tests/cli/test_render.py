from __future__ import annotations

import pytest
from rich.console import Console

from iris.cli.render import ChatRenderer
from iris.exceptions import HITLCheckpointInvalidError
from iris.hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    PermissionPrompt,
    QuestionPrompt,
    ToolCallSnapshot,
)


def _renderer() -> tuple[ChatRenderer, Console]:
    console = Console(record=True, width=160, color_system=None)
    return ChatRenderer(console), console


def _permission_interaction() -> HumanInteraction:
    request = HumanInteractionRequest(
        tool_call=ToolCallSnapshot(
            tool_call_id="call_write",
            tool_name="write_file",
            arguments={"路径": "资料/计划.md", "内容": "你好"},
            workspace_root=r"J:\Tsubaki-01\Iris",
            fingerprint="a" * 64,
        ),
        prompt=PermissionPrompt(reason="工具需要写入工作区"),
    )
    return HumanInteraction(
        interaction_id="int_11111111111111111111111111111111",
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        tool_call_id="call_write",
        request=request,
    )


def _question_interaction() -> HumanInteraction:
    request = HumanInteractionRequest(
        tool_call=ToolCallSnapshot(
            tool_call_id="call_question",
            tool_name="ask_question",
            arguments={"question": "请选择部署环境", "options": ["测试", "生产"]},
            workspace_root=r"J:\Tsubaki-01\Iris",
            fingerprint="b" * 64,
        ),
        prompt=QuestionPrompt(question="请选择部署环境", options=["测试", "生产"]),
    )
    return HumanInteraction(
        interaction_id="int_22222222222222222222222222222222",
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        tool_call_id="call_question",
        request=request,
    )


def test_render_permission_interaction_contains_exact_call_details() -> None:
    renderer, console = _renderer()
    interaction = _permission_interaction()

    renderer.render_permission_interaction(interaction)

    output = console.export_text()
    assert interaction.interaction_id in output
    assert "write_file" in output
    assert '"路径": "资料/计划.md"' in output
    assert '"内容": "你好"' in output
    assert "工具需要写入工作区" in output
    assert r"J:\Tsubaki-01\Iris" in output
    assert "本次批准只适用于该调用" in output
    assert "\\u8def" not in output


def test_render_question_interaction_contains_options_and_free_text_notice() -> None:
    renderer, console = _renderer()
    interaction = _question_interaction()

    renderer.render_question_interaction(interaction)

    output = console.export_text()
    assert interaction.interaction_id in output
    assert "请选择部署环境" in output
    assert "1. 测试" in output
    assert "2. 生产" in output
    assert "也可输入自由文本" in output


def test_render_recovery_notice_identifies_interaction() -> None:
    renderer, console = _renderer()
    interaction = _question_interaction()

    renderer.render_recovery_notice(interaction)

    output = console.export_text()
    assert "恢复" in output
    assert interaction.interaction_id in output
    assert interaction.status.value in output


def test_renderer_rejects_mismatched_interaction_kind_and_request() -> None:
    renderer, _ = _renderer()

    with pytest.raises(HITLCheckpointInvalidError, match="permission"):
        renderer.render_permission_interaction(_question_interaction())
    with pytest.raises(HITLCheckpointInvalidError, match="question"):
        renderer.render_question_interaction(_permission_interaction())
