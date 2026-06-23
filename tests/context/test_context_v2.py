from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from iris.context import (
    ContextBuilder,
    ContextBuildInput,
    ContextPosition,
    ContextSlot,
    ContextTemplateSpec,
    MemoryContextInput,
    MemoryContextItem,
    SystemPromptSpec,
    load_context_build_input,
    load_context_config,
)
from iris.exceptions import IrisContextError
from iris.message import Role


def test_inline_system_memory_and_runtime_context_messages_are_generated() -> None:
    output = ContextBuilder().build(
        ContextBuildInput(
            agent_id="agent",
            system=SystemPromptSpec(inline="你是助手"),
            memory=MemoryContextInput(
                entries=[
                    MemoryContextItem(
                        id="mem_1",
                        source="sqlite",
                        text="用户喜欢 <brief> & direct",
                    )
                ],
                warnings=["可能过期"],
            ),
            environment_state={"cwd": "J:/repo", "sandbox": "workspace-write"},
            turn_constraints=["不要猜测文件内容"],
            current_input="当前用户输入不应被复制到 context 输出",
        )
    )

    assert output.system_message.role == Role.SYSTEM
    assert output.system_message.text.startswith('<system_context version="1">')
    assert (
        "<base_instructions>你是助手</base_instructions>" in output.system_message.text
    )

    assert len(output.memory_messages) == 1
    memory_message = output.memory_messages[0]
    assert memory_message.role == Role.USER
    assert memory_message.sender == "context"
    assert '<memory_context version="1">' in memory_message.text
    assert 'id="mem_1"' in memory_message.text
    assert "用户喜欢 &lt;brief&gt; &amp; direct" in memory_message.text
    assert "可能过期" in memory_message.text

    assert len(output.before_current_input_messages) == 1
    runtime_message = output.before_current_input_messages[0]
    assert runtime_message.role == Role.USER
    assert runtime_message.sender == "context"
    assert '<runtime_context version="1">' in runtime_message.text
    assert "environment_state" in runtime_message.text
    assert "turn_constraints" in runtime_message.text
    assert "不要猜测文件内容" in runtime_message.text

    rendered = "\n".join(
        [
            output.system_message.text,
            *(message.text for message in output.memory_messages),
            *(message.text for message in output.before_current_input_messages),
        ]
    )
    assert "当前用户输入不应被复制到 context 输出" not in rendered


def test_slots_are_filtered_and_sorted() -> None:
    output = ContextBuilder().build(
        ContextBuildInput(
            agent_id="agent",
            system=SystemPromptSpec(inline="base"),
            slots=[
                ContextSlot(
                    name="late",
                    position=ContextPosition.SYSTEM,
                    content="late",
                    order=30,
                ),
                ContextSlot(
                    name="early",
                    position=ContextPosition.SYSTEM,
                    content="early",
                    order=10,
                ),
                ContextSlot(
                    name="disabled",
                    position=ContextPosition.SYSTEM,
                    content="disabled",
                    enabled=False,
                    order=1,
                ),
            ],
        )
    )

    text = output.system_message.text
    assert "disabled" not in text
    assert text.index("base") < text.index("early") < text.index("late")
    assert [slot.name for slot in output.slots] == [
        "base_instructions",
        "early",
        "late",
    ]


def test_template_system_prompt_uses_xml_jinja_file(tmp_path: Path) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_text(
        '<system_context version="{{ version }}">'
        "<identity>{{ system.identity }}</identity>"
        "<slot_count>{{ slots|length }}</slot_count>"
        "</system_context>",
        encoding="utf-8",
    )

    output = ContextBuilder().build(
        ContextBuildInput(
            agent_id="agent",
            template_base_dir=tmp_path,
            system=SystemPromptSpec(
                mode="template",
                template_path="system.xml.j2",
                variables={"identity": "本地助手 & 文件助手"},
            ),
        )
    )

    assert output.system_message.text == (
        '<system_context version="1"><identity>本地助手 &amp; 文件助手</identity>'
        "<slot_count>0</slot_count></system_context>"
    )


def test_template_errors_use_context_exception(tmp_path: Path) -> None:
    output_path = tmp_path / "missing.xml.j2"

    with pytest.raises(IrisContextError):
        ContextBuilder().build(
            ContextBuildInput(
                agent_id="agent",
                template_base_dir=tmp_path,
                system=SystemPromptSpec(
                    mode="template",
                    template_path=output_path.name,
                ),
            )
        )


def test_template_runtime_errors_use_context_exception(tmp_path: Path) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_text("{{ missing_value }}", encoding="utf-8")

    with pytest.raises(IrisContextError):
        ContextBuilder().build(
            ContextBuildInput(
                agent_id="agent",
                template_base_dir=tmp_path,
                system=SystemPromptSpec(
                    mode="template",
                    template_path=template.name,
                ),
            )
        )


def test_memory_template_uses_documented_entries(
    tmp_path: Path,
) -> None:
    template = tmp_path / "memory.xml.j2"
    template.write_text(
        """
<memory_context version="{{ version }}">
{% for item in memory.entries %}
  <memory id="{{ item.id }}" source="{{ item.source }}">{{ item.text }}</memory>
{% endfor %}
</memory_context>
""".strip(),
        encoding="utf-8",
    )

    output = ContextBuilder().build(
        ContextBuildInput(
            agent_id="agent",
            template_base_dir=tmp_path,
            system=SystemPromptSpec(inline="你是助手"),
            templates=ContextTemplateSpec(memory=template.name),
            memory=MemoryContextInput(
                entries=[
                    MemoryContextItem(
                        id="mem_1",
                        source="sqlite",
                        text="abcdefghij",
                    )
                ],
            ),
        )
    )

    assert output.memory_messages[0].text == (
        '<memory_context version="1">\n'
        '  <memory id="mem_1" source="sqlite">abcdefghij</memory>\n'
        "</memory_context>"
    )


def test_template_paths_reject_absolute_and_parent_segments(tmp_path: Path) -> None:
    absolute_template = tmp_path / "system.xml.j2"
    absolute_template.write_text("<system_context />", encoding="utf-8")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    with pytest.raises(IrisContextError):
        ContextBuilder().build(
            ContextBuildInput(
                agent_id="agent",
                template_base_dir=tmp_path,
                system=SystemPromptSpec(
                    mode="template",
                    template_path=absolute_template,
                ),
            )
        )

    with pytest.raises(IrisContextError):
        ContextBuilder().build(
            ContextBuildInput(
                agent_id="agent",
                template_base_dir=config_dir,
                system=SystemPromptSpec(
                    mode="template",
                    template_path="../system.xml.j2",
                ),
            )
        )


def test_relative_template_path_without_base_does_not_use_cwd() -> None:
    with pytest.raises(IrisContextError):
        ContextBuilder().build(
            ContextBuildInput(
                agent_id="agent",
                system=SystemPromptSpec(inline="你是助手"),
                templates=ContextTemplateSpec(system="pyproject.toml"),
            )
        )


def test_output_slots_follow_message_position_order() -> None:
    output = ContextBuilder().build(
        ContextBuildInput(
            agent_id="agent",
            system=SystemPromptSpec(inline="system"),
            memory=MemoryContextInput(entries=[MemoryContextItem(text="memory")]),
            environment_state={"cwd": "repo"},
        )
    )

    assert [slot.position for slot in output.slots] == [
        ContextPosition.SYSTEM,
        ContextPosition.MEMORY,
        ContextPosition.BEFORE_CURRENT_INPUT,
    ]


def test_disabled_memory_does_not_surface_messages_or_warnings() -> None:
    output = ContextBuilder().build(
        ContextBuildInput(
            agent_id="agent",
            system=SystemPromptSpec(inline="system"),
            memory=MemoryContextInput(
                enabled=False,
                entries=[MemoryContextItem(text="memory")],
                warnings=["不应输出"],
            ),
        )
    )

    assert output.memory_messages == []
    assert output.warnings == []


def test_context_slot_rejects_legacy_budget_chars() -> None:
    with pytest.raises(ValidationError):
        ContextSlot(
            name="legacy",
            position=ContextPosition.SYSTEM,
            content="legacy",
            budget_chars=5,
        )


def test_memory_context_input_rejects_legacy_max_chars() -> None:
    with pytest.raises(ValidationError):
        MemoryContextInput(
            max_chars=5,
            entries=[MemoryContextItem(text="abcdefghij")],
        )


def test_load_context_config_converts_yaml_to_build_input(tmp_path: Path) -> None:
    config_path = tmp_path / "context.yaml"
    config_path.write_text(
        """
version: 1
templates:
  system: templates/system.xml.j2
  memory: templates/memory.xml.j2
  before_current_input: templates/runtime.xml.j2
system:
  identity: 你是本地助手。
  behavior_rules:
    - 不要猜测文件内容。
  response_style:
    - 简洁回答。
memory:
  enabled: true
  warnings:
    - 记忆可能过期。
  entries:
    - id: mem_yaml
      source: yaml
      text: YAML 中的记忆
before_current_input:
  environment_state:
    cwd: repo
  turn_constraints:
    - 遵循当前请求。
metadata:
  source: yaml
""".strip(),
        encoding="utf-8",
    )

    config = load_context_config(config_path)
    input_data = config.to_build_input(
        agent_id="agent",
        environment_state={"timezone": "Asia/Shanghai"},
        turn_constraints=["运行态约束。"],
        current_input="不要复制我",
        metadata={"trace_id": "trace-1"},
    )

    assert input_data.system.mode == "template"
    assert (
        str(input_data.system.template_path).replace("\\", "/")
        == "templates/system.xml.j2"
    )
    assert input_data.system.variables["identity"] == "你是本地助手。"
    assert (
        str(input_data.templates.memory).replace("\\", "/") == "templates/memory.xml.j2"
    )
    assert (
        str(input_data.templates.before_current_input).replace("\\", "/")
        == "templates/runtime.xml.j2"
    )
    assert input_data.memory is not None
    assert input_data.memory.entries[0].id == "mem_yaml"
    assert input_data.environment_state == {"cwd": "repo", "timezone": "Asia/Shanghai"}
    assert input_data.turn_constraints == ["遵循当前请求。", "运行态约束。"]
    assert input_data.metadata == {"source": "yaml", "trace_id": "trace-1"}


def test_load_context_build_input_uses_config_directory_as_template_base(
    tmp_path: Path,
) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "system.xml.j2").write_text(
        '<system_context version="{{ version }}">'
        "<identity>{{ system.identity }}</identity>"
        "</system_context>",
        encoding="utf-8",
    )
    config_path = tmp_path / "context.yaml"
    config_path.write_text(
        """
templates:
  system: templates/system.xml.j2
system:
  identity: 模板助手
""".strip(),
        encoding="utf-8",
    )

    input_data = load_context_build_input(config_path, agent_id="agent")
    output = ContextBuilder().build(input_data)

    assert input_data.template_base_dir == tmp_path
    assert output.system_message.text == (
        '<system_context version="1"><identity>模板助手</identity></system_context>'
    )


def test_load_context_config_rejects_non_object_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "context.yaml"
    config_path.write_text("- bad", encoding="utf-8")

    with pytest.raises(IrisContextError):
        load_context_config(config_path)


@pytest.mark.parametrize(
    "yaml_text",
    [
        """
templates:
  typo: templates/system.xml.j2
system:
  identity: 助手
""",
        """
system:
  identity: 助手
memory:
  enabled: true
  typo: bad
""",
        """
system:
  identity: 助手
slots:
  - name: project_state
    position: before_current_input
    content: ok
    typo: bad
""",
    ],
)
def test_load_context_config_rejects_unknown_nested_fields(
    tmp_path: Path,
    yaml_text: str,
) -> None:
    config_path = tmp_path / "context.yaml"
    config_path.write_text(yaml_text.strip(), encoding="utf-8")

    with pytest.raises(IrisContextError):
        load_context_config(config_path)
