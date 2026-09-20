from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest

from iris.context import (
    CONTEXT_SENDER,
    ContextBuilder,
    ContextBuildInput,
    ContextSection,
    ContextSlot,
    ContextXmlRenderer,
)
from iris.exceptions import IrisContextError
from iris.message import Role


def test_builder_renders_three_sections_with_expected_roles_and_roots() -> None:
    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                slots=[ContextSlot(name="base_instructions", content="你是助手")]
            ),
            memory=ContextSection(
                slots=[
                    ContextSlot(
                        name="memory",
                        content="用户偏好简洁回答",
                        attributes={"source": "sqlite"},
                    )
                ]
            ),
            before_current_input=ContextSection(
                slots=[
                    ContextSlot(
                        name="environment_state",
                        content={"cwd": "J:/repo"},
                    )
                ]
            ),
        )
    )

    assert output.system.role == Role.SYSTEM
    assert output.system.text == (
        "<system_context>\n  <base_instructions>你是助手</base_instructions>\n</system_context>"
    )
    assert "version" not in output.system.text

    assert output.memory is not None
    assert output.memory.role == Role.USER
    assert output.memory.sender == CONTEXT_SENDER
    assert output.memory.text.startswith("<memory_context>")
    assert '<memory source="sqlite">用户偏好简洁回答</memory>' in output.memory.text
    assert "version" not in output.memory.text

    assert output.before_current_input is not None
    assert output.before_current_input.role == Role.USER
    assert output.before_current_input.sender == CONTEXT_SENDER
    assert output.before_current_input.text.startswith("<before_current_input_context>")
    assert "version" not in output.before_current_input.text


def test_builder_filters_disabled_slots_and_sorts_by_order_then_name() -> None:
    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                slots=[
                    ContextSlot(name="late", content="late", order=20),
                    ContextSlot(name="z_same", content="z", order=10),
                    ContextSlot(name="a_same", content="a", order=10),
                    ContextSlot(
                        name="disabled",
                        content="disabled",
                        order=1,
                        enabled=False,
                    ),
                ]
            )
        )
    )

    text = output.system.text
    assert "disabled" not in text
    assert text.index("<a_same>") < text.index("<z_same>") < text.index("<late>")


def test_builder_preserves_nested_values_attributes_and_xml_escaping() -> None:
    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                slots=[
                    ContextSlot(
                        name="structured",
                        content={
                            "enabled": True,
                            "items": ["<first>", None],
                        },
                        attributes={"source": "local & remote"},
                    )
                ]
            )
        )
    )

    root = ElementTree.fromstring(output.system.text)
    structured = root.find("structured")

    assert structured is not None
    assert structured.attrib == {"source": "local & remote"}

    enabled = structured.find("./item[@name='enabled']")
    assert enabled is not None
    assert enabled.text == "true"

    items = structured.find("./item[@name='items']")
    assert items is not None
    assert [item.text for item in items.findall("item")] == ["<first>", None]


def test_xml_renderer_rejects_unsafe_root_tag() -> None:
    with pytest.raises(IrisContextError):
        ContextXmlRenderer().render_section(
            "x><injected",
            [ContextSlot(name="instructions", content="content")],
        )


def test_system_addendum_follows_custom_template(tmp_path: Path) -> None:
    """自定义模板不引用额外 slot 时，窗口文本仍在同一 system 消息中。"""
    template = tmp_path / "system.j2"
    template.write_text("{{ slots[0].content }}", encoding="utf-8")
    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                slots=[ContextSlot(name="instructions", content="基础指令")], template=template
            )
        ),
        system_addendum="记忆概览",
    )
    assert output.system.role is Role.SYSTEM
    assert output.system.text == "基础指令\n\n记忆概览"
    assert output.memory is None


def test_system_max_chars_includes_addendum(tmp_path: Path) -> None:
    """字符上限拥有最终 system 文本，而不是只限制模板正文。"""
    template = tmp_path / "system.j2"
    template.write_text("base", encoding="utf-8")
    context = ContextBuildInput(
        system=ContextSection(
            slots=[ContextSlot(name="instructions", content="base")],
            template=template,
            max_chars=9,
        )
    )
    assert ContextBuilder().build(context, system_addendum="abc").system.text == "base\n\nabc"
    with pytest.raises(IrisContextError, match="字符上限"):
        ContextBuilder().build(context, system_addendum="abcd")
