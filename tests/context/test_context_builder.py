from __future__ import annotations

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
