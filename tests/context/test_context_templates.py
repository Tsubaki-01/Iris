from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_core import PydanticSerializationError

from iris.context import (
    ContextBuilder,
    ContextBuildInput,
    ContextSection,
    ContextSlot,
    ContextTemplateRenderer,
)
from iris.exceptions import IrisContextError


def _system_section() -> ContextSection:
    return ContextSection(slots=[ContextSlot(name="instructions", content="content")])


class _UnserializableContent:
    pass


class _MutatingTemplateRenderer(ContextTemplateRenderer):
    def render_file(
        self,
        template_path: Path,
        context: dict[str, Any],
    ) -> str:
        slot = context["slots"][0]
        slot["content"]["nested"].append("mutated")
        slot["attributes"]["source"] = "mutated"
        return "<system_context />"


def test_template_only_receives_sorted_enabled_slots(tmp_path: Path) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_text(
        "<system_context>"
        "<system_defined>{{ system is defined }}</system_defined>"
        "{% for slot in slots %}"
        "<{{ slot.name }}>{{ slot.content }}</{{ slot.name }}>"
        "{% endfor %}"
        "</system_context>",
        encoding="utf-8",
    )

    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                template=template,
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
                ],
            )
        )
    )

    assert output.system.text == (
        "<system_context><system_defined>False</system_defined>"
        "<a_same>a</a_same><z_same>z</z_same><late>late</late>"
        "</system_context>"
    )


def test_template_content_keeps_xml_autoescaping(tmp_path: Path) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_text(
        "<system_context>{{ slots[0].content }}</system_context>",
        encoding="utf-8",
    )

    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                template=template,
                slots=[
                    ContextSlot(
                        name="instructions",
                        content="<local> & remote",
                    )
                ],
            )
        )
    )

    assert output.system.text == (
        "<system_context>&lt;local&gt; &amp; remote</system_context>"
    )


def test_default_xml_escapes_numeric_character_reference_as_text() -> None:
    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                slots=[ContextSlot(name="instructions", content="&#x1;")]
            )
        )
    )

    assert output.system.text == (
        "<system_context>\n"
        "  <instructions>&amp;#x1;</instructions>\n"
        "</system_context>"
    )


def test_default_xml_allows_exact_limit_and_rejects_one_less() -> None:
    builder = ContextBuilder()
    rendered = builder.build(ContextBuildInput(system=_system_section())).system.text
    exact_section = ContextSection(
        max_chars=len(rendered),
        slots=[ContextSlot(name="instructions", content="content")],
    )

    exact_output = builder.build(ContextBuildInput(system=exact_section))

    assert exact_output.system.text == rendered

    limit = len(rendered) - 1
    with pytest.raises(IrisContextError) as exc_info:
        builder.build(
            ContextBuildInput(
                system=exact_section.model_copy(update={"max_chars": limit})
            )
        )

    assert exc_info.value.context == {
        "section": "system",
        "limit": limit,
        "actual": len(rendered),
    }


def test_template_limit_checks_complete_rendered_text(tmp_path: Path) -> None:
    template = tmp_path / "memory.xml.j2"
    template.write_text(
        "<memory_context>\n"
        "  <fixed>prefix</fixed>\n"
        "  <item>{{ slots[0].content }}</item>\n"
        "</memory_context>",
        encoding="utf-8",
    )
    slot = ContextSlot(name="memory", content="remember")
    rendered = ContextTemplateRenderer().render_file(
        template,
        {"slots": [slot.model_dump(mode="json")]},
    )
    limit = len(rendered) - 1

    with pytest.raises(IrisContextError) as exc_info:
        ContextBuilder().build(
            ContextBuildInput(
                system=_system_section(),
                memory=ContextSection(
                    template=template,
                    max_chars=limit,
                    slots=[slot],
                ),
            )
        )

    assert exc_info.value.context == {
        "section": "memory",
        "limit": limit,
        "actual": len(rendered),
    }


def test_template_serialization_error_uses_context_error(tmp_path: Path) -> None:
    template = tmp_path / "system.xml.j2"
    section = ContextSection(
        template=template,
        slots=[
            ContextSlot(
                name="instructions",
                content=_UnserializableContent(),
            )
        ],
    )

    with pytest.raises(IrisContextError) as exc_info:
        ContextBuilder().build(ContextBuildInput(system=section))

    assert exc_info.value.message == "context 模板上下文序列化失败"
    assert exc_info.value.context["section"] == "system"
    assert exc_info.value.context["path"] == str(template)
    assert exc_info.value.context["error"]
    assert isinstance(exc_info.value.__cause__, PydanticSerializationError)


def test_template_context_copy_does_not_modify_section_or_slots(
    tmp_path: Path,
) -> None:
    template = tmp_path / "system.xml.j2"
    section = ContextSection(
        template=template,
        slots=[
            ContextSlot(
                name="instructions",
                content={"nested": ["value"]},
                attributes={"source": "local"},
            ),
            ContextSlot(
                name="disabled",
                content="disabled",
                enabled=False,
            ),
        ],
    )
    original = section.model_dump(mode="python")

    ContextBuilder(template_renderer=_MutatingTemplateRenderer()).build(
        ContextBuildInput(system=section)
    )

    assert section.model_dump(mode="python") == original


def test_empty_optional_section_does_not_open_missing_template(
    tmp_path: Path,
) -> None:
    output = ContextBuilder().build(
        ContextBuildInput(
            system=_system_section(),
            memory=ContextSection(
                template=tmp_path / "missing.xml.j2",
                slots=[
                    ContextSlot(
                        name="memory",
                        content="disabled",
                        enabled=False,
                    )
                ],
            ),
        )
    )

    assert output.memory is None


def test_missing_template_file_uses_context_error(tmp_path: Path) -> None:
    missing_template = tmp_path / "missing.xml.j2"

    with pytest.raises(IrisContextError) as exc_info:
        ContextBuilder().build(
            ContextBuildInput(
                system=ContextSection(
                    template=missing_template,
                    slots=[ContextSlot(name="instructions", content="content")],
                )
            )
        )

    assert exc_info.value.context["path"] == str(missing_template)


def test_template_undefined_variable_uses_context_error(tmp_path: Path) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_text("{{ missing_value }}", encoding="utf-8")

    with pytest.raises(IrisContextError) as exc_info:
        ContextBuilder().build(
            ContextBuildInput(
                system=ContextSection(
                    template=template,
                    slots=[ContextSlot(name="instructions", content="content")],
                )
            )
        )

    assert exc_info.value.context["path"] == str(template)


def test_template_invalid_utf8_uses_context_error(tmp_path: Path) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_bytes(b"\xff")

    with pytest.raises(IrisContextError) as exc_info:
        ContextBuilder().build(
            ContextBuildInput(
                system=ContextSection(
                    template=template,
                    slots=[ContextSlot(name="instructions", content="content")],
                )
            )
        )

    assert exc_info.value.message == "context 模板渲染失败"
    assert exc_info.value.context["path"] == str(template)
    assert exc_info.value.context["error"]
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_template_runtime_error_uses_context_error(tmp_path: Path) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_text("{{ 1 / 0 }}", encoding="utf-8")

    with pytest.raises(IrisContextError) as exc_info:
        ContextBuilder().build(
            ContextBuildInput(
                system=ContextSection(
                    template=template,
                    slots=[ContextSlot(name="instructions", content="content")],
                )
            )
        )

    assert exc_info.value.message == "context 模板渲染失败"
    assert exc_info.value.context["path"] == str(template)
    assert exc_info.value.context["error"]
    assert isinstance(exc_info.value.__cause__, ZeroDivisionError)


def test_template_preserves_fixed_control_character(tmp_path: Path) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_text("<system_context>\x01</system_context>", encoding="utf-8")

    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                template=template,
                slots=[ContextSlot(name="instructions", content="content")],
            )
        )
    )

    assert output.system.text == "<system_context>\x01</system_context>"


def test_template_preserves_slot_control_character(tmp_path: Path) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_text(
        "<system_context>{{ slots[0].content }}</system_context>",
        encoding="utf-8",
    )

    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                template=template,
                slots=[ContextSlot(name="instructions", content="a\x01b")],
            )
        )
    )

    assert output.system.text == "<system_context>a\x01b</system_context>"


@pytest.mark.parametrize("reference", ["&#1;", "&#x1;", "&#X1;"])
def test_template_preserves_invalid_numeric_character_reference_as_prompt_text(
    tmp_path: Path,
    reference: str,
) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_text(reference, encoding="utf-8")

    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                template=template,
                slots=[ContextSlot(name="instructions", content="content")],
            )
        )
    )

    assert output.system.text == reference


@pytest.mark.parametrize(
    "reference",
    [
        "&#xD800;",
        "&#x110000;",
    ],
)
def test_template_preserves_out_of_range_numeric_character_reference_as_prompt_text(
    tmp_path: Path,
    reference: str,
) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_text(reference, encoding="utf-8")

    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                template=template,
                slots=[ContextSlot(name="instructions", content="content")],
            )
        )
    )

    assert output.system.text == reference


@pytest.mark.parametrize(
    "reference",
    ["&#9;", "&#xA;", "&#XA;", "&#x20;", "&#x1F600;"],
)
def test_template_allows_valid_numeric_character_reference(
    tmp_path: Path,
    reference: str,
) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_text(reference, encoding="utf-8")

    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                template=template,
                slots=[ContextSlot(name="instructions", content="content")],
            )
        )
    )

    assert output.system.text == reference


def test_template_allows_escaped_numeric_character_reference(
    tmp_path: Path,
) -> None:
    template = tmp_path / "system.xml.j2"
    template.write_text("&amp;#x1;", encoding="utf-8")

    output = ContextBuilder().build(
        ContextBuildInput(
            system=ContextSection(
                template=template,
                slots=[ContextSlot(name="instructions", content="content")],
            )
        )
    )

    assert output.system.text == "&amp;#x1;"


def test_before_current_input_limit_uses_section_name() -> None:
    section = ContextSection(
        max_chars=1,
        slots=[ContextSlot(name="environment_state", content="state")],
    )

    with pytest.raises(IrisContextError) as exc_info:
        ContextBuilder().build(
            ContextBuildInput(
                system=_system_section(),
                before_current_input=section,
            )
        )

    assert exc_info.value.context["section"] == "before_current_input"
