"""上下文系统构建器。"""

from __future__ import annotations

from typing import Literal

from pydantic_core import PydanticSerializationError

from ..exceptions import IrisContextError
from ..message import Msg
from .models import (
    ContextBuildInput,
    ContextBuildOutput,
    ContextSection,
    ContextSlot,
)
from .renderer import ContextTemplateRenderer, ContextXmlRenderer

CONTEXT_SENDER = "context"
SectionName = Literal["system", "memory", "before_current_input"]

_ROOT_TAGS: dict[SectionName, str] = {
    "system": "system_context",
    "memory": "memory_context",
    "before_current_input": "before_current_input_context",
}


class ContextBuilder:
    """构建三个固定位置的 context 消息。"""

    def __init__(
        self,
        *,
        xml_renderer: ContextXmlRenderer | None = None,
        template_renderer: ContextTemplateRenderer | None = None,
    ) -> None:
        self.xml_renderer = xml_renderer or ContextXmlRenderer()
        self.template_renderer = template_renderer or ContextTemplateRenderer()

    def build(
        self, input_data: ContextBuildInput, *, system_addendum: str = ""
    ) -> ContextBuildOutput:
        """构建 system、memory 和 current input 前置 context 消息。"""
        system_text = self.render_section("system", input_data.system, addendum=system_addendum)
        memory_text = self._render_optional_section("memory", input_data.memory)
        return ContextBuildOutput(
            system=Msg.system(system_text),
            memory=(
                Msg.user(memory_text, sender=CONTEXT_SENDER) if memory_text is not None else None
            ),
            before_current_input=self.build_before_current_input(input_data.before_current_input),
        )

    def build_before_current_input(self, section: ContextSection | None) -> Msg | None:
        """独立构建输入前置消息，避免输入归档时重复渲染固定上下文。"""
        text = self._render_optional_section("before_current_input", section)
        if text is None:
            return None
        return Msg.user(
            text,
            sender=CONTEXT_SENDER,
            metadata={"context_kind": "before_current_input"},
        )

    def _render_optional_section(
        self,
        section_name: SectionName,
        section: ContextSection | None,
    ) -> str | None:
        """渲染一个可选的 context section"""
        if section is None:
            return None
        slots = _enabled_slots(section)
        if not slots:
            return None
        return self.render_section(section_name, section, slots=slots)

    def render_section(
        self,
        section_name: SectionName,
        section: ContextSection,
        *,
        slots: list[ContextSlot] | None = None,
        addendum: str = "",
    ) -> str:
        """渲染一个已定义的 section，供固定上下文和独立动态消息复用。"""
        enabled_slots = slots if slots is not None else _enabled_slots(section)
        if section.template is not None:
            try:
                template_context = {
                    "slots": [slot.model_dump(mode="json") for slot in enabled_slots]
                }
            except PydanticSerializationError as exc:
                raise IrisContextError(
                    "context 模板上下文序列化失败",
                    section=section_name,
                    path=str(section.template),
                    error=str(exc),
                ) from exc
            rendered = self.template_renderer.render_file(
                section.template,
                template_context,
            )
        else:
            rendered = self.xml_renderer._render_trusted_section(
                _ROOT_TAGS[section_name],
                enabled_slots,
            )
        if addendum:
            rendered = f"{rendered}\n\n{addendum}"
        _validate_max_chars(
            rendered,
            section_name=section_name,
            max_chars=section.max_chars,
        )
        return rendered


def _validate_max_chars(
    rendered: str,
    *,
    section_name: SectionName,
    max_chars: int | None,
) -> None:
    if max_chars is None:
        return
    actual = len(rendered)
    if actual > max_chars:
        raise IrisContextError(
            "context section 超出字符上限",
            section=section_name,
            limit=max_chars,
            actual=actual,
        )


def _enabled_slots(section: ContextSection) -> list[ContextSlot]:
    return sorted(
        (slot for slot in section.slots if slot.enabled),
        key=lambda slot: (slot.order, slot.name),
    )


__all__ = ["CONTEXT_SENDER", "ContextBuilder"]
