"""上下文系统构建器。"""

from __future__ import annotations

from typing import Any, Literal

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

    def build(self, input_data: ContextBuildInput) -> ContextBuildOutput:
        """构建 system、memory 和 current input 前置 context 消息。"""
        system_text = self._render_section("system", input_data.system)
        memory_text = self._render_optional_section("memory", input_data.memory)
        before_input_text = self._render_optional_section(
            "before_current_input",
            input_data.before_current_input,
        )
        return ContextBuildOutput(
            system=Msg.system(system_text),
            memory=(
                Msg.user(memory_text, sender=CONTEXT_SENDER) if memory_text is not None else None
            ),
            before_current_input=(
                Msg.user(before_input_text, sender=CONTEXT_SENDER)
                if before_input_text is not None
                else None
            ),
        )

    def fingerprint_payload(self, input_data: ContextBuildInput) -> dict[str, Any]:
        """返回实际启用的结构化输入与模板版本，不执行渲染。

        Args:
            input_data (ContextBuildInput): 运行环境已经加载的 context 输入。

        Returns:
            dict[str, Any]: 忽略来源路径和未启用内容的恢复指纹输入。

        Raises:
            IrisContextError: 有效模板来源无法读取或解析。
        """
        payload: dict[str, Any] = {}
        for section_name, section in (
            ("system", input_data.system),
            ("memory", input_data.memory),
            ("before_current_input", input_data.before_current_input),
        ):
            if section is None:
                continue
            slots = _enabled_slots(section)
            # memory 可在 step 0 由 run options 注入 slots，因此其配置也属于恢复环境。
            if not slots and (
                section_name != "memory" or (section.template is None and section.max_chars is None)
            ):
                continue
            payload[section_name] = {
                "slots": [slot.model_dump(mode="json") for slot in slots],
                "max_chars": section.max_chars,
                "template_version": (
                    self.template_renderer.content_version(section.template)
                    if section.template is not None
                    else None
                ),
            }
        return payload

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
        return self._render_section(section_name, section, slots=slots)

    def _render_section(
        self,
        section_name: SectionName,
        section: ContextSection,
        *,
        slots: list[ContextSlot] | None = None,
    ) -> str:
        """渲染一个 context section"""
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
