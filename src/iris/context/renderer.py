"""将结构化上下文渲染为 XML。"""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape, quoteattr

from ..exceptions import IrisContextError
from .models import ContextSlot, _is_safe_xml_name


class ContextXmlRenderer:
    """将结构化 context slot 渲染为 XML。"""

    def render_section(
        self,
        root_tag: str,
        slots: list[ContextSlot],
    ) -> str:
        """将一个 section 的 slot 渲染为 XML。"""
        if not _is_safe_xml_name(root_tag):
            raise IrisContextError("context XML 根标签必须是安全的 XML 名称")
        return self._render_trusted_section(root_tag, slots)

    def _render_trusted_section(
        self,
        root_tag: str,
        slots: list[ContextSlot],
    ) -> str:
        """渲染框架已验证或固定定义的 XML 根标签。"""
        rendered_slots = [self.render_slot(slot) for slot in slots]
        body = "\n".join(_indent(slot_xml, spaces=2) for slot_xml in rendered_slots)
        return f"<{root_tag}>\n{body}\n</{root_tag}>"

    def render_slot(self, slot: ContextSlot) -> str:
        """将单个 slot 渲染为 XML 元素。"""
        attributes = "".join(
            f" {name}={quoteattr(str(value))}" for name, value in sorted(slot.attributes.items())
        )
        inner = _render_value(slot.content)
        if not inner:
            return f"<{slot.name}{attributes} />"
        return f"<{slot.name}{attributes}>{inner}</{slot.name}>"


def _render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return escape(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, dict):
        items: list[str] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            rendered_key = str(key)
            rendered = _render_value(item)
            if rendered:
                items.append(f"<item name={quoteattr(rendered_key)}>{rendered}</item>")
            else:
                items.append(f"<item name={quoteattr(rendered_key)} />")
        return "\n" + "\n".join(_indent(item, spaces=2) for item in items) + "\n"
    if isinstance(value, list | tuple):
        items = []
        for item in value:
            rendered = _render_value(item)
            if rendered:
                items.append(f"<item>{rendered}</item>")
            else:
                items.append("<item />")
        return "\n" + "\n".join(_indent(item, spaces=2) for item in items) + "\n"
    return escape(str(value))


def _indent(text: str, *, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())


__all__ = ["ContextXmlRenderer"]
