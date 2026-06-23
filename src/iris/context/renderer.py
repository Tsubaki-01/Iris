"""上下文系统的 XML 与 Jinja2 渲染器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from ..exceptions import IrisContextError
from .models import ContextPosition, ContextSlot

ROOT_TAGS: dict[ContextPosition, str] = {
    ContextPosition.SYSTEM: "system_context",
    ContextPosition.MEMORY: "memory_context",
    ContextPosition.BEFORE_CURRENT_INPUT: "runtime_context",
}


class ContextXmlRenderer:
    """将结构化 context slot 渲染为 XML。"""

    def render_position(
        self,
        position: ContextPosition,
        slots: list[ContextSlot],
        *,
        version: int,
    ) -> str:
        """渲染某个固定 context 位置中的启用 slot。"""
        root_tag = ROOT_TAGS[position]
        rendered_slots = [
            self.render_slot(slot)
            for slot in sorted(slots, key=lambda item: (item.order, item.name))
        ]
        if not rendered_slots:
            return f'<{root_tag} version="{version}" />'
        body = "\n".join(_indent(slot_xml, spaces=2) for slot_xml in rendered_slots)
        return f'<{root_tag} version="{version}">\n{body}\n</{root_tag}>'

    def render_slot(self, slot: ContextSlot) -> str:
        """将单个 slot 渲染为 XML 元素。"""
        attributes = "".join(
            f" {name}={quoteattr(str(value))}"
            for name, value in sorted(slot.attributes.items())
        )
        inner = _render_value(slot.content)
        if not inner:
            return f"<{slot.name}{attributes} />"
        return f"<{slot.name}{attributes}>{inner}</{slot.name}>"


class ContextTemplateRenderer:
    """从文件渲染 XML Jinja2 模板。"""

    def render_file(
        self,
        template_path: Path,
        context: dict[str, Any],
    ) -> str:
        """使用 XML 自动转义渲染一个 Jinja2 模板文件。"""
        if not template_path.exists():
            raise IrisContextError("context 模板不存在", path=str(template_path))
        if not template_path.is_file():
            raise IrisContextError("context 模板路径不是文件", path=str(template_path))
        try:
            from jinja2 import (
                Environment,
                FileSystemLoader,
                StrictUndefined,
                TemplateError,
                select_autoescape,
            )
        except ImportError as exc:
            raise IrisContextError("渲染 context 模板需要安装 Jinja2") from exc
        environment = Environment(
            loader=FileSystemLoader(str(template_path.parent)),
            autoescape=select_autoescape(
                enabled_extensions=("xml", "j2", "xml.j2"),
                default_for_string=True,
                default=True,
            ),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        try:
            template = environment.get_template(template_path.name)
            return template.render(**context).strip()
        except (TemplateError, TypeError) as exc:
            raise IrisContextError(
                "context 模板渲染失败",
                path=str(template_path),
                error=str(exc),
            ) from exc


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
            rendered = _render_value(item)
            if rendered:
                items.append(f"<item name={quoteattr(str(key))}>{rendered}</item>")
            else:
                items.append(f"<item name={quoteattr(str(key))} />")
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


__all__ = ["ContextTemplateRenderer", "ContextXmlRenderer", "ROOT_TAGS"]
