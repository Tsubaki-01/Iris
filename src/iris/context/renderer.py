"""上下文系统的 XML 与 Jinja2 渲染器。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from ..exceptions import IrisContextError
from .models import ContextSlot, _is_safe_xml_name

_NUMERIC_CHARACTER_REFERENCE_RE = re.compile(
    r"&#(?:(?P<hex>[xX][0-9A-Fa-f]+)|(?P<decimal>[0-9]+));"
)


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
        rendered_slots = [self.render_slot(slot) for slot in slots]
        body = "\n".join(_indent(slot_xml, spaces=2) for slot_xml in rendered_slots)
        return f"<{root_tag}>\n{body}\n</{root_tag}>"

    def render_slot(self, slot: ContextSlot) -> str:
        """将单个 slot 渲染为 XML 元素。"""
        attributes = "".join(
            f" {name}={quoteattr(_validate_xml_text(str(value), location='attribute'))}"
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
            rendered = template.render(**context).strip()
        except Exception as exc:
            raise IrisContextError(
                "context 模板渲染失败",
                path=str(template_path),
                error=str(exc),
            ) from exc
        _validate_xml_text(rendered, location="template")
        return rendered


def _render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return escape(_validate_xml_text(value, location="content"))
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return _validate_xml_text(str(value), location="content")
    if isinstance(value, dict):
        items: list[str] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            rendered_key = _validate_xml_text(str(key), location="attribute")
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
    rendered = _validate_xml_text(str(value), location="content")
    return escape(rendered)


def _validate_xml_text(value: str, *, location: str) -> str:
    for character in value:
        _validate_xml_codepoint(ord(character), location=location)
    if location != "template":
        return value
    for match in _NUMERIC_CHARACTER_REFERENCE_RE.finditer(value):
        hexadecimal = match.group("hex")
        codepoint = (
            int(hexadecimal[1:], 16)
            if hexadecimal is not None
            else int(match.group("decimal"), 10)
        )
        _validate_xml_codepoint(codepoint, location=location)
    return value


def _validate_xml_codepoint(codepoint: int, *, location: str) -> None:
    if (
        codepoint in (0x09, 0x0A, 0x0D)
        or 0x20 <= codepoint <= 0xD7FF
        or 0xE000 <= codepoint <= 0xFFFD
        or 0x10000 <= codepoint <= 0x10FFFF
    ):
        return
    raise IrisContextError(
        "context XML 包含非法字符",
        codepoint=f"U+{codepoint:04X}",
        location=location,
    )


def _indent(text: str, *, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())


__all__ = ["ContextTemplateRenderer", "ContextXmlRenderer"]
