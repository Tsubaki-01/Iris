"""上下文系统的 XML 与 Jinja2 渲染器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    Template,
    TemplateError,
    select_autoescape,
)

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


class ContextTemplateRenderer:
    """使用 Jinja 原生加载、编译缓存和默认更新检测渲染 XML 模板。"""

    def __init__(self) -> None:
        """按入口目录复用 Environment，由 Jinja 管理模板缓存。"""
        self._environments: dict[Path, Environment] = {}

    def render_file(
        self,
        template_path: Path,
        context: dict[str, Any],
    ) -> str:
        """使用 XML 自动转义和当前数据渲染模板。

        Args:
            template_path (Path): 模板入口文件路径。
            context (dict[str, Any]): 本次渲染的数据；不会写入编译模板缓存。

        Returns:
            str: 去除首尾空白后的模板输出。

        Raises:
            IrisContextError: 模板读取、解析或执行失败。
        """
        try:
            template = self._load_template(template_path)
        except (OSError, UnicodeError, TemplateError) as exc:
            raise IrisContextError(
                "context 模板来源读取或解析失败",
                path=str(template_path),
                error=str(exc),
            ) from exc
        try:
            return template.render(**context).strip()
        except Exception as exc:
            raise IrisContextError(
                "context 模板渲染失败",
                path=str(template_path),
                error=str(exc),
            ) from exc

    def _load_template(self, template_path: Path) -> Template:
        """每次通过 get_template 检查入口更新，依赖由 Jinja 按需加载。"""
        template_path = template_path.resolve()
        directory = template_path.parent
        environment = self._environments.get(directory)
        if environment is None:
            environment = Environment(
                loader=FileSystemLoader(str(directory)),
                autoescape=select_autoescape(
                    enabled_extensions=("xml", "j2", "xml.j2"),
                    default_for_string=True,
                    default=True,
                ),
                undefined=StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            self._environments[directory] = environment
        return environment.get_template(template_path.name)


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


__all__ = ["ContextTemplateRenderer", "ContextXmlRenderer"]
