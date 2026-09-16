"""上下文系统的 XML 与 Jinja2 渲染器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from jinja2 import (
    DictLoader,
    Environment,
    FileSystemLoader,
    StrictUndefined,
    Template,
    TemplateError,
    TemplateNotFound,
    meta,
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
    """从首次读取的来源快照渲染 XML Jinja2 模板。

    每个模板及其静态依赖只读取一次；修改文件后应创建新的 renderer。
    """

    def __init__(self) -> None:
        """为当前 renderer 保留按模板入口索引的来源快照。"""
        self._snapshots: dict[Path, Template] = {}

    def render_file(
        self,
        template_path: Path,
        context: dict[str, Any],
    ) -> str:
        """使用 XML 自动转义渲染同一来源快照。

        Args:
            template_path (Path): 模板入口文件路径。
            context (dict[str, Any]): 本次渲染的数据；不会写入来源快照。

        Returns:
            str: 去除首尾空白后的模板输出。

        Raises:
            IrisContextError: 来源无法冻结或模板执行失败。
        """
        template = self._snapshot(template_path)
        try:
            return template.render(**context).strip()
        except Exception as exc:
            raise IrisContextError(
                "context 模板渲染失败",
                path=str(template_path),
                error=str(exc),
            ) from exc

    def _snapshot(self, template_path: Path) -> Template:
        """冻结入口和依赖，后续渲染复用同一来源。"""
        template_path = template_path.resolve()
        cached = self._snapshots.get(template_path)
        if cached is not None:
            return cached
        loader = FileSystemLoader(str(template_path.parent))
        environment = Environment(
            loader=loader,
            autoescape=select_autoescape(
                enabled_extensions=("xml", "j2", "xml.j2"),
                default_for_string=True,
                default=True,
            ),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            auto_reload=False,
        )
        try:
            sources = _read_template_sources(environment, loader, template_path.name)
            environment.loader = DictLoader(sources)
            template = environment.get_template(template_path.name)
        except (OSError, UnicodeError, TemplateError) as exc:
            raise IrisContextError(
                "context 模板来源读取或解析失败",
                path=str(template_path),
                error=str(exc),
            ) from exc
        self._snapshots[template_path] = template
        return template


def _read_template_sources(
    environment: Environment,
    loader: FileSystemLoader,
    root_name: str,
) -> dict[str, str]:
    """只读取静态引用闭包，保留可选依赖在首次读取时的缺失事实。"""
    sources: dict[str, str] = {}
    visited: set[str] = set()
    pending = [root_name]
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        try:
            source, _, _ = loader.get_source(environment, name)
        except TemplateNotFound:
            if name == root_name:
                raise
            # 缺失的静态候选不加入快照，运行时仍由 Jinja 处理 ignore missing / 列表备用。
            continue
        sources[name] = source
        for dependency in meta.find_referenced_templates(environment.parse(source)):
            if dependency is None:
                raise IrisContextError(
                    "context 模板依赖必须使用静态文件名；请在条件分支中分别 include/import/extends "
                    "固定文件名，而不是使用动态文件名表达式",
                    template=name,
                )
            pending.append(dependency)
    return sources


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
