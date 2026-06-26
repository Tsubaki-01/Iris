"""Context YAML 配置加载。"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from ..exceptions import IrisContextError
from .models import ContextBuildInput

_SECTION_NAMES = ("system", "memory", "before_current_input")


def _is_unsupported_windows_template_path(
    value: str,
    *,
    host_is_absolute: bool,
) -> bool:
    windows_path = PureWindowsPath(value)
    return not host_is_absolute and bool(windows_path.anchor or windows_path.drive)


def load_context_build_input(path: str | Path) -> ContextBuildInput:
    """从 YAML 文件加载 ContextBuilder 输入。"""
    config_path = Path(path)
    try:
        content = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise IrisContextError(
            "读取 context 配置失败",
            path=str(config_path),
            error=str(exc),
        ) from exc

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise IrisContextError(
            "解析 context 配置失败",
            path=str(config_path),
            error=str(exc),
        ) from exc

    if not isinstance(data, dict):
        raise IrisContextError(
            "context 配置顶层必须是对象",
            path=str(config_path),
        )

    normalized = _resolve_section_templates(data, config_path=config_path)
    try:
        return ContextBuildInput.model_validate(normalized)
    except (ValidationError, TypeError) as exc:
        raise IrisContextError(
            "context 配置校验失败",
            path=str(config_path),
            error=str(exc),
        ) from exc


def _resolve_section_templates(
    data: dict[Any, Any],
    *,
    config_path: Path,
) -> dict[Any, Any]:
    normalized = data.copy()
    for section_name in _SECTION_NAMES:
        section = normalized.get(section_name)
        if not isinstance(section, dict):
            continue
        normalized_section = section.copy()
        template = normalized_section.get("template")
        if isinstance(template, str | Path):
            template_value = str(template)
            try:
                template_path = Path(template)
                host_is_absolute = template_path.is_absolute()
                if _is_unsupported_windows_template_path(
                    template_value,
                    host_is_absolute=host_is_absolute,
                ):
                    raise IrisContextError(
                        "context 模板路径不能是锚定的非绝对路径",
                        path=str(config_path),
                        section=section_name,
                        template=template_value,
                    )
                if not host_is_absolute:
                    template_path = (config_path.parent / template_path).resolve()
            except IrisContextError:
                raise
            except (OSError, ValueError) as exc:
                raise IrisContextError(
                    "解析 context 模板路径失败",
                    path=str(config_path),
                    section=section_name,
                    template=template_value,
                    error=str(exc),
                ) from exc
            normalized_section["template"] = template_path
        normalized[section_name] = normalized_section
    return normalized


__all__ = ["load_context_build_input"]
