"""共享 Jinja 文件加载、编译缓存和纯文本渲染。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template, TemplateError

from ..exceptions import IrisTemplateError


class TemplateRenderer:
    """按入口目录复用 Jinja 环境，由模板显式选择 XML 转义。"""

    def __init__(self) -> None:
        """创建实例级编译缓存，不提前读取模板。"""
        self._environments: dict[Path, Environment] = {}

    def render_file(self, template_path: Path, context: dict[str, Any]) -> str:
        """使用当前变量渲染文件，保留 Jinja 原生输出。

        Args:
            template_path: 模板入口路径，依赖从入口父目录加载。
            context: 当次变量，不保存为环境 globals 或最终结果缓存。

        Returns:
            渲染后的文本，不额外裁剪首尾空白。

        Raises:
            IrisTemplateError: 模板读取、解析或执行失败。
        """
        try:
            template = self._load_template(template_path)
        except (OSError, UnicodeError, TemplateError) as exc:
            raise IrisTemplateError(
                "模板来源读取或解析失败", path=str(template_path), error=str(exc)
            ) from exc
        try:
            return template.render(**context)
        except Exception as exc:
            raise IrisTemplateError(
                "模板渲染失败", path=str(template_path), error=str(exc)
            ) from exc

    def _load_template(self, template_path: Path) -> Template:
        """检查入口更新，依赖按 Jinja 原生执行规则加载。"""
        template_path = template_path.resolve()
        directory = template_path.parent
        environment = self._environments.get(directory)
        if environment is None:
            environment = Environment(
                loader=FileSystemLoader(str(directory)),
                autoescape=False,
                undefined=StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            self._environments[directory] = environment
        return environment.get_template(template_path.name)
