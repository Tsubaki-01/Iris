"""共享 Jinja 文件加载、编译缓存和纯文本渲染。"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

from jinja2 import (
    BaseLoader,
    Environment,
    FileSystemLoader,
    StrictUndefined,
    Template,
    TemplateError,
    TemplateNotFound,
)
from jinja2.loaders import split_template_path

from ..exceptions import IrisTemplateError
from .sources import SourceDocument


class _FrozenLoader(BaseLoader):
    """保存文件内容，实际使用时才解码与编译。"""

    def __init__(self, sources: dict[str, bytes]) -> None:
        self._sources = sources

    def get_source(
        self, environment: Environment, template: str
    ) -> tuple[str, str | None, Callable[[], bool] | None]:
        """仅从固定集合取得当前依赖，缺失项不会回读磁盘。"""
        name = os.path.normcase("/".join(split_template_path(template)))
        try:
            source = self._sources[name]
        except KeyError as exc:
            raise TemplateNotFound(template) from exc
        return source.decode("utf-8"), None, None


def _environment(loader: BaseLoader) -> Environment:
    """文件源与内存源共享同一组 Jinja 语义。"""
    return Environment(
        loader=loader,
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


class TemplateRenderer:
    """按入口目录复用 Jinja 环境，由模板显式选择 XML 转义。"""

    def __init__(self) -> None:
        """创建实例级编译缓存，不提前读取模板。"""
        self._environments: dict[Path, Environment] = {}
        self._frozen = False
        self._documents: tuple[SourceDocument, ...] | None = None

    @classmethod
    def freeze_directories(cls, directories: Iterable[Path]) -> TemplateRenderer:
        """捕获多个入口目录的全部可加载源，按执行分支惰性编译。

        Args:
            directories: 入口父目录；每个目录的依赖集合彼此隔离。

        Returns:
            只从内存加载模板的 renderer，继续接受动态变量。

        Raises:
            IrisTemplateError: 读取来源文件失败。
        """
        renderer = cls()
        renderer._frozen = True
        for directory in directories:
            directory = directory.resolve()
            path = directory
            try:
                sources = {}
                for name in FileSystemLoader(str(directory)).list_templates():
                    path = directory / name
                    sources[os.path.normcase(name)] = path.read_bytes()
            except OSError as exc:
                raise IrisTemplateError("模板来源读取失败", path=str(path), error=str(exc)) from exc
            renderer._environments[directory] = _environment(_FrozenLoader(sources))
        return renderer

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

    def source_documents(self) -> tuple[SourceDocument, ...]:
        """枚举已冻结源，不访问磁盘或提前编译未使用的模板。"""
        if not self._frozen:
            return ()
        if self._documents is not None:
            return self._documents
        documents = []
        for directory, environment in self._environments.items():
            loader = cast(_FrozenLoader, environment.loader)
            for name, content in loader._sources.items():
                try:
                    text = content.decode("utf-8")
                except UnicodeError:
                    documents.append(
                        SourceDocument("template", str(directory / name), None, "not_utf8")
                    )
                else:
                    documents.append(SourceDocument("template", str(directory / name), text))
        self._documents = tuple(documents)
        return self._documents

    def with_template(self, template_path: Path, source: str) -> TemplateRenderer:
        """在同一冻结来源中替换一个入口，返回独立候选 renderer。

        不读取文件，也不改变原实例；其他来源及动态依赖保持原快照。
        """
        path = template_path.resolve()
        environment = self._environments.get(path.parent)
        if not self._frozen or environment is None:
            raise IrisTemplateError("候选模板需要已冻结的来源", path=str(template_path))
        loader = cast(_FrozenLoader, environment.loader)
        name = os.path.normcase(path.name)
        if name not in loader._sources:
            raise IrisTemplateError("候选模板不在冻结来源中", path=str(template_path))
        candidate = TemplateRenderer()
        candidate._frozen = True
        candidate._environments = dict(self._environments)
        candidate._environments[path.parent] = _environment(
            _FrozenLoader({**loader._sources, name: source.encode("utf-8")})
        )
        return candidate

    def _load_template(self, template_path: Path) -> Template:
        """检查入口更新，依赖按 Jinja 原生执行规则加载。"""
        template_path = template_path.resolve()
        directory = template_path.parent
        environment = self._environments.get(directory)
        if environment is None:
            if self._frozen:
                raise TemplateNotFound(str(template_path))
            environment = _environment(FileSystemLoader(str(directory)))
            self._environments[directory] = environment
        return environment.get_template(template_path.name)
