"""固定模板清单、项目补齐及内存来源快照。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ..exceptions import IrisTemplateError
from ..utils import TemplateRenderer

PROMPT_IDS = (
    "compaction",
    "compaction_input",
    "goal_context",
    "goal_continuation",
    "memory_context",
    "memory_dream",
    "memory_flush",
    "memory_overview",
    "memory_recall_instruction",
    "skill_catalog_usage",
    "todo_context",
    "todo_reminder",
    "tool_discovery_instruction",
)
_PROMPT_FILES = {prompt_id: f"{prompt_id}.j2" for prompt_id in PROMPT_IDS}


def _publish_seed(target: Path, content: bytes) -> None:
    """先完成临时文件，再原子发布且不替换已存在的目标。"""
    # 暂存放在模板根之外，避免并发快照枚举到随后被清理的内部文件。
    stream = NamedTemporaryFile(dir=target.parent.parent, prefix=f".{target.name}.", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(content)
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
    finally:
        temporary.unlink()


@dataclass(frozen=True, slots=True)
class PromptSource:
    """已初始化、根目录固定的项目模板来源。"""

    root: Path

    @classmethod
    def initialize(cls, workspace_root: Path, root: str = ".iris/prompts") -> PromptSource:
        """相对 root workspace 解析目录，只补齐缺少的默认模板。

        Args:
            workspace_root: 已选定的 root workspace。
            root: 项目模板目录，可为绝对路径。

        Returns:
            根目录已解析的来源；现有模板正文保持不变。

        Raises:
            IrisTemplateError: 目录创建、默认资源读取或模板发布失败。
        """
        directory = (workspace_root / root).resolve()
        target = directory
        try:
            directory.mkdir(parents=True, exist_ok=True)
            seeds = files("iris.prompts")
            for filename in _PROMPT_FILES.values():
                target = directory / filename
                if not target.exists():
                    _publish_seed(target, seeds.joinpath(filename).read_bytes())
        except OSError as exc:
            raise IrisTemplateError("项目模板初始化失败", path=str(target), error=str(exc)) from exc
        return cls(directory)

    def snapshot(self) -> PromptSnapshot:
        """固定当前目录全部可加载源，后续渲染只使用内存。"""
        return PromptSnapshot(self.root, TemplateRenderer.freeze_directories([self.root]))


@dataclass(frozen=True, slots=True)
class PromptSnapshot:
    """一次操作采用的模板源；业务变量仍在每次渲染时传入。"""

    root: Path
    renderer: TemplateRenderer

    def render(self, prompt_id: str, variables: dict[str, Any]) -> str:
        """按固定 ID 渲染模板，保留 Jinja 原始输出。

        Args:
            prompt_id: 不带扩展名的固定命名模板 ID。
            variables: 当前调用的领域变量。

        Returns:
            当次渲染文本。

        Raises:
            IrisTemplateError: ID 未定义，或模板读取、解析与渲染失败。
        """
        try:
            filename = _PROMPT_FILES[prompt_id]
        except KeyError as exc:
            raise IrisTemplateError(
                "未知项目模板", prompt_id=prompt_id, path=str(self.root)
            ) from exc
        return self.renderer.render_file(self.root / filename, variables)
