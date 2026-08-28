"""按 catalog 名称安全读取 SKILL.md 的只读工具。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import (
    IrisSkillNotFoundError,
    IrisSkillPathError,
    IrisToolExecutionError,
    IrisToolValidationError,
)
from ..message import TextBlock
from ..tools import (
    BaseTool,
    ReadFileInput,
    ToolCapability,
    ToolDefinition,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolResult,
    WorkspaceFileService,
    schema_from_pydantic_model,
)
from ._paths import is_resolved_within
from .models import _NAME_RE, SkillMetadata
from .registry import SkillRegistry

logger = logging.getLogger(__name__)


class LoadSkillInput(BaseModel):
    """load_skill 仅允许模型提交 registry 名称。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=_NAME_RE.pattern)


class LoadSkillTool(BaseTool):
    """按名称读取当前 live SKILL.md，不执行其中内容。"""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        file_service: WorkspaceFileService | None = None,
        max_result_chars: int = 50000,
    ) -> None:
        self.registry = registry
        self.file_service = file_service or WorkspaceFileService()
        self.definition = ToolDefinition(
            name="load_skill",
            description=(
                "按 catalog 中的 skill name 读取对应 SKILL.md 指令；"
                "只返回 Markdown，不执行脚本"
            ),
            input_schema=schema_from_pydantic_model(LoadSkillInput),
            capabilities={ToolCapability.READ},
            group="skill",
            deferred=False,
            max_result_chars=max_result_chars,
        )

    @property
    def input_model(self) -> type[BaseModel]:
        """返回严格的名称输入模型。"""
        return LoadSkillInput

    def validate_input(self, params: dict[str, Any]) -> BaseModel:
        """使用 Pydantic 校验模型输入。"""
        return LoadSkillInput.model_validate(params)

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """复查 live path 后通过共享文件服务读取 Skill。"""
        input_data = LoadSkillInput.model_validate(params)
        try:
            metadata = self.registry.get(input_data.name)
        except IrisSkillNotFoundError:
            return self._error_result(
                code="SKILL_NOT_FOUND",
                message=f"Skill 不存在: {input_data.name}",
                retryable=True,
                details={
                    "name": input_data.name,
                    "available": list(self.registry.names()),
                },
            )

        try:
            live_file = self._resolve_live_file(metadata, context)
        except IrisSkillPathError as exc:
            return self._path_error(input_data.name, exc)

        try:
            service_target = self.file_service.resolve_path(
                metadata.relative_skill_file,
                context,
            )
        except IrisToolValidationError as exc:
            return self._path_error(
                input_data.name,
                IrisSkillPathError("Skill 文件路径越出 workspace", path=str(exc)),
            )
        if service_target != live_file:
            return self._path_error(
                input_data.name,
                IrisSkillPathError(
                    "Skill metadata 与文件服务目标不一致",
                    path=str(service_target),
                    expected=str(live_file),
                ),
            )

        try:
            text = self.file_service.read_file(
                ReadFileInput(file_path=metadata.relative_skill_file),
                context,
            )
        except IrisToolValidationError as exc:
            if exc.message.startswith("PATH_OUTSIDE_WORKSPACE"):
                return self._path_error(
                    input_data.name,
                    IrisSkillPathError("Skill 文件路径越出 workspace", path=str(exc)),
                )
            raise
        except IrisToolExecutionError as exc:
            reason = (
                "SKILL.md 不存在或不是普通文件"
                if exc.message.startswith("FILE_NOT_FOUND:")
                else "SKILL.md 读取失败"
            )
            return self._read_error(input_data.name, reason)
        except UnicodeDecodeError:
            return self._read_error(
                input_data.name,
                "SKILL.md 不是有效的 UTF-8 文本",
            )
        except OSError:
            return self._read_error(input_data.name, "SKILL.md 读取失败")

        return ToolResult(
            tool_use_id="",
            tool_name="load_skill",
            content=[TextBlock(text=text)] if text else [],
        )

    def _resolve_live_file(
        self,
        metadata: SkillMetadata,
        context: ToolExecutionContext,
    ) -> Path:
        workspace = context.workspace_root.resolve(strict=False)
        root = metadata.root_dir.resolve(strict=False)
        skill_file = metadata.skill_file.resolve(strict=False)
        if not is_resolved_within(root, workspace):
            raise IrisSkillPathError(
                "Skill root 不在 workspace 内",
                path=str(root),
                workspace_root=str(workspace),
            )
        if not is_resolved_within(skill_file, root):
            raise IrisSkillPathError(
                "Skill 文件不在 Skill root 内",
                path=str(skill_file),
                root=str(root),
            )
        return skill_file

    def _path_error(self, name: str, error: IrisSkillPathError) -> ToolResult:
        logger.warning("load_skill path validation failed for %s: %s", name, error)
        return self._error_result(
            code="SKILL_PATH_ERROR",
            message=f"Skill 路径已失效或越出允许范围: {name}",
            retryable=False,
            details={"name": name},
        )

    def _read_error(self, name: str, reason: str) -> ToolResult:
        return self._error_result(
            code="SKILL_READ_ERROR",
            message=f"无法读取 Skill: {name}",
            retryable=True,
            details={"name": name, "reason": reason},
        )

    @staticmethod
    def _error_result(
        *,
        code: str,
        message: str,
        retryable: bool,
        details: dict[str, Any],
    ) -> ToolResult:
        return ToolResult(
            tool_use_id="",
            tool_name="load_skill",
            is_error=True,
            error=ToolErrorInfo(
                code=code,
                message=message,
                retryable=retryable,
                details=details,
            ),
        )


__all__ = ["LoadSkillTool"]
