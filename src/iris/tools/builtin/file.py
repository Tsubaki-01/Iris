"""Workspace 文件工具。"""

from __future__ import annotations

import re
import tempfile
from itertools import islice
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from ...exceptions import IrisToolExecutionError, IrisToolValidationError
from ...message import TextBlock
from ..base import BaseTool, ToolCapability, ToolDefinition, ToolExecutionContext, ToolResult
from ..permissions import ReadFileState, WorkspacePolicy
from ..registry import ToolRegistry
from ..schema import schema_from_pydantic_model


class ReadFileInput(BaseModel):
    """读取文件参数。"""

    file_path: str
    offset: int | None = None
    limit: int | None = None

    @field_validator("offset", "limit")
    @classmethod
    def _validate_non_negative(cls, value: int | None) -> int | None:
        """校验分页参数非负。"""
        if value is not None and value < 0:
            raise ValueError("offset/limit 必须非负")
        return value

    @field_validator("limit")
    @classmethod
    def _validate_limit(cls, value: int | None) -> int | None:
        """限制单次读取行数。"""
        if value is not None and value > 10000:
            raise ValueError("limit 不能超过 10000")
        return value


class ListFilesInput(BaseModel):
    """列出文件参数。"""

    path: str = "."
    pattern: str | None = None
    max_results: int = 200

    @field_validator("max_results")
    @classmethod
    def _validate_max_results(cls, value: int) -> int:
        """限制最大结果数。"""
        if value < 0 or value > 10000:
            raise ValueError("max_results 必须在 0..10000 范围内")
        return value


class GrepSearchInput(BaseModel):
    """文本搜索参数。"""

    pattern: str
    path: str = "."
    max_results: int = 200

    @field_validator("max_results")
    @classmethod
    def _validate_max_results(cls, value: int) -> int:
        """限制最大结果数。"""
        if value < 0 or value > 10000:
            raise ValueError("max_results 必须在 0..10000 范围内")
        return value


class WriteFileInput(BaseModel):
    """写入文件参数。"""

    file_path: str
    content: str


class EditFileInput(BaseModel):
    """编辑文件参数。"""

    file_path: str
    old_string: str
    new_string: str

    @field_validator("old_string")
    @classmethod
    def _validate_old_string(cls, value: str) -> str:
        """替换源字符串不能为空。"""
        if not value:
            raise ValueError("old_string 不能为空")
        return value


class FileTool(BaseTool):
    """文件工具基类。"""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[BaseModel],
        capabilities: set[ToolCapability],
        max_result_chars: int,
    ) -> None:
        """初始化文件工具定义。"""
        self._input_model = input_model
        self.workspace_policy = WorkspacePolicy()
        self.definition = ToolDefinition(
            name=name,
            description=description,
            input_schema=schema_from_pydantic_model(input_model),
            capabilities=capabilities,
            group="file",
            max_result_chars=max_result_chars,
        )

    def input_model(self) -> type[BaseModel] | None:
        """返回输入模型。"""
        return self._input_model

    def validate_input(self, params: dict[str, Any]) -> BaseModel:
        """校验输入参数。"""
        return self._input_model.model_validate(params)

    def _resolve(self, file_path: str, context: ToolExecutionContext) -> Path:
        """解析 workspace 内路径。"""
        return self.workspace_policy.resolve_path(file_path, workspace_root=context.workspace_root)

    def _read_state(self, context: ToolExecutionContext) -> ReadFileState:
        """获取或初始化上下文读取状态。"""
        if context.read_state is None:
            context.read_state = ReadFileState()
        if not isinstance(context.read_state, ReadFileState):
            raise IrisToolValidationError("read_state 类型无效")
        return context.read_state

    def _require_fresh_read(self, path: Path, context: ToolExecutionContext) -> None:
        """要求文件已读且 mtime/size 未变化。"""
        state = self._read_state(context)
        record = state.get(path)
        if record is None:
            raise IrisToolExecutionError("FILE_NOT_READ: 写入已有文件前必须先读取")
        stat = path.stat()
        if record.mtime_ns != stat.st_mtime_ns or record.size_bytes != stat.st_size:
            raise IrisToolExecutionError("STALE_FILE_STATE: 文件已在读取后发生变化")

    def _atomic_write(self, path: Path, content: str) -> None:
        """同目录临时文件原子替换写入。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
        ) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        temp_path.replace(path)

    def _iter_files_in_context(
        self,
        root: Path,
        context: ToolExecutionContext,
        pattern: str = "*",
    ) -> list[Path]:
        """列出 context workspace 内的普通文件，跳过逃逸符号链接。"""
        candidates = [root] if root.is_file() else sorted(root.rglob(pattern))
        files: list[Path] = []
        workspace_root = context.workspace_root.resolve()
        for candidate in candidates:
            resolved = candidate.resolve(strict=False)
            if not self.workspace_policy.is_within_workspace(resolved, workspace_root):
                continue
            if resolved.is_file():
                files.append(resolved)
        return files


class ReadFileTool(FileTool):
    """读取 workspace 文件。"""

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行读取。"""
        input_data = ReadFileInput.model_validate(params)
        path = self._resolve(input_data.file_path, context)
        if not path.exists():
            raise IrisToolExecutionError("FILE_NOT_FOUND: 文件不存在")
        if not path.is_file():
            raise IrisToolExecutionError("FILE_NOT_FOUND: 路径不是文件")
        offset = input_data.offset or 0
        limit = input_data.limit if input_data.limit is not None else 10000
        with path.open("r", encoding="utf-8") as handle:
            selected = [line.rstrip("\n") for line in islice(handle, offset, offset + limit)]
        content = "\n".join(
            f"{index}: {line}" for index, line in enumerate(selected, start=offset + 1)
        )
        self._read_state(context).update(path)
        return ToolResult(tool_use_id="", tool_name=self.name, content=[TextBlock(text=content)])


class ListFilesTool(FileTool):
    """列出 workspace 文件。"""

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行文件列出。"""
        input_data = ListFilesInput.model_validate(params)
        root = self._resolve(input_data.path, context)
        if not root.exists():
            raise IrisToolExecutionError("FILE_NOT_FOUND: 路径不存在")
        pattern = input_data.pattern or "*"
        paths = self._iter_files_in_context(root, context, pattern)
        if len(paths) > input_data.max_results:
            paths = paths[: input_data.max_results]
        workspace_root = context.workspace_root.resolve()
        content = "\n".join(str(path.resolve().relative_to(workspace_root)) for path in paths)
        return ToolResult(tool_use_id="", tool_name=self.name, content=[TextBlock(text=content)])


class GrepSearchTool(FileTool):
    """搜索 workspace 文本。"""

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行文本搜索。"""
        input_data = GrepSearchInput.model_validate(params)
        if input_data.max_results == 0:
            return ToolResult(tool_use_id="", tool_name=self.name, content=[])
        root = self._resolve(input_data.path, context)
        try:
            regex = re.compile(input_data.pattern)
        except re.error as exc:
            raise IrisToolValidationError(
                "invalid regex pattern",
                pattern=input_data.pattern,
            ) from exc
        matches: list[str] = []
        files = self._iter_files_in_context(root, context)
        workspace_root = context.workspace_root.resolve()
        for path in files:
            if ".iris" in path.parts:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if regex.search(line):
                    relative = path.resolve().relative_to(workspace_root)
                    matches.append(f"{relative}:{line_number}: {line}")
                    if len(matches) >= input_data.max_results:
                        break
            if len(matches) >= input_data.max_results:
                break
        return ToolResult(
            tool_use_id="",
            tool_name=self.name,
            content=[TextBlock(text="\n".join(matches))],
        )


class WriteFileTool(FileTool):
    """写入 workspace 文件。"""

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行写入。"""
        input_data = WriteFileInput.model_validate(params)
        path = self._resolve(input_data.file_path, context)
        if path.exists():
            self._require_fresh_read(path, context)
        self._atomic_write(path, input_data.content)
        self._read_state(context).update(path)
        return ToolResult(
            tool_use_id="",
            tool_name=self.name,
            content=[TextBlock(text=f"WROTE: {path}")],
        )


class EditFileTool(FileTool):
    """编辑 workspace 文件。"""

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行唯一字符串替换。"""
        input_data = EditFileInput.model_validate(params)
        path = self._resolve(input_data.file_path, context)
        if not path.exists():
            raise IrisToolExecutionError("FILE_NOT_FOUND: 文件不存在")
        self._require_fresh_read(path, context)
        content = path.read_text(encoding="utf-8")
        count = content.count(input_data.old_string)
        if count == 0:
            raise IrisToolExecutionError("MATCH_NOT_FOUND: 未找到 old_string")
        if count > 1:
            raise IrisToolExecutionError("AMBIGUOUS_MATCH: old_string 匹配多处")
        self._atomic_write(path, content.replace(input_data.old_string, input_data.new_string, 1))
        self._read_state(context).update(path)
        return ToolResult(
            tool_use_id="",
            tool_name=self.name,
            content=[TextBlock(text=f"EDITED: {path}")],
        )


def register_file_tools(*, max_result_chars: int = 50000) -> ToolRegistry:
    """创建并返回已注册文件工具的 registry。"""
    registry = ToolRegistry()
    for tool in (
        ReadFileTool(
            name="read_file",
            description="读取 workspace 内文本文件",
            input_model=ReadFileInput,
            capabilities={ToolCapability.READ},
            max_result_chars=max_result_chars,
        ),
        ListFilesTool(
            name="list_files",
            description="列出 workspace 内文件",
            input_model=ListFilesInput,
            capabilities={ToolCapability.READ},
            max_result_chars=max_result_chars,
        ),
        GrepSearchTool(
            name="grep_search",
            description="搜索 workspace 内文本文件",
            input_model=GrepSearchInput,
            capabilities={ToolCapability.READ},
            max_result_chars=max_result_chars,
        ),
        WriteFileTool(
            name="write_file",
            description="写入 workspace 内文本文件",
            input_model=WriteFileInput,
            capabilities={ToolCapability.WRITE},
            max_result_chars=max_result_chars,
        ),
        EditFileTool(
            name="edit_file",
            description="编辑 workspace 内已读取文本文件",
            input_model=EditFileInput,
            capabilities={ToolCapability.WRITE},
            max_result_chars=max_result_chars,
        ),
    ):
        registry.register(tool)
    return registry
