"""Workspace 文件工具。

本模块提供内置文件工具、共享文件服务和注册入口。工具类只负责 Iris 工具协议适配，
具体的路径解析、读取状态校验和文件写入规则由 :class:`WorkspaceFileService` 统一处理。

Example:
    registry = register_file_tools()
"""

# region imports
from __future__ import annotations

import asyncio
import os
import re
import tempfile
from abc import abstractmethod
from collections.abc import Callable, Iterator
from fnmatch import fnmatchcase
from itertools import islice
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel, Field, field_validator

from ...exceptions import IrisToolExecutionError, IrisToolValidationError
from ...message import TextBlock
from ..base import (
    BaseTool,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
)
from ..permissions import ReadFileRecord, ReadFileState, WorkspacePolicy
from ..registry import ToolRegistry
from ..schema import schema_from_pydantic_model

# endregion

InputT = TypeVar("InputT", bound=BaseModel)


def _matches_rglob_pattern(
    path_parts: tuple[str, ...],
    pattern_parts: tuple[str, ...],
) -> bool:
    """按 pathlib rglob 语义匹配路径段，包括 `**` 的零段匹配。

    Args:
        path_parts (tuple[str, ...]): 相对搜索根的路径段。
        pattern_parts (tuple[str, ...]): 已带隐式递归前缀的 glob pattern 段。

    Returns:
        bool: 路径是否满足递归 glob pattern。
    """
    # --- 1. 扩展零段状态 ---
    states = {0}

    def include_zero_segment_matches(current: set[int]) -> set[int]:
        expanded = set(current)
        for initial in current:
            position = initial
            while position < len(pattern_parts) and pattern_parts[position] == "**":
                position += 1
                expanded.add(position)
        return expanded

    # --- 2. 消费路径段 ---
    for part in path_parts:
        next_states: set[int] = set()
        for position in include_zero_segment_matches(states):
            if position == len(pattern_parts):
                continue
            pattern_part = pattern_parts[position]
            if pattern_part == "**":
                next_states.add(position)
            elif fnmatchcase(os.path.normcase(part), os.path.normcase(pattern_part)):
                next_states.add(position + 1)
        if not next_states:
            return False
        states = next_states

    # --- 3. 判断接受状态 ---
    return len(pattern_parts) in include_zero_segment_matches(states)


class ReadFileInput(BaseModel):
    """读取文件工具的输入模型。

    Attributes:
        file_path (str): 相对 workspace 根目录的文本文件路径。
        offset (int | None): 起始行偏移量，从 0 开始。默认为 None。
        limit (int | None): 最多读取的行数，最大 1000。默认为 None。
        with_line_numbers (bool): 是否返回带 `L0001 |` 前缀的行号视图。
    """

    file_path: str
    offset: int | None = Field(default=None, ge=0)
    limit: int | None = Field(default=None, ge=0, le=1000)
    with_line_numbers: bool = False


class ListFilesInput(BaseModel):
    """列出文件工具的输入模型。

    Attributes:
        path (str): 要列出的 workspace 内目录或文件路径。默认为当前目录。
        pattern (str | None): 传给 pathlib 的 glob 匹配模式。默认为 None。
        max_results (int): 最多返回的路径数量，范围为 0..1000。
    """

    path: str = "."
    pattern: str | None = None
    max_results: int = Field(default=200, ge=0, le=1000)


class GrepSearchInput(BaseModel):
    """文本搜索工具的输入模型。

    Attributes:
        pattern (str): Python 正则表达式搜索模式。
        path (str): 搜索起点，必须位于 workspace 内。默认为当前目录。
        max_results (int): 最多返回的匹配数量，范围为 0..1000。
    """

    pattern: str
    path: str = "."
    max_results: int = Field(default=200, ge=0, le=1000)


class WriteFileInput(BaseModel):
    """写入文件工具的输入模型。

    Attributes:
        file_path (str): 相对 workspace 根目录的目标文件路径。
        content (str): 要写入的完整文本内容。
    """

    file_path: str
    content: str


class EditFileInput(BaseModel):
    """编辑文件工具的输入模型。

    Attributes:
        file_path (str): 相对 workspace 根目录的目标文件路径。
        old_string (str): 要被替换的唯一原文片段。
        new_string (str): 替换后的文本片段。
    """

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


class WorkspaceFileService:
    """封装文件工具共享的 workspace 文件操作。

    服务层集中处理路径策略、读取状态和原子写入，避免每个工具类重复实现权限边界。

    Attributes:
        workspace_policy (WorkspacePolicy): 用于解析和限制 workspace 路径的策略。
    """

    # ==========================================
    #                    初始化
    # ==========================================
    # region
    def __init__(self, workspace_policy: WorkspacePolicy | None = None) -> None:
        """创建文件服务实例。

        Args:
            workspace_policy (WorkspacePolicy | None): 路径访问策略。为 None 时使用默认策略。
        """
        self.workspace_policy = workspace_policy or WorkspacePolicy()

    # endregion

    # ==========================================
    #                路径与读取状态
    # ==========================================
    # region
    def resolve_path(self, path: str, context: ToolExecutionContext) -> Path:
        """解析并校验 workspace 内路径。

        Args:
            path (str): 用户传入的相对或绝对路径。
            context (ToolExecutionContext): 当前工具执行上下文。

        Returns:
            Path: 已按 workspace 策略解析的路径。

        Raises:
            IrisToolValidationError: 当路径越出 workspace 或策略拒绝访问时。
        """
        return self.workspace_policy.resolve_path(path, workspace_root=context.workspace_root)

    def ensure_read_state(self, context: ToolExecutionContext) -> ReadFileState:
        """获取或初始化文件读取状态。

        Args:
            context (ToolExecutionContext): 当前工具执行上下文。

        Returns:
            ReadFileState: 可记录文件 mtime/size 的读取状态对象。

        Raises:
            IrisToolValidationError: 当上下文中已有不兼容的 read_state 时。
        """
        if context.read_state is None:
            context.read_state = ReadFileState()
        if not isinstance(context.read_state, ReadFileState):
            raise IrisToolValidationError("read_state 类型无效")
        return context.read_state

    def record_read(self, path: Path, context: ToolExecutionContext) -> None:
        """记录文件已读状态，供后续写入做乐观锁校验。

        Args:
            path (Path): 已读取的真实文件路径。
            context (ToolExecutionContext): 当前工具执行上下文。
        """
        self.ensure_read_state(context).update(path)

    def require_fresh_read(self, path: Path, context: ToolExecutionContext) -> None:
        """要求文件已读且 mtime/size 未变化。

        Args:
            path (Path): 即将被写入或编辑的真实文件路径。
            context (ToolExecutionContext): 当前工具执行上下文。

        Raises:
            IrisToolExecutionError: 当文件未读，或读取后又被外部修改时。
        """
        record = self.ensure_read_state(context).get(path)
        if record is None:
            raise IrisToolExecutionError("FILE_NOT_READ: 写入已有文件前必须先读取")
        stat = path.stat()
        if record.mtime_ns != stat.st_mtime_ns or record.size_bytes != stat.st_size:
            raise IrisToolExecutionError("STALE_FILE_STATE: 文件已在读取后发生变化")

    # endregion

    # ==========================================
    #                  文件操作
    # ==========================================
    # region
    def atomic_write(self, path: Path, content: str) -> None:
        """通过同目录临时文件执行原子替换写入。

        Args:
            path (Path): 目标文件路径。
            content (str): 要写入的完整文本内容。
        """
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

    def iter_files(
        self,
        root: Path,
        context: ToolExecutionContext,
        pattern: str = "*",
        *,
        limit: int | None = None,
        ignore_directory: Callable[[Path], bool] | None = None,
    ) -> Iterator[Path]:
        """按文件系统发现顺序迭代普通文件，并跳过逃逸符号链接。

        Args:
            root (Path): 搜索起点，可以是文件或目录。
            context (ToolExecutionContext): 当前工具执行上下文。
            pattern (str): 目录递归搜索使用的 glob 模式。默认为 "*"。
            limit (int | None): 最多产出的文件数。为 None 时不限制。
            ignore_directory (Callable[[Path], bool] | None): 下降前调用的目录过滤器。

        Yields:
            Path: 位于 workspace 内的真实普通文件路径。
        """
        if limit == 0:
            return

        workspace_root = context.workspace_root.resolve()
        if root.is_file():
            resolved = root.resolve(strict=False)
            if self.workspace_policy.is_within_workspace(resolved, workspace_root):
                yield resolved
            return

        if pattern.endswith(os.sep) or (os.altsep is not None and pattern.endswith(os.altsep)):
            return
        pattern_parts = ("**", *Path(pattern).parts)

        yielded = 0
        pending_directories = [root]
        while pending_directories:
            directory = pending_directories.pop()
            discovered_directories: list[Path] = []
            with os.scandir(directory) as entries:
                for entry in entries:
                    if limit is not None and yielded >= limit:
                        return
                    candidate = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        if ignore_directory is not None and ignore_directory(candidate):
                            continue
                        resolved_directory = candidate.resolve(strict=False)
                        if not self.workspace_policy.is_within_workspace(
                            resolved_directory,
                            workspace_root,
                        ):
                            continue
                        discovered_directories.append(candidate)
                        continue
                    if not entry.is_file(follow_symlinks=False) and not entry.is_symlink():
                        continue
                    if not _matches_rglob_pattern(
                        candidate.relative_to(root).parts,
                        pattern_parts,
                    ):
                        continue
                    resolved = candidate.resolve(strict=False)
                    if not self.workspace_policy.is_within_workspace(resolved, workspace_root):
                        continue
                    if resolved.is_file():
                        yielded += 1
                        yield resolved
                        if limit is not None and yielded >= limit:
                            return
            pending_directories.extend(reversed(discovered_directories))

    def read_file_observed(
        self,
        params: ReadFileInput,
        context: ToolExecutionContext,
    ) -> tuple[str, ReadFileRecord]:
        """读取文件片段并返回与已打开文件绑定的不可变观测。

        Args:
            params (ReadFileInput): 读取路径、分页和行号参数。
            context (ToolExecutionContext): 只提供 workspace 等不可变执行事实的 worker context。

        Returns:
            tuple[str, ReadFileRecord]: 文本片段及同一已打开文件的 mtime/size 观测。

        Raises:
            IrisToolExecutionError: 路径不存在或不是普通文件。
            IrisToolValidationError: 路径越出 workspace。
            OSError: 打开、读取或观测文件失败。
            UnicodeDecodeError: 文件不是有效 UTF-8 文本。
        """
        path = self.resolve_path(params.file_path, context)
        if not path.exists():
            raise IrisToolExecutionError("FILE_NOT_FOUND: 文件不存在")
        if not path.is_file():
            raise IrisToolExecutionError("FILE_NOT_FOUND: 路径不是文件")
        offset = params.offset or 0
        limit = params.limit if params.limit is not None else 1000
        with path.open("r", encoding="utf-8") as handle:
            selected = [line.rstrip("\n") for line in islice(handle, offset, offset + limit)]
            stat = os.fstat(handle.fileno())
        if params.with_line_numbers:
            content = "\n".join(
                f"L{index:04d} | {line}" for index, line in enumerate(selected, start=offset + 1)
            )
        else:
            content = "\n".join(selected)
        return (
            content,
            ReadFileRecord(
                path=path,
                mtime_ns=stat.st_mtime_ns,
                size_bytes=stat.st_size,
            ),
        )

    def read_file(self, params: ReadFileInput, context: ToolExecutionContext) -> str:
        """读取文件片段并更新读取状态。

        Args:
            params (ReadFileInput): 读取路径和分页参数。
            context (ToolExecutionContext): 当前工具执行上下文。

        Returns:
            str: 默认返回原始文本片段；请求行号时返回带 `L0001 |` 前缀的文本内容。

        Raises:
            IrisToolExecutionError: 当路径不存在或不是普通文件时。
        """
        content, record = self.read_file_observed(params, context)
        self.ensure_read_state(context).merge(record)
        return content

    def list_files(self, params: ListFilesInput, context: ToolExecutionContext) -> str:
        """列出 workspace 内文件。

        Args:
            params (ListFilesInput): 搜索起点、匹配模式和最大结果数。
            context (ToolExecutionContext): 当前工具执行上下文。

        Returns:
            str: 以换行分隔的 workspace 相对路径列表。

        Raises:
            IrisToolExecutionError: 当起点路径不存在时。
        """
        if params.max_results == 0:
            return ""

        root = self.resolve_path(params.path, context)
        if not root.exists():
            raise IrisToolExecutionError("FILE_NOT_FOUND: 路径不存在")
        paths = self.iter_files(
            root,
            context,
            params.pattern or "*",
            limit=params.max_results,
        )
        workspace_root = context.workspace_root.resolve()
        return "\n".join(str(path.relative_to(workspace_root)) for path in paths)

    def grep_search(self, params: GrepSearchInput, context: ToolExecutionContext) -> str:
        """搜索 workspace 内文本内容。

        Args:
            params (GrepSearchInput): 正则模式、搜索起点和最大结果数。
            context (ToolExecutionContext): 当前工具执行上下文。

        Returns:
            str: 以换行分隔的 `path:line: text` 匹配列表。

        Raises:
            IrisToolValidationError: 当正则表达式无效时。
        """
        if params.max_results == 0:
            return ""

        # --- 1. 校验正则 ---
        root = self.resolve_path(params.path, context)
        try:
            regex = re.compile(params.pattern)
        except re.error as exc:
            raise IrisToolValidationError("invalid regex pattern", pattern=params.pattern) from exc
        if not root.exists():
            return ""

        # --- 2. 扫描文本文件 ---
        matches: list[str] = []
        workspace_root = context.workspace_root.resolve()
        if ".iris" in root.relative_to(workspace_root).parts:
            return ""
        for path in self.iter_files(
            root,
            context,
            ignore_directory=lambda directory: directory.name == ".iris",
        ):
            with path.open("r", encoding="utf-8") as handle:
                try:
                    for line_number, line in enumerate(handle, start=1):
                        line = line.rstrip("\r\n")
                        if regex.search(line):
                            relative = path.relative_to(workspace_root)
                            matches.append(f"{relative}:{line_number}: {line}")
                            if len(matches) >= params.max_results:
                                return "\n".join(matches)
                except UnicodeDecodeError:
                    continue

        # --- 3. 返回匹配 ---
        return "\n".join(matches)

    def write_file(self, params: WriteFileInput, context: ToolExecutionContext) -> str:
        """写入新文件或覆盖已读且未变的已有文件。

        Args:
            params (WriteFileInput): 目标路径和完整文件内容。
            context (ToolExecutionContext): 当前工具执行上下文。

        Returns:
            str: 写入成功后的状态文本。

        Raises:
            IrisToolExecutionError: 当已有文件未读或读取后发生变化时。
        """
        path = self.resolve_path(params.file_path, context)
        if path.exists():
            self.require_fresh_read(path, context)
        self.atomic_write(path, params.content)
        self.record_read(path, context)
        return f"WROTE: {path.relative_to(context.workspace_root.resolve()).as_posix()}"

    def edit_file(self, params: EditFileInput, context: ToolExecutionContext) -> str:
        """对已读且未变的文件执行唯一字符串替换。

        Args:
            params (EditFileInput): 目标路径、唯一原文片段和替换文本。
            context (ToolExecutionContext): 当前工具执行上下文。

        Returns:
            str: 编辑成功后的状态文本。

        Raises:
            IrisToolExecutionError: 当文件不存在、读取状态过期或匹配文本不唯一时。
        """
        path = self.resolve_path(params.file_path, context)
        if not path.exists():
            raise IrisToolExecutionError("FILE_NOT_FOUND: 文件不存在")
        self.require_fresh_read(path, context)
        content = path.read_text(encoding="utf-8")
        count = content.count(params.old_string)
        if count == 0:
            raise IrisToolExecutionError("MATCH_NOT_FOUND: 未找到 old_string")
        if count > 1:
            raise IrisToolExecutionError("AMBIGUOUS_MATCH: old_string 匹配多处")
        self.atomic_write(path, content.replace(params.old_string, params.new_string, 1))
        self.record_read(path, context)
        return f"EDITED: {path.relative_to(context.workspace_root.resolve()).as_posix()}"

    # endregion


class FileTool(BaseTool, Generic[InputT]):  # noqa: UP046
    """Iris 文件工具协议适配基类。

    子类只声明工具元数据并实现业务方法，协议字段、输入校验和文本结果包装由基类统一处理。

    Attributes:
        name (ClassVar[str]): 工具注册名称。
        description (ClassVar[str]): 暴露给模型的工具说明。
        input_type (type[InputT]): Pydantic 输入模型类型。
        capabilities (ClassVar[set[ToolCapability]]): 工具读写能力声明。
        file_service (WorkspaceFileService): 共享文件服务实例。
        definition (ToolDefinition): Iris 工具定义。
    """

    # ==========================================
    #                    元数据
    # ==========================================
    # region
    name: ClassVar[str]
    description: ClassVar[str]
    input_type: type[InputT]
    capabilities: ClassVar[set[ToolCapability]]
    # endregion

    # ==========================================
    #                    初始化
    # ==========================================
    # region
    def __init__(
        self,
        *,
        file_service: WorkspaceFileService | None = None,
        max_result_chars: int = 50000,
    ) -> None:
        """创建文件工具实例。

        Args:
            file_service (WorkspaceFileService | None): 共享文件服务。为 None 时创建默认服务。
            max_result_chars (int): 单次工具结果允许返回给模型的最大字符数。
        """
        self.file_service = file_service or WorkspaceFileService()
        self.definition = ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=schema_from_pydantic_model(self.input_type),
            capabilities=self.capabilities,
            group="file",
            max_result_chars=max_result_chars,
        )

    # endregion

    # ==========================================
    #                  工具协议
    # ==========================================
    # region
    @property
    def input_model(self) -> type[BaseModel] | None:
        """返回用于协议层校验的 Pydantic 输入模型。

        Returns:
            type[BaseModel] | None: 当前工具的输入模型类型。
        """
        return self.input_type

    def validate_input(self, params: dict[str, Any]) -> BaseModel:
        """校验原始工具调用参数。

        Args:
            params (dict[str, Any]): 模型传入的原始参数字典。

        Returns:
            BaseModel: 已通过 Pydantic 校验的输入模型实例。
        """
        return self.input_type.model_validate(params)

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """适配工具协议并调用具体业务实现。

        Args:
            params (BaseModel | dict[str, Any]): 已解析或待解析的工具调用参数。
            context (ToolExecutionContext): 当前工具执行上下文。

        Returns:
            ToolResult: 可转换为模型消息的工具结果。
        """
        input_data = cast(InputT, params)
        return await self._impl(input_data, context)

    # endregion

    # ==========================================
    #                  实现辅助
    # ==========================================
    # region
    @abstractmethod
    async def _impl(self, params: InputT, context: ToolExecutionContext) -> ToolResult:
        """执行具体文件工具业务。

        Args:
            params (InputT): 已校验的输入模型。
            context (ToolExecutionContext): 当前工具执行上下文。

        Returns:
            ToolResult: 具体工具返回的协议结果。
        """
        raise NotImplementedError

    def _text_result(self, content: str) -> ToolResult:
        """构造文本工具结果。

        Args:
            content (str): 要返回给模型的文本内容。

        Returns:
            ToolResult: 包含单个文本块的工具结果；空文本会返回空内容列表。
        """
        return ToolResult(
            tool_use_id="",
            tool_name=self.name,
            content=[TextBlock(text=content)] if content else [],
        )

    # endregion


class ReadFileTool(FileTool[ReadFileInput]):
    """读取 workspace 文本文件的工具。"""

    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = "读取 workspace 内文本文件；需要定位或编辑时可请求行号"
    input_type: type[ReadFileInput] = ReadFileInput
    capabilities: ClassVar[set[ToolCapability]] = {ToolCapability.READ}

    async def _impl(
        self,
        params: ReadFileInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """调用文件服务读取文本片段。"""
        worker_context = context.model_copy(update={"read_state": None})
        content, record = await asyncio.to_thread(
            self.file_service.read_file_observed,
            params,
            worker_context,
        )
        self.file_service.ensure_read_state(context).merge(record)
        return self._text_result(content)


class ListFilesTool(FileTool[ListFilesInput]):
    """列出 workspace 文件路径的工具。"""

    name: ClassVar[str] = "list_files"
    description: ClassVar[str] = "列出 workspace 内文件"
    input_type: type[ListFilesInput] = ListFilesInput
    capabilities: ClassVar[set[ToolCapability]] = {ToolCapability.READ}

    async def _impl(
        self,
        params: ListFilesInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """调用文件服务列出 workspace 文件。"""
        worker_context = context.model_copy(update={"read_state": None})
        content = await asyncio.to_thread(self.file_service.list_files, params, worker_context)
        return self._text_result(content)


class GrepSearchTool(FileTool[GrepSearchInput]):
    """按正则搜索 workspace 文本内容的工具。"""

    name: ClassVar[str] = "grep_search"
    description: ClassVar[str] = "搜索 workspace 内文本文件"
    input_type: type[GrepSearchInput] = GrepSearchInput
    capabilities: ClassVar[set[ToolCapability]] = {ToolCapability.READ}

    async def _impl(
        self,
        params: GrepSearchInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """调用文件服务执行文本搜索。"""
        worker_context = context.model_copy(update={"read_state": None})
        content = await asyncio.to_thread(self.file_service.grep_search, params, worker_context)
        return self._text_result(content)


class WriteFileTool(FileTool[WriteFileInput]):
    """写入 workspace 文本文件的工具。"""

    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = "写入 workspace 内文本文件"
    input_type: type[WriteFileInput] = WriteFileInput
    capabilities: ClassVar[set[ToolCapability]] = {ToolCapability.WRITE}

    async def _impl(
        self,
        params: WriteFileInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """调用文件服务写入完整文本内容。"""
        return self._text_result(self.file_service.write_file(params, context))


class EditFileTool(FileTool[EditFileInput]):
    """编辑 workspace 已读取文本文件的工具。"""

    name: ClassVar[str] = "edit_file"
    description: ClassVar[str] = "编辑 workspace 内已读取文本文件"
    input_type: type[EditFileInput] = EditFileInput
    capabilities: ClassVar[set[ToolCapability]] = {ToolCapability.WRITE}

    async def _impl(
        self,
        params: EditFileInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """调用文件服务执行唯一字符串替换。"""
        return self._text_result(self.file_service.edit_file(params, context))


# ==========================================
#                工具类注册表
# ==========================================
# region constants
FILE_TOOL_CLASSES: tuple[type[FileTool[Any]], ...] = (
    ReadFileTool,
    ListFilesTool,
    GrepSearchTool,
    WriteFileTool,
    EditFileTool,
)
# endregion


def register_file_tools(
    *,
    registry: ToolRegistry | None = None,
    max_result_chars: int = 50000,
    file_service: WorkspaceFileService | None = None,
) -> ToolRegistry:
    """注册文件工具并返回 registry。

    Args:
        registry (ToolRegistry | None): 要扩展的已有 registry。为 None 时创建新 registry。
        max_result_chars (int): 每个文件工具允许返回给模型的最大字符数。
        file_service (WorkspaceFileService | None): 供所有文件工具共享的服务实例。
            为 None 时创建默认服务。

    Returns:
        ToolRegistry: 已按稳定顺序注册内置文件工具的 registry。
            如果传入了 `registry`，返回值就是同一个对象，便于和记忆工具等其它工具组合注册。
    """
    # 允许调用方把文件工具追加到已有 registry；未传入时保持独立注册入口的旧行为。
    registry = registry or ToolRegistry()
    service = file_service or WorkspaceFileService()
    for tool_cls in FILE_TOOL_CLASSES:
        registry.register(tool_cls(file_service=service, max_result_chars=max_result_chars))
    return registry
