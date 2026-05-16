"""Workspace 文件工具。

提供用于文件操作的工具集合。

Example:
    registry = register_file_tools(max_result_chars=50000)
"""

# region imports
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

# endregion


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
        if value is not None and value > 1000:
            raise ValueError("limit 不能超过 1000")
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
        if value < 0 or value > 1000:
            raise ValueError("max_results 必须在 0..1000 范围内")
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
        if value < 0 or value > 1000:
            raise ValueError("max_results 必须在 0..1000 范围内")
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
    """文件工具基类。

    提供核心的文件抽象，处理路径解析并保证文件状态一致。

    Attributes:
        workspace_policy (WorkspacePolicy): 约束工作区读取等权限的策略机制。
        definition (ToolDefinition): 暴漏给大语言模型的工具使用模式数据。

    Example:
        class WriteTool(FileTool): ...
    """

    # ==========================================
    #               Initialization
    # ==========================================
    # region
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[BaseModel],
        capabilities: set[ToolCapability],
        max_result_chars: int,
    ) -> None:
        """初始化文件工具定义。

        设置统一属性以及权限策略控制系统，建立对外部系统透明操作的基础。

        Args:
            name (str): 工具调用名称。
            description (str): 描述该文件工具用途的文字。
            input_model (type[BaseModel]): 规定执行此工具调用的输入形状结构。
            capabilities (set[ToolCapability]): 提供该工具包含的能力边界。
            max_result_chars (int): 输出结果允许的最大长度。
        """
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

    # endregion

    # ==========================================
    #               Public Methods
    # ==========================================
    # region
    @property
    def input_model(self) -> type[BaseModel] | None:
        """返回输入模型。"""
        return self._input_model

    def validate_input(self, params: dict[str, Any]) -> BaseModel:
        """校验输入参数。"""
        return self._input_model.model_validate(params)

    # endregion

    # ==========================================
    #               Helper Methods
    # ==========================================
    # region
    def _resolve(self, file_path: str, context: ToolExecutionContext) -> Path:
        """解析 workspace 内合规路径。"""
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

    # endregion


class ReadFileTool(FileTool):
    """读取 workspace 文件。

    暴露文件内容的实际读取逻辑，并保持读取到上下文中，作未来编辑使用。

    Example:
        tool = ReadFileTool(name="read_file", description="读文件", input_model=ReadFileInput,
                            capabilities={ToolCapability.READ}, max_result_chars=1)
    """

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行读取。

        完成分页与文本拼接提取。

        Args:
            params (BaseModel | dict[str, Any]): 输入的数据或 Pydantic 模型封装态。
            context (ToolExecutionContext): 包括工作区的基础执行运行文环境封装态。

        Returns:
            ToolResult: 带有 TextBlock 返回包装的安全输出标准物。

        Raises:
            IrisToolExecutionError: 当文件不存在、路径非文件时拦截返回错误提示。

        Example:
            res = await self.arun({"file_path": "a.py"}, ctx)
        """
        # --- 1. 加载并检查参数 ---
        input_data = ReadFileInput.model_validate(params)
        path = self._resolve(input_data.file_path, context)
        if not path.exists():
            raise IrisToolExecutionError("FILE_NOT_FOUND: 文件不存在")
        if not path.is_file():
            raise IrisToolExecutionError("FILE_NOT_FOUND: 路径不是文件")

        # --- 2. 提取分页内容 ---
        offset = input_data.offset or 0
        limit = input_data.limit if input_data.limit is not None else 1000
        with path.open("r", encoding="utf-8") as handle:
            selected = [line.rstrip("\n") for line in islice(handle, offset, offset + limit)]
        content = "\n".join(
            f"{index}: {line}" for index, line in enumerate(selected, start=offset + 1)
        )

        # --- 3. 同步状态并返回 ---
        self._read_state(context).update(path)
        return ToolResult(tool_use_id="", tool_name=self.name, content=[TextBlock(text=content)])


class ListFilesTool(FileTool):
    """列出 workspace 文件。

    为代理提供探测未明工程整体与局部的架构感知。

    Example:
        tool = ListFilesTool(name="list_files", description="检索列表", input_model=ListFilesInput,
                            capabilities={ToolCapability.READ}, max_result_chars=1)
    """

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行文件列出。

        应用模式匹配获取限定集合下的展示级输出串列表集。

        Args:
            params (BaseModel | dict[str, Any]): 被提供的结构化目录读取限制及匹配要求设定。
            context (ToolExecutionContext): 工具工作区所属运行宿主上下文态。

        Returns:
            ToolResult: 结果带文件列表字符序列态模型。

        Raises:
            IrisToolExecutionError: 对于要求查询的指定节点未找到抛出中断性提示。

        Example:
            r = await self.arun({"path": "."}, ctx)
        """
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
    """搜索 workspace 文本。

    利用正则表达式在全部普通文件中跨文件找寻内容，并排查非法逃逸文件。

    Example:
        tool = GrepSearchTool(name="grep_search", description="搜索", input_model=GrepSearchInput,
                            capabilities={ToolCapability.READ}, max_result_chars=1)
    """

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行文本搜索。

        进行受控遍历并且过滤内置配置、不可读物并产生带有文件、行列信息的匹配。

        Args:
            params (BaseModel | dict[str, Any]): 必须带有正则表达式且验证限流要求的定义集模型。
            context (ToolExecutionContext): 工作环境态实体模型。

        Returns:
            ToolResult: 带有所有被命中并序列化输出至极限之内的 TextBlock。

        Raises:
            IrisToolValidationError: 对于用户端书写有错误的非法的正则表达式直接拦截抛出以规避执行错误。

        Example:
            r = await self.arun({"pattern": "^import "}, ctx)
        """
        # --- 1. 验证输入并提前返回 ---
        input_data = GrepSearchInput.model_validate(params)
        if input_data.max_results == 0:
            return ToolResult(tool_use_id="", tool_name=self.name, content=[])

        # --- 2. 编译模式并列出文件树 ---
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

        # --- 3. 在有效的文本文件中处理匹配项 ---
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

        # --- 4. 返回序列化的匹配结果 ---
        return ToolResult(
            tool_use_id="",
            tool_name=self.name,
            content=[TextBlock(text="\n".join(matches))],
        )


class WriteFileTool(FileTool):
    """写入 workspace 文件。

    支撑由零建新或由历史产生安全强制状态验证后的一次性覆盖全量重写能力工具实现类。

    Example:
        tool = WriteFileTool(name="write_file", description="写入", input_model=WriteFileInput,
                            capabilities={ToolCapability.WRITE}, max_result_chars=1)
    """

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行写入。

        以保证系统并发与覆写不出问题的方式调用其底层原子的改动替换落盘处理机制操作接口实现层。

        Args:
            params (BaseModel | dict[str, Any]): 包含内容与指向的数据类参数。
            context (ToolExecutionContext): 带有上下文以及防脏读要求的历史表执行环实例。

        Returns:
            ToolResult: 只宣告完毕路径的标志性操作提示包裹对象。

        Raises:
            IrisToolExecutionError: 当强制依赖检测底层异常如读版本时会从下层溢出异常。

        Example:
            r = await self.arun({"file_path": "a.txt", "content": "123"}, ctx)
        """
        input_data = WriteFileInput.model_validate(params)
        path = self._resolve(input_data.file_path, context)
        if path.exists():
            self._require_fresh_read(path, context)  # Must have been read safely.
        self._atomic_write(path, input_data.content)
        self._read_state(context).update(path)
        return ToolResult(
            tool_use_id="",
            tool_name=self.name,
            content=[TextBlock(text=f"WROTE: {path}")],
        )


class EditFileTool(FileTool):
    """编辑 workspace 文件。

    运用明确并独一的一个代码段替换形式规避对于结构不可名状产生的大范围替换的不可测现象保障操作原子化。

    Example:
        tool = EditFileTool(name="edit_file", description="替换", input_model=EditFileInput,
                            capabilities={ToolCapability.WRITE}, max_result_chars=1)
    """

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行唯一字符串替换。

        核对多重态确保所指代的只允许单一无二的存在方可进行原地的文本切片修改换源后刷新到真实盘面。

        Args:
            params (BaseModel | dict[str, Any]): 老文字源并携带新修替目标的描述包表。
            context (ToolExecutionContext): 必须存有被验证历史的操作文背景实体。

        Returns:
            ToolResult: 返回执行被改变后的通知标识。

        Raises:
            IrisToolExecutionError: 当发现不存在匹配或有多重同源时产生不可逾越错误以示用户。

        Example:
            r = await self.arun({"file_path": "a.txt", "old_string": "x", "new_string": "y"}, ctx)
        """
        # --- 1. 加载并读取当前状态 ---
        input_data = EditFileInput.model_validate(params)
        path = self._resolve(input_data.file_path, context)
        if not path.exists():
            raise IrisToolExecutionError("FILE_NOT_FOUND: 文件不存在")
        self._require_fresh_read(path, context)
        content = path.read_text(encoding="utf-8")

        # --- 2. 检查严格的唯一边界 ---
        count = content.count(input_data.old_string)
        if count == 0:
            raise IrisToolExecutionError("MATCH_NOT_FOUND: 未找到 old_string")
        if count > 1:
            raise IrisToolExecutionError("AMBIGUOUS_MATCH: old_string 匹配多处")

        # --- 3. 原子化替换并返回 ---
        self._atomic_write(path, content.replace(input_data.old_string, input_data.new_string, 1))
        self._read_state(context).update(path)
        return ToolResult(
            tool_use_id="",
            tool_name=self.name,
            content=[TextBlock(text=f"EDITED: {path}")],
        )


def register_file_tools(*, max_result_chars: int = 50000) -> ToolRegistry:
    """创建并返回已注册文件工具的 registry。

    用以集成装配各种子类型成统一对外的组。

    Args:
        max_result_chars (int): 指定工具内建对最大返文流字符量的标准长度。 Defaults to 50000.

    Returns:
        ToolRegistry: 全部实例化的工具注册清单包裹对象化存在体系。

    Example:
        reg = register_file_tools(max_result_chars=100)
    """
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
