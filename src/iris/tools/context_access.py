"""当前会话上下文的只读工具与宿主读取协议。"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import IrisToolExecutionError
from ..message import TextBlock
from ._io import run_tool_io
from .base import (
    BaseTool,
    ToolCapability,
    ToolDefinition,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolResult,
)
from .schema import schema_from_pydantic_model


class ContextReadInput(BaseModel):
    """读取已提交消息或单个结果的有界字符页。"""

    ref: str = Field(pattern=r"^(?:message:[0-9]+|result:[0-9]+:[0-9]+)$")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=4000, ge=1, le=8000)
    representation: Literal["text", "raw"] = Field(
        default="text",
        description=(
            "回读历史原文通常使用text：获取当时保存的完整最终模型文本。"
            "raw仅用于已有artifact的result引用，读取原生文件（如MCP原始JSON）；"
            "普通内联结果和message引用不支持raw。"
        ),
    )
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextSearchInput(BaseModel):
    """在当前会话的已保存正文与预览内查找子串。"""

    query: str = Field(pattern=r"\S")
    after: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=20)
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True, slots=True)
class ContextReadPage:
    """正文分页结果；offset 均为 Python Unicode 字符位置。"""

    ref: str
    representation: str
    offset: int
    next_offset: int
    has_more: bool
    content: str


@dataclass(frozen=True, slots=True)
class ContextSearchHit:
    """一条消息中的首个命中与稳定原文位置。"""

    ref: str
    role: str
    tool_name: str
    snippet: str


@dataclass(frozen=True, slots=True)
class ContextSearchPage:
    """有限扫描的命中及后续扫描游标。"""

    matches: tuple[ContextSearchHit, ...]
    next_after: int
    has_more: bool


class ContextAccessPort(Protocol):
    """宿主提供当前 session 的原文读取；工具不持有 lifecycle store。"""

    def read(
        self, session_id: str, params: ContextReadInput, workspace_root: Path
    ) -> ContextReadPage:
        """读取一个原文字符页。"""
        ...

    def search(self, session_id: str, params: ContextSearchInput) -> ContextSearchPage:
        """扫描一页已提交历史，不读取 artifact 正文。"""
        ...


class ContextReadTool(BaseTool):
    """根据模型视图给出的稳定 ref 读取当时保存的材料。"""

    def __init__(self, access: ContextAccessPort) -> None:
        """绑定宿主读取协议，并声明只读工具 schema。"""
        self.access = access
        self.definition = ToolDefinition(
            name="context_read",
            description=(
                "分页读取当前会话的 message:<index> 或 result:<message_index>:<block_index>；"
                "默认text读取当时保存的完整最终文本，包括内联或外置结果；"
                "raw仅适用于已有artifact的result引用。不会重新执行工具。"
            ),
            input_schema=schema_from_pydantic_model(ContextReadInput),
            capabilities={ToolCapability.READ},
            group="context",
            max_result_chars=12000,
        )

    @property
    def input_model(self) -> type[BaseModel]:
        """返回首次工具输入校验模型。"""
        return ContextReadInput

    def validate_input(self, params: dict[str, Any]) -> BaseModel:
        """在 LLM 输入边界解析参数。"""
        return ContextReadInput.model_validate(params)

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        """将完整的一次本地读取放入同一个 worker。"""
        try:
            page = await run_tool_io(
                lambda: self.access.read(
                    context.session_id, cast(ContextReadInput, params), context.workspace_root
                )
            )
        except IrisToolExecutionError as exc:
            return _read_error(context, exc)
        header = (
            f"{page.ref} {page.representation} offset={page.offset} "
            f"next_offset={page.next_offset} has_more={str(page.has_more).lower()}\n"
        )
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=self.name,
            content=[TextBlock(text=header + page.content)],
            data=asdict(page),
        )


class ContextSearchTool(BaseTool):
    """查找当前会话已归档正文及预览，不扫描外置原文。"""

    def __init__(self, access: ContextAccessPort) -> None:
        """绑定宿主搜索协议，并声明有限扫描工具 schema。"""
        self.access = access
        self.definition = ToolDefinition(
            name="context_search",
            description=(
                "在当前会话已提交正文和工具预览中以 Unicode casefold 子串搜索；"
                "不搜索 artifact 完整正文。每次最多扫描200条消息，"
                "无命中且has_more为true时继续next_after；用context_read展开命中。"
            ),
            input_schema=schema_from_pydantic_model(ContextSearchInput),
            capabilities={ToolCapability.READ},
            group="context",
            max_result_chars=12000,
        )

    @property
    def input_model(self) -> type[BaseModel]:
        """返回首次工具输入校验模型。"""
        return ContextSearchInput

    def validate_input(self, params: dict[str, Any]) -> BaseModel:
        """在 LLM 输入边界解析参数。"""
        return ContextSearchInput.model_validate(params)

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        """一次 worker 中完成有限历史扫描。"""
        page = await run_tool_io(
            lambda: self.access.search(context.session_id, cast(ContextSearchInput, params))
        )
        text = f"next_after={page.next_after} has_more={str(page.has_more).lower()}\n"
        text += "\n".join(
            f"{hit.ref} {hit.role} {hit.tool_name}\n{hit.snippet}" for hit in page.matches
        )
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=self.name,
            content=[TextBlock(text=text)],
            data=asdict(page),
        )


def _read_error(context: ToolExecutionContext, error: IrisToolExecutionError) -> ToolResult:
    return ToolResult(
        tool_use_id=context.call_id,
        tool_name=context.tool_name,
        is_error=True,
        error=ToolErrorInfo(code=str(error.context["code"]), message=error.message),
    )
