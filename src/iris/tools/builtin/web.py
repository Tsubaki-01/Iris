"""通过 Tavily basic 搜索来源和提取网页正文的内置工具。"""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
    ValidationError,
)

from ...exceptions import IrisToolExecutionError, IrisToolValidationError
from ...message import TextBlock
from ..base import BaseTool, ToolCapability, ToolDefinition, ToolExecutionContext, ToolResult
from ..schema import schema_from_pydantic_model
from ._tavily import post_tavily

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
HttpUrl = Annotated[AnyHttpUrl, PlainSerializer(str, return_type=str)]


class WebSearchInput(BaseModel):
    """搜索来源的模型参数；深度固定为 basic。"""

    model_config = ConfigDict(extra="forbid")

    query: NonEmptyText = Field(description="搜索查询。")
    max_results: int = Field(default=10, ge=1, le=20, strict=True, description="最多返回的来源数。")
    time_range: Literal["day", "week", "month", "year"] | None = Field(
        default=None, description="按 Tavily 的来源日期语义限定时间范围。"
    )
    include_domains: list[str] = Field(
        default_factory=list, max_length=300, description="仅搜索这些域名。"
    )
    exclude_domains: list[str] = Field(
        default_factory=list, max_length=150, description="排除这些域名。"
    )


class WebFetchInput(BaseModel):
    """批量提取正文或按 query 选取摘录的模型参数。"""

    model_config = ConfigDict(extra="forbid")

    urls: list[HttpUrl] = Field(min_length=1, max_length=20, description="要提取的 HTTP(S) URL。")
    query: NonEmptyText | None = Field(
        default=None,
        description="可选摘录查询，对整批 URL 生效；省略时返回服务提取的完整正文。",
    )


class _SearchResult(BaseModel):
    """搜索结果中实际消费的来源字段。"""

    title: str
    url: str
    content: str


class _SearchResponse(BaseModel):
    """搜索接口的外部响应。"""

    results: list[_SearchResult]


class _ExtractResult(BaseModel):
    """提取成功的 URL 与正文，不假设服务返回标题。"""

    url: str
    raw_content: str


class _ExtractFailure(BaseModel):
    """提取失败的 URL 与服务原因。"""

    url: str
    error: str


class _ExtractResponse(BaseModel):
    """提取接口的成功与失败结果。"""

    results: list[_ExtractResult]
    failed_results: list[_ExtractFailure]


class WebSearchTool(BaseTool):
    """获取搜索来源供主模型判断，不生成答案或追加检索。"""

    def __init__(self, *, api_key: str) -> None:
        """接收由配置构造器或 SDK host 提供的 Tavily 凭据。"""
        self._api_key = api_key
        self.definition = ToolDefinition(
            name="web_search",
            description="搜索网页来源，返回标题、URL 和相关片段；可用 web_fetch 读取正文。",
            input_schema=schema_from_pydantic_model(WebSearchInput),
            capabilities={ToolCapability.NETWORK},
            group="web",
            context_retention="observation",
        )

    @property
    def input_model(self) -> type[BaseModel]:
        """返回用于 schema 和输入边界的唯一模型。"""
        return WebSearchInput

    def validate_input(self, params: dict[str, Any]) -> WebSearchInput:
        """在原始工具参数首次进入时解析输入。"""
        try:
            return WebSearchInput.model_validate(params)
        except ValidationError as exc:
            raise IrisToolValidationError("web_search 参数校验失败", errors=exc.errors()) from exc

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        """执行一次 basic 搜索并按服务顺序呈现完整来源片段。"""
        inputs = cast(WebSearchInput, params)
        payload: dict[str, Any] = {
            "query": inputs.query,
            "max_results": inputs.max_results,
            "search_depth": "basic",
            "topic": "general",
            "include_answer": False,
            "include_raw_content": False,
            "auto_parameters": False,
        }
        if inputs.time_range is not None:
            payload["time_range"] = inputs.time_range
        if inputs.include_domains:
            payload["include_domains"] = inputs.include_domains
            payload["include_domains_mode"] = "filter"
        if inputs.exclude_domains:
            payload["exclude_domains"] = inputs.exclude_domains
        response = await post_tavily(
            endpoint="search",
            api_key=self._api_key,
            payload=payload,
            response_model=_SearchResponse,
        )
        sections = [f"# Web Search\n\nQuery: {inputs.query}\nResults: {len(response.results)}"]
        sections.extend(
            f"## {index}. {result.title}\nURL: {result.url}\n\n{result.content}"
            for index, result in enumerate(response.results, start=1)
        )
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=self.name,
            content=[TextBlock(text="\n\n".join(sections))],
        )


class WebFetchTool(BaseTool):
    """使用 Tavily Extract basic 获取批量正文或定向摘录。"""

    def __init__(self, *, api_key: str) -> None:
        """接收由配置构造器或 SDK host 提供的 Tavily 凭据。"""
        self._api_key = api_key
        self.definition = ToolDefinition(
            name="web_fetch",
            description=(
                "批量读取 URL：省略 query 获取服务提取的完整正文，"
                "提供 query 获取相关摘录；结果保留来源 URL 及各页面失败原因。"
            ),
            input_schema=schema_from_pydantic_model(WebFetchInput),
            capabilities={ToolCapability.NETWORK},
            group="web",
            context_retention="observation",
        )

    @property
    def input_model(self) -> type[BaseModel]:
        """返回用于 schema 和输入边界的唯一模型。"""
        return WebFetchInput

    def validate_input(self, params: dict[str, Any]) -> WebFetchInput:
        """在原始工具参数首次进入时解析输入。"""
        try:
            return WebFetchInput.model_validate(params)
        except ValidationError as exc:
            raise IrisToolValidationError("web_fetch 参数校验失败", errors=exc.errors()) from exc

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        """执行一次批量提取，保留全部成功正文并前置失败摘要。"""
        inputs = cast(WebFetchInput, params)
        payload: dict[str, Any] = {
            "urls": [str(url) for url in inputs.urls],
            "extract_depth": "basic",
            "format": "markdown",
        }
        mode = "full_content"
        if inputs.query is not None:
            payload["query"] = inputs.query
            mode = "excerpts"
        response = await post_tavily(
            endpoint="extract",
            api_key=self._api_key,
            payload=payload,
            response_model=_ExtractResponse,
        )
        summary = ["# Web Fetch", "", f"Mode: {mode}"]
        if inputs.query is not None:
            summary.append(f"Query: {inputs.query}")
        summary.extend(
            [f"Succeeded: {len(response.results)}", f"Failed: {len(response.failed_results)}"]
        )
        sections = ["\n".join(summary)]
        if response.failed_results:
            failures = "\n".join(
                f"- {failure.url}\n  Reason: {failure.error}" for failure in response.failed_results
            )
            sections.append(f"## Failed URLs\n\n{failures}")
        if not response.results:
            raise IrisToolExecutionError(
                "Tavily Extract 未返回正文。\n\n" + "\n\n".join(sections), tool_name=self.name
            )
        sections.extend(
            f"## Content: {result.url}\n\n{result.raw_content}" for result in response.results
        )
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=self.name,
            content=[TextBlock(text="\n\n".join(sections))],
        )


__all__ = ["WebFetchInput", "WebFetchTool", "WebSearchInput", "WebSearchTool"]
