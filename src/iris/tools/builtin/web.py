"""通过 Tavily 或 DeepSeek 搜索来源和提取网页正文的内置工具。"""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
    UrlConstraints,
    ValidationError,
    field_validator,
)

from ...exceptions import IrisToolExecutionError, IrisToolValidationError
from ...message import TextBlock
from ..base import BaseTool, ToolCapability, ToolDefinition, ToolExecutionContext, ToolResult
from ..schema import schema_from_pydantic_model
from ._web_http import WebProvider, post_web

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


class DeepSeekWebSearchInput(BaseModel):
    """DeepSeek 搜索参数，过滤条件直接写入原生 query。"""

    model_config = ConfigDict(extra="forbid")

    query: NonEmptyText = Field(
        max_length=128,
        description=(
            "1–128 个字符的查询；支持 site:、-site:、after:、before:、lang: 和引号精确匹配。"
            "多个包含站点用 OR 连接，例如 Python site:docs.python.org OR site:peps.python.org。"
        ),
    )
    max_results: int = Field(default=10, ge=1, le=50, strict=True, description="最多返回的来源数。")


class DeepSeekWebFetchInput(BaseModel):
    """DeepSeek 单页 Markdown 抓取参数。"""

    model_config = ConfigDict(extra="forbid")

    url: Annotated[HttpUrl, UrlConstraints(max_length=8192)] = Field(
        description="要读取的单个 HTTP(S) URL，最多 8192 字符，不能包含用户名或密码。"
    )

    @field_validator("url")
    @classmethod
    def _reject_url_credentials(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        """在输入边界落实 DeepSeek 的 URL 请求格式约束。"""
        if value.username is not None or value.password is not None:
            raise ValueError("DeepSeek URL 不能包含用户名或密码")
        return value


class _SearchResult(BaseModel):
    """搜索结果中实际消费的来源字段。"""

    title: str
    url: str
    content: str


class _SearchResponse(BaseModel):
    """搜索接口的外部响应。"""

    results: list[_SearchResult]


class _DeepSeekSearchResult(BaseModel):
    """DeepSeek 搜索结果中面向模型的来源字段。"""

    title: str
    url: str
    summary: str


class _DeepSeekSearchResponse(BaseModel):
    """DeepSeek Search 的外部响应。"""

    results: list[_DeepSeekSearchResult]


class _DeepSeekFetchResponse(BaseModel):
    """DeepSeek Fetch 的页面标题与 Markdown 正文。"""

    title: str
    content: str


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

    def __init__(self, *, api_key: str, provider: WebProvider = "tavily") -> None:
        """接收服务选择及对应凭据，构造时固定模型可见的参数。"""
        self._api_key = api_key
        self.provider = provider
        self._input_model = {
            "tavily": WebSearchInput,
            "deepseek": DeepSeekWebSearchInput,
        }[provider]
        self.definition = ToolDefinition(
            name="web_search",
            description="搜索网页来源，返回标题、URL 和相关片段；可用 web_fetch 读取正文。",
            input_schema=schema_from_pydantic_model(self.input_model),
            capabilities={ToolCapability.NETWORK},
            group="web",
            context_retention="observation",
        )

    @property
    def input_model(self) -> type[BaseModel]:
        """返回用于 schema 和输入边界的唯一模型。"""
        return self._input_model

    def validate_input(self, params: dict[str, Any]) -> WebSearchInput | DeepSeekWebSearchInput:
        """在原始工具参数首次进入时解析输入。"""
        try:
            return self._input_model.model_validate(params)
        except ValidationError as exc:
            raise IrisToolValidationError("web_search 参数校验失败", errors=exc.errors()) from exc

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        """执行一次搜索并按服务顺序呈现供模型阅读的来源片段。"""
        inputs = cast(WebSearchInput | DeepSeekWebSearchInput, params)
        if self.provider == "deepseek":
            response = await post_web(
                provider=self.provider,
                endpoint="search",
                api_key=self._api_key,
                payload={
                    "query": inputs.query,
                    "count": inputs.max_results,
                    "need_content": False,
                },
                response_model=_DeepSeekSearchResponse,
            )
            sources = [(item.title, item.url, item.summary) for item in response.results]
        else:
            sources = await self._search_tavily(cast(WebSearchInput, inputs))
        sections = [f"# Web Search\n\nQuery: {inputs.query}\nResults: {len(sources)}"]
        sections.extend(
            f"## {index}. {title}\nURL: {url}\n\n{content}"
            for index, (title, url, content) in enumerate(sources, start=1)
        )
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=self.name,
            content=[TextBlock(text="\n\n".join(sections))],
        )

    async def _search_tavily(self, inputs: WebSearchInput) -> list[tuple[str, str, str]]:
        """保留 Tavily basic 的筛选参数与来源片段。"""
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
        response = await post_web(
            provider=self.provider,
            endpoint="search",
            api_key=self._api_key,
            payload=payload,
            response_model=_SearchResponse,
        )
        return [(item.title, item.url, item.content) for item in response.results]


class WebFetchTool(BaseTool):
    """使用所选服务获取网页正文，Tavily 额外支持批量与定向摘录。"""

    def __init__(self, *, api_key: str, provider: WebProvider = "tavily") -> None:
        """接收服务选择及对应凭据，构造时固定模型可见的参数。"""
        self._api_key = api_key
        self.provider = provider
        self._input_model = {"tavily": WebFetchInput, "deepseek": DeepSeekWebFetchInput}[provider]
        self.definition = ToolDefinition(
            name="web_fetch",
            description=(
                "读取单个 URL 的网页标题与完整 Markdown 正文，保留来源 URL。"
                if provider == "deepseek"
                else "批量读取 URL：省略 query 获取服务提取的完整正文，"
                "提供 query 获取相关摘录；结果保留来源 URL 及各页面失败原因。"
            ),
            input_schema=schema_from_pydantic_model(self.input_model),
            capabilities={ToolCapability.NETWORK},
            group="web",
            context_retention="observation",
        )

    @property
    def input_model(self) -> type[BaseModel]:
        """返回用于 schema 和输入边界的唯一模型。"""
        return self._input_model

    def validate_input(self, params: dict[str, Any]) -> WebFetchInput | DeepSeekWebFetchInput:
        """在原始工具参数首次进入时解析输入。"""
        try:
            return self._input_model.model_validate(params)
        except ValidationError as exc:
            raise IrisToolValidationError("web_fetch 参数校验失败", errors=exc.errors()) from exc

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        """按服务契约提取正文，保留来源与可用的失败信息。"""
        if self.provider == "deepseek":
            url = str(cast(DeepSeekWebFetchInput, params).url)
            page = await post_web(
                provider=self.provider,
                endpoint="fetch",
                api_key=self._api_key,
                payload={"url": url},
                response_model=_DeepSeekFetchResponse,
            )
            return ToolResult(
                tool_use_id=context.call_id,
                tool_name=self.name,
                content=[
                    TextBlock(
                        text=f"# Web Fetch\n\nURL: {url}\nTitle: {page.title}\n\n{page.content}"
                    )
                ],
            )
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
        response = await post_web(
            provider=self.provider,
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


__all__ = [
    "DeepSeekWebFetchInput",
    "DeepSeekWebSearchInput",
    "WebFetchInput",
    "WebFetchTool",
    "WebProvider",
    "WebSearchInput",
    "WebSearchTool",
]
