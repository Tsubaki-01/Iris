"""工具内核基础模型与 callable 适配器。

定义工具抽象层和标准交互协议，使得底层业务逻辑能与外层模型隔离。

Example:
    tool = CallableTool(my_func)
    print(tool.name)
"""

# region imports
from __future__ import annotations

import inspect
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..exceptions import IrisToolExecutionError, IrisToolValidationError
from ..message import TextBlock
from .schema import callable_input_model, schema_from_callable, schema_from_pydantic_model

# endregion


class ToolCapability(StrEnum):
    """工具能力标签。"""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    MCP = "mcp"
    AGENT = "agent"


class ToolExecutionMode(StrEnum):
    """工具执行模式。"""

    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"


class ToolDefinition(BaseModel):
    """工具暴露给 registry 和 provider schema 的定义。

    统一了所有工具的对外元数据，在 LLM 发现和挂载时作为标准格式读取。

    Attributes:
        name (str): 暴露给 LLM 的工具名称。
        description (str): 提供给 LLM 的功能描述和参数说明。
        input_schema (dict[str, Any]): OpenAPI 格式的 json schema，规范入参。
        capabilities (set[ToolCapability]): 安全隔离级别标识标签。
        group (str): 工具分类，通常用于批量注册或过滤。
        aliases (tuple[str, ...]): 可能的同义名称。
        deferred (bool): 标识是否延迟计算或初始化。
        max_result_chars (int): 执行结果文本的最长限制，防止输出撑爆上下文。
        preview_chars (int): 输出超长截断时提供给人类审查的最大字数。
        metadata (dict[str, Any]): 存放其他拓展属性。

    Example:
        defn = ToolDefinition(name="ls", description="List files", input_schema={"type": "object"})
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    capabilities: set[ToolCapability] = Field(default_factory=set)
    group: str = "core"
    aliases: tuple[str, ...] = ()
    deferred: bool = False
    max_result_chars: int = 50000
    preview_chars: int = 8000
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ==========================================
    #               Validation Methods
    # ==========================================
    # region
    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """校验 provider 可接受的工具名。

        保证名称格式符合各大服务商（如 OpenAI/Anthropic）的模型约束，防止接口调用时拒绝。

        Args:
            value (str): 用户提供的原始工具名称。

        Returns:
            str: 校验通过后的合法名称。

        Raises:
            ValueError: 名称为空、包含非法字符或长度超限时报错。

        Example:
            name = _validate_name("hello_world")
        """
        if not value:
            raise ValueError("工具名不能为空")
        if len(value) > 64:
            raise ValueError("工具名不能超过 64 个字符")
        if not (value[0].isalpha() or value[0] == "_"):
            raise ValueError("工具名必须以字母或下划线开头")
        if not all(char.isalnum() or char == "_" for char in value):
            raise ValueError("工具名只能包含字母、数字和下划线")
        return value

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        """校验工具描述非空。

        大模型极度依赖清晰的描述来判断是否调用本工具，因此强制要求有内容。

        Args:
            value (str): 注入的文本描述。

        Returns:
            str: 通过后的描述本身。

        Raises:
            ValueError: 若全是空格或为空则直接阻断实例化。

        Example:
            doc = _validate_description("Fetch user data")
        """
        if not value.strip():
            raise ValueError("工具描述不能为空")
        return value

    @field_validator("input_schema")
    @classmethod
    def _validate_input_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        """校验最小 JSON Schema object 结构。

        确保生成的协议符合 JSON Schema 标准约定。必须为 object，必须带必要的根节点。

        Args:
            value (dict[str, Any]): 自动提取或手写的 schema 树。

        Returns:
            dict[str, Any]: 填补缺失了 properties 等节点后的健壮结构。

        Raises:
            ValueError: 若根节点类型不是 object 时引发。

        Example:
            schema = _validate_input_schema({"type": "object"})
        """
        if value.get("type") != "object":
            raise ValueError("工具输入 schema 必须是 object")
        value.setdefault("properties", {})
        value.setdefault("required", [])
        return value

    # endregion


class ToolExecutionContext(BaseModel):
    """一次工具执行的上下文。

    在整个生命周期内记录调用的追踪凭证与授权状态，方便透传。

    Attributes:
        call_id (str): Provider 下发的该次调用跨请求链路唯一标识。
        tool_name (str): 当前引发执行的工具引用名字。
        workspace_root (Path): 提供给底层寻找本地存储等参照的路径根基。
        session_id (str): 全局长连接会话标识。
        agent_id (str): 发起调用的智能体身份 ID。
        permission_mode (str): 操作受限时的静默降级或拦截等级许可。
        metadata (dict[str, Any]): 附加运行态透传的配置或临时钩子容器字典。
        read_state (Any | None): 模型长效调用的工具游标恢复引用对象。

    Example:
        ctx = ToolExecutionContext(workspace_root=Path("."))
    """

    call_id: str = ""
    tool_name: str = ""
    workspace_root: Path
    session_id: str = ""
    agent_id: str = ""
    permission_mode: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)
    read_state: Any | None = None


class ToolErrorInfo(BaseModel):
    """结构化工具错误。

    统一包装内部报错后展示给 LLM，令其有机会在感知异常后执行自修正及重试。

    Attributes:
        code (str): 错误特征常量串，用于标识失败的大致类型。
        message (str): 给模型及用户的更详尽具体错误原由解释。
        retryable (bool): 代表当前工具失败是否可通过修正输入后二次挽回。
        details (dict[str, Any]): 更底层的 traceback 之类的额外上下文。

    Example:
        err = ToolErrorInfo(code="404", message="File not found")
    """

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    # ==========================================
    #               Validation Methods
    # ==========================================
    # region
    @field_validator("code", "message")
    @classmethod
    def _validate_not_empty(cls, value: str) -> str:
        """校验错误码和错误信息非空。

        大模型依赖必须存在的描述以便判断错位点，不接受空白结构。

        Args:
            value (str): 需要赋值被检查字段的字符串本体。

        Returns:
            str: 判定合法之后的值对象自身。

        Raises:
            ValueError: 发现字符串没有实际阅读意义或空置时产生。

        Example:
            val = _validate_not_empty("timeout")
        """
        if not value:
            raise ValueError("工具错误信息不能为空")
        return value

    # endregion


class ToolArtifact(BaseModel):
    """工具产物引用。

    用于工具向调用端传回极其长的数据或多媒体（如图片，PDF）时的高级非文本占位替身。

    Attributes:
        path (Path): 本地产物存储写盘目标的绝对路径。
        mime_type (str): HTTP 体系下资源的标准网络文件表示。
        size_bytes (int): 生成资源的总占用尺寸。
        preview (str): 用于防止阻塞 LLM 但又令其了解大概的精简版信息。

    Example:
        art = ToolArtifact(path=Path("/tmp/a.png"))
    """

    path: Path
    mime_type: str = "text/plain"
    size_bytes: int = 0
    preview: str = ""


class ToolResult(BaseModel):
    """Iris 内部工具执行结果。

    携带业务函数的原始回应并转化为标准的大模式结果结构，作为工具调用的唯一下游出口承载体。

    Attributes:
        tool_use_id (str): 与此结果匹配的工具调用回执 ID。
        tool_name (str): 引发此结果的工具映射名。
        content (list[TextBlock]): 向模型展示和传输的实际文本块。
        is_error (bool): 是否执行发生错误且模型应当感知。
        error (ToolErrorInfo | None): 被格式化后的异常。
        data (dict[str, Any]): 透明传递被抽取出的纯数据副本。
        artifact (ToolArtifact | None): 二进制类文件与图像等产物指针。
        stats (dict[str, Any]): 运行时性能耗时和统计信息。
        metadata (dict[str, Any]): 与框架其他组件（如 Trace）的挂载点。

    Example:
        res = ToolResult(tool_use_id="123", tool_name="ls", content=[...])
    """

    tool_use_id: str
    tool_name: str
    content: list[TextBlock] = Field(default_factory=list)
    is_error: bool = False
    error: ToolErrorInfo | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    artifact: ToolArtifact | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_content(self) -> str:
        """返回可回灌给模型的文本内容。

        提供给不支持结构化复杂回传内容的老旧通道一个纯字符串。

        Returns:
            str: 格式为错误码或逐行合并的多 Block 纯文本内容。

        Example:
            txt_response = result.model_content()
        """
        if self.is_error and self.error is not None:
            return f"Error[{self.error.code}]: {self.error.message}"
        return "\n".join(block.text for block in self.content)

    def to_block_metadata(self) -> dict[str, Any]:
        """生成 ToolResultBlock.metadata 的标准子集。

        截取非侵入式的信息点透传到底层 Message 时避免混入内部未序列化对象。

        Returns:
            dict[str, Any]: 仅保留了追踪与调试相关干净状态的字典。

        Example:
            meta = result.to_block_metadata()
        """
        metadata: dict[str, Any] = {}
        if self.error is not None:
            metadata["error"] = self.error.model_dump()
        if self.stats:
            metadata["stats"] = self.stats
        if self.artifact is not None:
            metadata["artifact"] = self.artifact.model_dump()
        for key in ("permission", "trace_id", "extra"):
            if key in self.metadata:
                metadata[key] = self.metadata[key]
        if self.tool_name:
            metadata["tool_name"] = self.tool_name
        return metadata


class BaseTool(ABC):
    """所有工具实现的统一接口。

    提供工具描述、输入参数校验和异步执行的标准契约，任何具体工具都必须继承此规范。

    Attributes:
        definition (ToolDefinition): 描述工具的元数据及提供给 Provider 的 schema 信息。

    Example:
        class WebSearchTool(BaseTool):
            ...
    """

    definition: ToolDefinition

    # ==========================================
    #               Public API Methods
    # ==========================================
    # region
    @property
    def name(self) -> str:
        """返回工具名。

        用于在执行日志和与模型交互时标识此工具。

        Returns:
            str: 工具名，从 definition 中获取。

        Example:
            >>> tool.name
            'my_tool'
        """
        return self.definition.name

    def input_model(self) -> type[BaseModel] | None:
        """返回显式输入模型。

        如果工具明确定义了 Pydantic 模型作为输入，则返回此模型，否则为 None。

        Returns:
            type[BaseModel] | None: 可用于验证的对象类，或当未指定时返回 None。

        Example:
            model = tool.input_model()
        """
        return None

    def input_schema(self) -> dict[str, Any]:
        """返回工具输入 JSON Schema。

        将其作为 Provider 调用工具的规范协议约束。

        Returns:
            dict[str, Any]: 符合 JSON Schema 格式的字典。

        Example:
            schema = tool.input_schema()
        """
        return self.definition.input_schema

    # ==========================================
    #               Validation & Checks
    # ==========================================
    # region
    def validate_input(self, params: dict[str, Any]) -> BaseModel | dict[str, Any]:
        """校验工具输入参数。

        用于在实际执行代码前捕获不合规的输入，以提前熔断。

        Args:
            params (dict[str, Any]): 模型生成的原始请求参数字典。

        Returns:
            BaseModel | dict[str, Any]: 返回校验后的模型对象或字典。

        Raises:
            IrisToolValidationError: 当参数校验不通过时。

        Example:
            valid_params = tool.validate_input({"arg": 1})
        """
        return params

    def is_read_only(self, params: dict[str, Any]) -> bool:
        """判断工具是否只读。

        基于能力标签集合 (capabilities) 进行评估，帮助安全模块决定是否放行。

        Args:
            params (dict[str, Any]): 即将传递给工具的参数数据。

        Returns:
            bool: 没有任何写入和执行标签时返回 True，否则返回 False。

        Example:
            if tool.is_read_only({}): pass
        """
        del params
        write_capabilities = {
            ToolCapability.WRITE,
            ToolCapability.EXECUTE,
            ToolCapability.NETWORK,
            ToolCapability.MCP,
            ToolCapability.AGENT,
        }
        return not bool(self.definition.capabilities & write_capabilities)

    def is_destructive(self, params: dict[str, Any]) -> bool:
        """判断工具是否具有破坏性能力。

        用于执行高保真模式控制下的二次用户确认拦截。

        Args:
            params (dict[str, Any]): 工具调用参数。

        Returns:
            bool: True 表示可能对环境有破坏。

        Example:
            has_risk = tool.is_destructive(args)
        """
        del params
        return bool(self.definition.capabilities & {ToolCapability.WRITE, ToolCapability.EXECUTE})

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        """判断工具是否可并发执行。

        用于工具执行器决定该工具是否可以同批次一起被分发处理。

        Args:
            params (dict[str, Any]): 工具调用参数。

        Returns:
            bool: 默认为 True 可以并发执行。

        Example:
            can_run_together = tool.is_concurrency_safe(args)
        """
        del params
        return True

    # ==========================================
    #               Execution Target
    # ==========================================
    # region
    @abstractmethod
    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """异步执行工具。

        由子类实现具体业务。所有最终输出都要被包装进 ToolResult 标准响应中，提供给外部。

        Args:
            params (BaseModel | dict[str, Any]): 经过验证的入参。
            context (ToolExecutionContext): 此次运行周期的执行上下文。

        Returns:
            ToolResult: 工具执行的最终响应包裹。

        Raises:
            NotImplementedError: 子类未覆写时触发。

        Example:
            result = await tool.arun(args, ctx)
        """
        raise NotImplementedError

    # endregion


class CallableTool(BaseTool):
    """将普通 Python callable 适配为工具。

    支持通过装饰器快速将任意带类型注解的函数映射为大模型所需的高级工具结构。

    Attributes:
        func (Callable): 原始处理函数。
        preset_kwargs (dict): 预绑定的关键字参数值。

    Example:
        tool = CallableTool(my_func, name="my")
    """

    # ==========================================
    #               Initialization
    # ==========================================
    # region
    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        input_model: type[BaseModel] | None = None,
        preset_kwargs: dict[str, Any] | None = None,
        capabilities: set[ToolCapability] | None = None,
        group: str = "core",
        deferred: bool = False,
    ) -> None:
        """初始化 callable 工具。

        读取函数属性和注解推断出工具定义，用于自动绑定模型映射和 schema。

        Args:
            func (Callable[..., Any]): 被包装的基础函数。
            name (str | None): 可选自定义工具名称。
            description (str | None): 可选自定义描述，默认为 docstring。
            input_model (type[BaseModel] | None): 传入预定义的 pydantic model 校验类。
            preset_kwargs (dict[str, Any] | None): 这些参数不在 schema 暴露。
            capabilities (set[ToolCapability] | None): 设置工具对应的底层能力集。
            group (str): 属于的工具组分类。
            deferred (bool): 是否为延迟执行工具。

        Raises:
            ValueError: 解析模型约束异常时抛出。

        Example:
            tool = CallableTool(fetch_data)
        """
        self.func = func
        self._input_model = input_model
        self.preset_kwargs = dict(preset_kwargs or {})
        tool_name = name or getattr(func, "iris_tool_name", None) or func.__name__  # ty:ignore[unresolved-attribute]
        tool_description = (
            description
            or getattr(func, "iris_tool_description", None)
            or inspect.getdoc(func)
            or tool_name
        )
        tool_capabilities = capabilities or getattr(func, "iris_tool_capabilities", set())
        tool_group = getattr(func, "iris_tool_group", group) if group == "core" else group
        tool_deferred = deferred or bool(getattr(func, "iris_tool_deferred", False))
        preset_names = set(self.preset_kwargs)
        input_schema = (
            schema_from_pydantic_model(input_model)
            if input_model is not None
            else schema_from_callable(func, preset_kwargs=preset_names)
        )
        for name_ in preset_names:
            input_schema.get("properties", {}).pop(name_, None)
            if name_ in input_schema.get("required", []):
                input_schema["required"].remove(name_)
        self._generated_model = (
            input_model if input_model is not None else callable_input_model(func, preset_names)
        )
        self.definition = ToolDefinition(
            name=tool_name,
            description=tool_description,
            input_schema=input_schema,
            capabilities=set(tool_capabilities),
            group=tool_group,
            deferred=tool_deferred,
        )

    # endregion
    # ==========================================
    #               Validation Methods
    # ==========================================
    # region
    def input_model(self) -> type[BaseModel] | None:
        """返回 callable 的输入模型。

        将动态生成或传入的 Pydantic 数据结构作为统一的模型输出形式。

        Returns:
            type[BaseModel] | None: 对应的参数校验模型类。

        Example:
            model = tool.input_model()
        """
        return self._generated_model

    def validate_input(self, params: dict[str, Any]) -> BaseModel | dict[str, Any]:
        """使用 Pydantic 模型校验 callable 输入。

        确保模型参数加上预设参数可以满足原始目标函数所有的参数需求限制，提前规避运行时错误。

        Args:
            params (dict[str, Any]): 模型决定使用的请求参数字典。

        Returns:
            BaseModel | dict[str, Any]: 校验并清洗好的数据对象。

        Raises:
            IrisToolValidationError: 当提供的参数和预设参数冲突或不符标准。

        Example:
            validated = tool.validate_input({"url": "http://.."})
        """
        overlap = set(params) & set(self.preset_kwargs)
        if overlap:
            raise IrisToolValidationError("工具参数不能覆盖预设参数", params=sorted(overlap))
        validation_params = {**params, **self.preset_kwargs}
        try:
            return self._generated_model.model_validate(validation_params)
        except ValidationError as exc:
            raise IrisToolValidationError("工具参数校验失败", errors=exc.errors()) from exc

    # endregion

    # ==========================================
    #               Core Execution
    # ==========================================
    # region
    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行 callable 并归一化返回值。

        接收处理数据、负责拦截运行报错并将任意格式的数据平滑转换为文本模型可接纳的形式。

        Args:
            params (BaseModel | dict[str, Any]): 已通过验证的请求数据体。
            context (ToolExecutionContext): 环境与运行状态上下文。

        Returns:
            ToolResult: 带有调用指标和元数据的执行结果。

        Raises:
            IrisToolExecutionError: 当 callable 在执行阶段发生预期外异常时。
            IrisToolValidationError: 若发生延迟验证错误时冒泡。

        Example:
            res = await tool.arun(args, ctx)
        """
        del context
        kwargs = params.model_dump() if isinstance(params, BaseModel) else dict(params)
        kwargs.update(self.preset_kwargs)
        start = time.perf_counter()
        try:
            value = self.func(**kwargs)
            if inspect.isawaitable(value):
                value = await value
        except IrisToolValidationError:
            raise
        except Exception as exc:
            raise IrisToolExecutionError(str(exc), tool_name=self.name) from exc
        result = self._normalize_result(value)
        result.stats.setdefault("elapsed_ms", round((time.perf_counter() - start) * 1000, 3))
        return result

    def _normalize_result(self, value: Any) -> ToolResult:
        """将 callable 返回值归一化为 ToolResult。

        支持直接返回 ToolResult，也支持将字符串或 JSON 取值包装为 TextBlock。

        Args:
            value (Any): 原函数调用的返回结果值。

        Returns:
            ToolResult: 统一标准的对外结构。

        Example:
            res = tool._normalize_result({"status": "ok"})
        """
        if isinstance(value, ToolResult):
            return value
        if isinstance(value, str):
            text = value
        elif value is None:
            text = ""
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            except TypeError:
                text = str(value)
        return ToolResult(
            tool_use_id="",
            tool_name=self.name,
            content=[TextBlock(text=text)] if text else [],
        )

    # endregion
