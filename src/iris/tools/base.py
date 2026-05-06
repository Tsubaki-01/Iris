"""工具内核基础模型与 callable 适配器。"""

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
    """工具暴露给 registry 和 provider schema 的定义。"""

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

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """校验 provider 可接受的工具名。"""
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
        """校验工具描述非空。"""
        if not value.strip():
            raise ValueError("工具描述不能为空")
        return value

    @field_validator("input_schema")
    @classmethod
    def _validate_input_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        """校验最小 JSON Schema object 结构。"""
        if value.get("type") != "object":
            raise ValueError("工具输入 schema 必须是 object")
        value.setdefault("properties", {})
        value.setdefault("required", [])
        return value


class ToolExecutionContext(BaseModel):
    """一次工具执行的上下文。"""

    call_id: str = ""
    tool_name: str = ""
    workspace_root: Path
    session_id: str = ""
    agent_id: str = ""
    permission_mode: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)
    read_state: Any | None = None


class ToolErrorInfo(BaseModel):
    """结构化工具错误。"""

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("code", "message")
    @classmethod
    def _validate_not_empty(cls, value: str) -> str:
        """校验错误码和错误信息非空。"""
        if not value:
            raise ValueError("工具错误信息不能为空")
        return value


class ToolArtifact(BaseModel):
    """工具产物引用。"""

    path: Path
    mime_type: str = "text/plain"
    size_bytes: int = 0
    preview: str = ""


class ToolResult(BaseModel):
    """Iris 内部工具执行结果。"""

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
        """返回可回灌给模型的文本内容。"""
        if self.is_error and self.error is not None:
            return f"Error[{self.error.code}]: {self.error.message}"
        return "\n".join(block.text for block in self.content)

    def to_block_metadata(self) -> dict[str, Any]:
        """生成 `ToolResultBlock.metadata` 的标准子集。"""
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
    """所有工具实现的统一接口。"""

    definition: ToolDefinition

    @property
    def name(self) -> str:
        """返回工具名。"""
        return self.definition.name

    def input_model(self) -> type[BaseModel] | None:
        """返回显式输入模型。"""
        return None

    def input_schema(self) -> dict[str, Any]:
        """返回工具输入 JSON Schema。"""
        return self.definition.input_schema

    def validate_input(self, params: dict[str, Any]) -> BaseModel | dict[str, Any]:
        """校验工具输入参数。"""
        return params

    def is_read_only(self, params: dict[str, Any]) -> bool:
        """判断工具是否只读。"""
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
        """判断工具是否具有破坏性能力。"""
        del params
        return bool(self.definition.capabilities & {ToolCapability.WRITE, ToolCapability.EXECUTE})

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        """判断工具是否可并发执行。"""
        del params
        return True

    @abstractmethod
    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """异步执行工具。"""
        raise NotImplementedError


class CallableTool(BaseTool):
    """将普通 Python callable 适配为工具。"""

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
        """初始化 callable 工具。"""
        self.func = func
        self._input_model = input_model
        self.preset_kwargs = dict(preset_kwargs or {})
        tool_name = name or getattr(func, "iris_tool_name", None) or func.__name__
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

    def input_model(self) -> type[BaseModel] | None:
        """返回 callable 的输入模型。"""
        return self._generated_model

    def validate_input(self, params: dict[str, Any]) -> BaseModel | dict[str, Any]:
        """使用 Pydantic 模型校验 callable 输入。"""
        overlap = set(params) & set(self.preset_kwargs)
        if overlap:
            raise IrisToolValidationError("工具参数不能覆盖预设参数", params=sorted(overlap))
        validation_params = {**params, **self.preset_kwargs}
        try:
            return self._generated_model.model_validate(validation_params)
        except ValidationError as exc:
            raise IrisToolValidationError("工具参数校验失败", errors=exc.errors()) from exc

    async def arun(
        self,
        params: BaseModel | dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行 callable 并归一化返回值。"""
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
        """将 callable 返回值归一化为 `ToolResult`。"""
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
