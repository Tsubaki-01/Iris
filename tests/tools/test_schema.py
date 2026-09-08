"""工具导出的 schema 与首次输入校验、实际函数参数保持一致。"""

from pathlib import Path
from typing import Annotated

import pytest
from pydantic import BaseModel, ConfigDict, Field

from iris.exceptions import IrisToolValidationError
from iris.message import ToolUseBlock
from iris.tools import CallableTool, ToolExecutionContext, ToolExecutor, ToolRegistry


class SearchQuery(BaseModel):
    """具有字段约束的结构化搜索参数。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


def test_fixed_tuple_schema_describes_each_position() -> None:
    """异构 tuple 的 schema 同时描述固定长度与各位置类型。"""

    def locate(pair: tuple[str, int]) -> str:
        """定位目标。

        Args:
            pair: 名称和序号。
        """
        return f"{pair[0]}:{pair[1]}"

    tool = CallableTool(locate)
    schema = tool.input_schema["properties"]["pair"]

    assert schema["prefixItems"] == [{"type": "string"}, {"type": "integer"}]
    assert schema["minItems"] == schema["maxItems"] == 2
    assert schema["description"] == "名称和序号。"
    assert tool.input_schema["required"] == ["pair"]


@pytest.mark.asyncio
async def test_annotated_optional_and_presets_match_actual_execution(tmp_path: Path) -> None:
    """保留 Annotated 约束、可选默认值与隐藏 preset 的实际执行语义。"""
    observed: list[tuple[int, str | None, str]] = []

    def search(
        limit: Annotated[int, Field(gt=0, le=5)],
        secret: str,
        query: str | None = None,
    ) -> str:
        """搜索内容。

        Args:
            limit: 最多返回的条目数。
            secret: 预设凭证。
            query: 可省略的查询。
        """
        observed.append((limit, query, secret))
        return "ok"

    registry = ToolRegistry()
    tool = registry.register_function(search, preset_kwargs={"secret": "token"})
    schema = tool.input_schema
    assert schema["properties"]["limit"]["exclusiveMinimum"] == 0
    assert schema["properties"]["limit"]["maximum"] == 5
    assert schema["properties"]["limit"]["description"] == "最多返回的条目数。"
    assert schema["properties"]["query"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    assert schema["properties"]["query"]["default"] is None
    assert schema["required"] == ["limit"]
    assert "secret" not in schema["properties"]

    executor = ToolExecutor(registry)
    context = ToolExecutionContext(workspace_root=tmp_path)
    for index, arguments in enumerate([{"limit": 2}, {"limit": 3, "query": None}]):
        result = await executor.execute_one(
            ToolUseBlock(id=str(index), name="search", input=arguments), context
        )
        assert not result.is_error
    for arguments in [{"limit": 0}, {"limit": 6}, {"limit": 2, "secret": "override"}]:
        result = await executor.execute_one(
            ToolUseBlock(id="invalid", name="search", input=arguments), context
        )
        assert result.error is not None
        assert result.error.code == "VALIDATION_ERROR"
    assert observed == [(2, None, "token"), (3, None, "token")]


@pytest.mark.asyncio
async def test_typed_tuple_and_nested_model_reach_callable(tmp_path: Path) -> None:
    """已验证字段保持 Python 类型，嵌套模型不在调用前重新变成字典。"""

    def search(pair: tuple[str, int], query: SearchQuery) -> str:
        return f"{pair!r}:{query.text}"

    registry = ToolRegistry()
    registry.register_function(search)
    executor = ToolExecutor(registry)
    context = ToolExecutionContext(workspace_root=tmp_path)
    result = await executor.execute_one(
        ToolUseBlock(
            id="valid", name="search", input={"pair": ["iris", 2], "query": {"text": "hi"}}
        ),
        context,
    )
    assert not result.is_error
    assert result.model_content == "('iris', 2):hi"
    for pair in [["iris", "wrong"], ["iris"], ["iris", 2, 3]]:
        result = await executor.execute_one(
            ToolUseBlock(
                id="invalid", name="search", input={"pair": pair, "query": {"text": "hi"}}
            ),
            context,
        )
        assert result.error is not None
        assert result.error.code == "VALIDATION_ERROR"


def test_explicit_model_keeps_root_schema_constraints() -> None:
    """显式输入模型拒绝额外字段的契约同时出现在导出 schema 中。"""

    def search(text: str) -> str:
        return text

    tool = CallableTool(search, input_model=SearchQuery)
    assert tool.input_schema["additionalProperties"] is False
    with pytest.raises(IrisToolValidationError):
        tool.validate_input({"text": "hi", "extra": True})


def test_callable_rejects_unsupported_signature() -> None:
    """位置专属参数和缺失注解仍在工具注册边界被拒绝。"""

    def positional(value: str, /) -> str:
        return value

    with pytest.raises(IrisToolValidationError):
        CallableTool(positional)

    def missing(value: str) -> str:
        return value

    missing.__annotations__.pop("value")
    with pytest.raises(IrisToolValidationError):
        CallableTool(missing)
