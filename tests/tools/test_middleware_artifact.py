"""工具最终结果在全部 middleware 后统一生成 artifact。"""

import json
import re
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest
from pydantic import BaseModel

from iris.message import TextBlock, ToolUseBlock
from iris.tools import (
    BaseTool,
    PermissionDecision,
    PermissionEffect,
    PermissionPolicy,
    ToolDefinition,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolRegistry,
    ToolResult,
)


@pytest.mark.asyncio
async def test_after_call_expansion_is_persisted_once(tmp_path: Path) -> None:
    """正常扩展结果的 hook 也遵守工具声明的长度阈值。"""

    class ExpandResult(ToolMiddleware):
        """为结果补充描述。"""

        async def after_call(
            self, tool: BaseTool, result: ToolResult, context: ToolExecutionContext
        ) -> ToolResult:
            """补充正文，保持工具 identity。"""
            return result.model_copy(update={"content": [TextBlock(text="expanded result")]})

    class SmallTool(BaseTool):
        """声明短结果阈值的正常工具。"""

        definition = ToolDefinition(
            name="small",
            description="短结果",
            input_schema={"type": "object", "properties": {}},
            max_result_chars=3,
            preview_chars=2,
        )

        async def arun(
            self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
        ) -> ToolResult:
            """返回阈值内的原始结果。"""
            return ToolResult(
                tool_use_id=context.call_id, tool_name=self.name, content=[TextBlock(text="ok")]
            )

    registry = ToolRegistry()
    registry.register(SmallTool())
    result = await ToolExecutor(registry, middleware=[ExpandResult()]).execute_one(
        ToolUseBlock(id="call", name="small", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert not result.is_error
    assert result.artifact is not None
    assert result.artifact.path.read_text(encoding="utf-8") == "expanded result"
    assert result.artifact.preview == "ex"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_id", "second_id"),
    [("call/1", "call_1"), ("Call", "call"), ("中文", "__"), ("", "default")],
)
@pytest.mark.parametrize("id_field", ["call", "session"])
async def test_different_ids_keep_both_artifacts(
    tmp_path: Path, first_id: str, second_id: str, id_field: str
) -> None:
    """不同调用或会话 ID 的大结果同时可读，路径仍位于 artifact 根目录。"""

    def echo(value: str) -> str:
        return value

    registry = ToolRegistry()
    tool = registry.register_function(echo)
    tool.definition.max_result_chars = 3
    executor = ToolExecutor(registry)
    artifacts = []
    for identifier, content in [(first_id, "first result"), (second_id, "second result")]:
        result = await executor.execute_one(
            ToolUseBlock(
                id=identifier if id_field == "call" else "same-call",
                name="echo",
                input={"value": content},
            ),
            ToolExecutionContext(
                workspace_root=tmp_path,
                session_id=identifier if id_field == "session" else "same-session",
            ),
        )
        assert not result.is_error
        assert result.artifact is not None
        result.artifact.path.relative_to(tmp_path / ".iris" / "tool-results")
        artifacts.append(result.artifact)

    assert artifacts[0].path != artifacts[1].path
    assert artifacts[0].path.read_text(encoding="utf-8") == "first result"
    assert artifacts[1].path.read_text(encoding="utf-8") == "second result"


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_final_result_respects_budget_and_keeps_full_output(
    tmp_path: Path, failed: bool
) -> None:
    """普通成功与错误结果使用包含提示和错误前缀的最终预算。"""
    full_text = "结果" * 5000

    def produce() -> ToolResult:
        """返回需要落盘的成功或错误正文。"""
        return ToolResult(
            tool_use_id="",
            tool_name="produce",
            content=[TextBlock(text=full_text)],
            is_error=failed,
            error=ToolErrorInfo(code="FAILED", message=full_text) if failed else None,
        )

    registry = ToolRegistry()
    tool = registry.register_function(produce)
    tool.definition.max_result_chars = 1000
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="bounded", name="produce", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert len(result.model_content) <= 1000
    assert result.is_error is failed
    assert result.artifact is not None
    assert str(result.artifact.path) in result.model_content
    assert result.artifact.path.read_text(encoding="utf-8") == (
        f"Error[FAILED]: {full_text}" if failed else full_text
    )
    if failed:
        assert result.error is not None and result.error.code == "FAILED"


@pytest.mark.asyncio
async def test_each_tool_owns_its_preview_length(tmp_path: Path) -> None:
    """普通工具各自声明的 preview_chars 在最终投影中生效。"""
    registry = ToolRegistry()

    def produce() -> str:
        """返回长文本。"""
        return "x" * 5000

    for name, preview_chars in [("short_preview", 5), ("long_preview", 31)]:
        tool = registry.register_function(produce, name=name)
        tool.definition.max_result_chars = 1000
        tool.definition.preview_chars = preview_chars
        result = await ToolExecutor(registry).execute_one(
            ToolUseBlock(id=name, name=name, input={}),
            ToolExecutionContext(workspace_root=tmp_path),
        )
        assert result.artifact is not None
        assert result.artifact.preview == "x" * preview_chars
        assert result.model_content.startswith("x" * preview_chars + "\n\n[")


@pytest.mark.asyncio
async def test_raised_error_is_bounded_and_preserved(tmp_path: Path) -> None:
    """工具抛出的长异常也经过最终保存，避免错误出口绕过预算。"""

    def fail() -> str:
        """模拟普通业务异常。"""
        raise RuntimeError("failure" * 2000)

    registry = ToolRegistry()
    tool = registry.register_function(fail)
    tool.definition.max_result_chars = 1000
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="failed", name="fail", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert len(result.model_content) <= 1000
    assert result.error is not None and result.error.code == "EXECUTION_ERROR"
    assert result.artifact is not None
    assert "failure" * 2000 in result.artifact.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_preflight_error_is_bounded_without_file_effect(tmp_path: Path) -> None:
    """预检拒绝只裁剪说明，不能为了保存错误写入 artifact。"""

    class Deny(PermissionPolicy):
        """提供长拒绝说明。"""

        def check(
            self, tool: BaseTool, params: dict[str, Any], context: ToolExecutionContext
        ) -> PermissionDecision:
            """在工具执行前拒绝。"""
            return PermissionDecision(effect=PermissionEffect.DENY, reason="denied" * 2000)

    def forbidden() -> str:
        """不应执行的工具。"""
        raise AssertionError("tool body must not execute")

    registry = ToolRegistry()
    tool = registry.register_function(forbidden)
    tool.definition.max_result_chars = 1000
    executor = ToolExecutor(registry, permission_policy=Deny())
    context = ToolExecutionContext(workspace_root=tmp_path)
    call = ToolUseBlock(id="denied", name="forbidden", input={})
    prepared = executor.prepare_many([call], context).calls[0]
    assert prepared.preflight_result is not None
    assert len(prepared.preflight_result.model_content) <= 1000
    result = await executor.execute_prepared(prepared, context)
    assert len(result.model_content) <= 1000
    assert result.artifact is None
    assert not (tmp_path / ".iris").exists()


@pytest.mark.asyncio
async def test_tiny_budget_keeps_retrieval_notice_without_preview(tmp_path: Path) -> None:
    """无法容纳完整路径的预算保留取回说明，预览不再额外占空间。"""

    def produce() -> str:
        """返回足以触发小预算的正文。"""
        return "preview text"

    registry = ToolRegistry()
    tool = registry.register_function(produce)
    tool.definition.max_result_chars = 3
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="small", name="produce", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert result.artifact is not None
    assert result.model_content.startswith("\n\n[")
    assert str(result.artifact.path) in result.model_content


@pytest.mark.asyncio
async def test_web_fetch_artifact_can_be_read_with_configured_file_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """批量网页长正文沿配置、HTTP、artifact 和 read_file 链路完整可读。"""
    import iris.config as iris_config
    from iris.agents import ToolsConfig, build_tool_registry

    body = "资料" * 30000 + "\nWEB_BODY_END"
    url = "https://example.com/article"
    failed_url = "https://example.com/unavailable"

    async def respond(
        client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        """只替换网络发送，保留真实客户端和工具响应解析。"""
        assert request.headers["Authorization"] == "Bearer tvly-artifact-test"
        assert str(request.url) == "https://api.tavily.com/extract"
        assert json.loads(request.content)["urls"] == [url, failed_url]
        return httpx.Response(
            200,
            json={
                "results": [{"url": url, "raw_content": body}],
                "failed_results": [{"url": failed_url, "error": "Unavailable"}],
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", respond)
    monkeypatch.setattr(iris_config, "_config", None)
    iris_config.init_config(provider_api_keys={"tavily": "tvly-artifact-test"})
    registry = build_tool_registry(ToolsConfig(builtin=["web.fetch", "file.read"]))
    executor = ToolExecutor(registry)
    context = ToolExecutionContext(workspace_root=tmp_path)

    result = await executor.execute_one(
        ToolUseBlock(id="fetch", name="web_fetch", input={"urls": [url, failed_url]}), context
    )

    assert not result.is_error
    assert result.artifact is not None
    assert failed_url in result.model_content
    assert str(result.artifact.path) in result.model_content
    assert body in result.artifact.path.read_text(encoding="utf-8")
    read_input: dict[str, Any] = {"file_path": str(result.artifact.path)}
    page = await executor.execute_one(
        ToolUseBlock(id="read-first", name="read_file", input=read_input), context
    )
    assert not page.is_error
    cursor = re.search(r"next_offset=(\d+), next_column=(\d+); has_more=true", page.model_content)
    assert cursor is not None
    read_input.update(offset=int(cursor[1]), column=int(cursor[2]))
    remainder = await executor.execute_one(
        ToolUseBlock(id="read-rest", name="read_file", input=read_input), context
    )
    assert not remainder.is_error
    assert "WEB_BODY_END" in remainder.model_content
    assert "has_more=false" in remainder.model_content
