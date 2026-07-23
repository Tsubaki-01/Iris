"""DeepSeek provider live 验证。"""

from __future__ import annotations

from pathlib import Path

from iris.message import LLMRequest, LLMResponse, Msg
from iris.providers import create_provider_client
from iris.runtime import RuntimeProvider

from .config import _safe_error_message
from .constants import DEFAULT_MODEL, DEFAULT_PROVIDER_ROUTE, PROVIDER_OK_TEXT
from .models import ScenarioReport
from .reporting import scenario_report
from .utils import _provider_smoke_ok


class RecordingRuntimeProvider:
    """包装真实 provider，记录 runtime 发出的 `LLMRequest`。"""

    def __init__(self, delegate: RuntimeProvider) -> None:
        """创建 recording wrapper。

        Args:
            delegate: 实际执行 DeepSeek API 调用的 provider。
        """
        self.delegate = delegate
        self.requests: list[LLMRequest] = []

    @property
    def api_call_count(self) -> int:
        """返回已委托的真实 API 调用次数。"""
        return len(self.requests)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """记录请求并转发给真实 provider。"""
        self.requests.append(request)
        return await self.delegate.complete(request)

    def request_snapshots(self) -> list[dict[str, object]]:
        """返回适合写入报告的 provider 请求摘要。"""
        return [
            {
                "index": index,
                "model": request.model,
                "message_count": len(request.messages),
                "roles": [message.role.value for message in request.messages],
                "tool_schema_names": _tool_schema_names(request.tools),
                "tool_choice": request.tool_choice,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "has_tool_result": any(message.tool_results for message in request.messages),
                "messages": [_message_snapshot(message) for message in request.messages],
            }
            for index, request in enumerate(self.requests, start=1)
        ]


def recording_provider() -> RecordingRuntimeProvider:
    """创建记录请求的真实 DeepSeek provider。"""
    return RecordingRuntimeProvider(create_provider_client(DEFAULT_PROVIDER_ROUTE, timeout=60))


def _tool_schema_names(tools: list[dict[str, object]]) -> list[str]:
    """从 OpenAI Chat 风格工具 schema 中提取工具名。"""
    names: list[str] = []
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
        elif isinstance(tool.get("name"), str):
            names.append(tool["name"])
    return names


def _message_snapshot(message: Msg) -> dict[str, object]:
    """返回单条消息的摘要。"""
    return {
        "role": message.role.value,
        "text_preview": _message_text_preview(message),
        "tool_call_names": [tool_call.name for tool_call in message.tool_calls],
        "tool_result_names": [tool_result.name for tool_result in message.tool_results],
        "has_tool_result": bool(message.tool_results),
    }


def _message_text_preview(message: Msg) -> str:
    """提取普通文本或工具结果内容摘要。"""
    text = message.text
    if text:
        return text[:500]
    return "\n".join(result.content for result in message.tool_results)[:500]


async def run_provider_smoke_live(work_dir: Path, retries: int) -> ScenarioReport:
    """验证 provider factory 与 DeepSeek 直接调用。"""
    del work_dir, retries
    client = create_provider_client(DEFAULT_PROVIDER_ROUTE, timeout=60)
    api_calls = 0
    try:
        api_calls = 1
        response = await client.complete(
            LLMRequest(
                model=DEFAULT_MODEL,
                messages=[
                    Msg.user("这是连通性验证。请只回答 IRIS_PROVIDER_OK，不要添加其他内容。")
                ],
                temperature=0,
                max_tokens=32,
                timeout=60,
            )
        )
    except Exception as exc:
        return scenario_report(
            name="provider_smoke_live",
            ok=False,
            status="error",
            api_calls=api_calls,
            steps=1,
            expected=PROVIDER_OK_TEXT,
            actual=exc.__class__.__name__,
            evidence={
                "provider_route": DEFAULT_PROVIDER_ROUTE,
                "model": DEFAULT_MODEL,
            },
            error_code=exc.__class__.__name__,
            error_message=_safe_error_message(exc),
        )

    text = response.to_msg().text.strip()
    ok = _provider_smoke_ok(text)
    return scenario_report(
        name="provider_smoke_live",
        ok=ok,
        status="ok" if ok else "assertion_failed",
        api_calls=api_calls,
        steps=1,
        expected=PROVIDER_OK_TEXT,
        actual=text,
        evidence={
            "provider": response.provider,
            "model": response.model or DEFAULT_MODEL,
            "total_tokens": response.total_tokens,
        },
        error_code="" if ok else "ASSERTION_FAILED",
        error_message="" if ok else "Provider smoke 输出不等于 IRIS_PROVIDER_OK",
    )
