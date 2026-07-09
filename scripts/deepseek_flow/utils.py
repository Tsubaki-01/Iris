"""DeepSeek live 验证场景共享工具函数。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from iris.message import LLMRequest

from .config import _redact_current_api_key
from .constants import PROVIDER_OK_TEXT, RUNTIME_OK_PREFIX
from .models import ScenarioReport


async def _retry_assertion(
    attempt: Callable[[], Awaitable[ScenarioReport]],
    retries: int,
) -> ScenarioReport:
    """对模型未按要求调用工具的场景做有限重试。"""
    last_report: ScenarioReport | None = None
    total_api_calls = 0
    total_steps = 0
    attempt_summaries: list[dict[str, object]] = []
    for attempt_index in range(retries + 1):
        report = await attempt()
        total_api_calls += int(report["api_calls"])
        total_steps += int(report["steps"])
        attempt_summaries.append(_attempt_summary(report, attempt_index + 1))
        _attach_retry_evidence(
            report,
            attempt_index=attempt_index + 1,
            total_api_calls=total_api_calls,
            total_steps=total_steps,
            attempt_summaries=attempt_summaries,
        )
        if report["ok"] or not _is_retryable_assertion(report):
            return report
        last_report = report
    if last_report is None:
        raise RuntimeError("retry attempt 未执行")
    return last_report


def _attach_retry_evidence(
    report: ScenarioReport,
    *,
    attempt_index: int,
    total_api_calls: int,
    total_steps: int,
    attempt_summaries: list[dict[str, object]],
) -> None:
    """把 retry 汇总写回当前报告。"""
    report["api_calls"] = total_api_calls
    report["steps"] = total_steps
    report["evidence"]["attempt"] = attempt_index
    report["evidence"]["attempts"] = list(attempt_summaries)


def _attempt_summary(report: ScenarioReport, attempt_index: int) -> dict[str, object]:
    """返回单次尝试的报告摘要。"""
    return {
        "attempt": attempt_index,
        "ok": report["ok"],
        "status": report["status"],
        "api_calls": report["api_calls"],
        "steps": report["steps"],
        "actual": report["actual"],
        "error_code": report["error_code"],
        "error_message": report["error_message"],
    }


def _is_retryable_assertion(report: ScenarioReport) -> bool:
    """判断是否属于模型未按要求响应导致的可重试断言失败。"""
    return (
        not bool(report["ok"])
        and report["status"] == "assertion_failed"
        and report["error_code"] == "ASSERTION_FAILED"
    )


def _tool_choice(name: str) -> dict[str, dict[str, str] | str]:
    """返回 OpenAI Chat 格式的强制工具选择参数。"""
    return {"type": "function", "function": {"name": name}}


def _request_has_tool_result(request: LLMRequest) -> bool:
    """判断 provider 请求中是否包含 tool result message。"""
    return any(message.tool_results for message in request.messages)


def _first_tool_error_code(results: list[Any]) -> str:
    """读取第一个工具结果错误码。"""
    if not results:
        return "NO_TOOL_RESULT"
    first = results[0]
    error = getattr(first, "error", None)
    return error.code if error is not None else "ok"


def _runtime_error_code(result: Any) -> str:
    """从 runtime result 中读取错误码。"""
    return result.error.code if result.error else "ASSERTION_FAILED"


def _runtime_error_message(result: Any, fallback: str) -> str:
    """从 runtime result 中读取错误信息。"""
    return _redact_current_api_key(result.error.message if result.error else fallback)


def _provider_smoke_ok(text: str) -> bool:
    """判断 provider smoke 输出是否严格匹配。"""
    return text.strip() == PROVIDER_OK_TEXT


def _runtime_final_ok(text: str, expected_token: str) -> bool:
    """判断 runtime read loop 最终回答是否严格匹配。"""
    stripped = text.strip()
    return (
        "\n" not in stripped
        and stripped.startswith(RUNTIME_OK_PREFIX)
        and expected_token in stripped
    )


def _scenario_dir(work_dir: Path, name: str) -> Path:
    """返回单个场景的隔离目录。"""
    path = work_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path
