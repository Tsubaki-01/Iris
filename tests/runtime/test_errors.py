from __future__ import annotations

from iris.exceptions import (
    IrisConfigError,
    IrisContextError,
    IrisMemoryError,
    IrisProviderError,
    IrisSessionError,
    IrisToolExecutionError,
)
from iris.runtime import (
    AgentRuntime,
    RuntimeEnvironment,
    RuntimeFactory,
    RuntimeMessageAssembler,
    RuntimeProvider,
    ToolBridge,
    normalize_runtime_error,
)
from iris.runtime.errors import error_result, tool_error_info
from iris.runtime.models import RuntimeErrorInfo, RuntimeStatus
from iris.tools import ToolErrorInfo, ToolResult


def test_domain_exceptions_map_to_stable_runtime_error_info() -> None:
    cases = [
        (IrisConfigError("配置错误"), "CONFIG_ERROR", "config"),
        (IrisContextError("context 错误"), "CONTEXT_ERROR", "context"),
        (IrisProviderError("provider 错误"), "PROVIDER_ERROR", "provider"),
        (IrisMemoryError("memory 错误"), "MEMORY_ERROR", "memory"),
        (IrisSessionError("session 错误"), "SESSION_ERROR", "session"),
        (IrisToolExecutionError("tool 错误"), "PROTOCOL_ERROR", "tool"),
        (RuntimeError("未知错误"), "RUNTIME_ERROR", "runtime"),
    ]

    for exception, code, source in cases:
        error = normalize_runtime_error(exception)

        assert error.code == code
        assert error.source == source
        assert error.message


def test_runtime_public_exports_include_stable_surface() -> None:
    assert AgentRuntime.__name__ == "AgentRuntime"
    assert RuntimeFactory.__name__ == "RuntimeFactory"
    assert RuntimeEnvironment.__name__ == "RuntimeEnvironment"
    assert RuntimeMessageAssembler.__name__ == "RuntimeMessageAssembler"
    assert RuntimeProvider.__name__ == "RuntimeProvider"
    assert ToolBridge.__name__ == "ToolBridge"
    assert callable(normalize_runtime_error)
    assert hasattr(AgentRuntime, "run_loop")


def test_error_result_preserves_runtime_error_context() -> None:
    error = RuntimeErrorInfo(
        code="SESSION_ERROR",
        message="session failed",
        source="session",
    )

    result = error_result(
        session_id="session-1",
        run_id="run-1",
        error=error,
        metadata={"trace_id": "trace-1"},
    )

    assert result.status is RuntimeStatus.ERROR
    assert result.error == error
    assert result.steps == 1
    assert result.metadata == {"trace_id": "trace-1"}


def test_tool_error_info_uses_first_structured_error_or_fallback() -> None:
    results = [
        ToolResult(
            tool_use_id="call-1",
            tool_name="write_note",
            is_error=True,
            error=ToolErrorInfo(
                code="PERMISSION_ERROR",
                message="denied",
                details={"effect": "deny"},
            ),
        )
    ]

    error = tool_error_info(results)
    fallback = tool_error_info([])

    assert error.model_dump() == {
        "code": "PERMISSION_ERROR",
        "message": "denied",
        "source": "tool",
        "details": {"effect": "deny"},
    }
    assert fallback.code == "TOOL_ERROR"
    assert fallback.source == "tool"
