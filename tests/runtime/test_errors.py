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
    RuntimeFactory,
    RuntimeMessageAssembler,
    RuntimeProvider,
    ToolBridge,
    normalize_runtime_error,
)


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
    assert RuntimeMessageAssembler.__name__ == "RuntimeMessageAssembler"
    assert RuntimeProvider.__name__ == "RuntimeProvider"
    assert ToolBridge.__name__ == "ToolBridge"
    assert callable(normalize_runtime_error)
    assert hasattr(AgentRuntime, "run_loop")
