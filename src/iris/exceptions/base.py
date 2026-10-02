"""Iris 异常基类与通用校验错误。"""

from typing import Any, ClassVar


class IrisError(Exception):
    """所有 Iris 特定错误的基类。"""

    runtime_error_source: ClassVar[str] = "runtime"
    runtime_error_code: ClassVar[str] = "RUNTIME_ERROR"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    @property
    def runtime_source(self) -> str:
        """返回 runtime 错误来源。"""
        return self.runtime_error_source

    @property
    def runtime_code(self) -> str:
        """返回 runtime 稳定错误码。"""
        return self.runtime_error_code

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class IrisValidationError(IrisError):
    """输入或配置校验失败时抛出。"""
