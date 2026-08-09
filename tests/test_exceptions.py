"""验证 Iris 公共异常边界。

Example:
    uv run pytest tests/test_exceptions.py
"""

import iris.exceptions as exceptions_package
import iris.tools as tools_package
from iris.exceptions import IrisError


def test_cancellation_requested_error_belongs_to_runtime_exception_boundary() -> None:
    """取消请求应由中央异常包拥有，而不是作为工具错误导出。"""
    cancellation_error = getattr(
        exceptions_package,
        "IrisCancellationRequestedError",
        None,
    )

    assert cancellation_error is not None
    assert issubclass(cancellation_error, IrisError)
    error = cancellation_error("activation 已请求取消")
    assert error.runtime_source == "runtime"
    assert error.runtime_code == "CANCELLATION_REQUESTED"
    assert not hasattr(tools_package, "CancellationRequestedError")
