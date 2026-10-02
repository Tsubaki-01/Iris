"""Hook 领域错误复用既有 Runtime 分类，不扩展 durable source 词表。"""

import pytest

from iris.exceptions import IrisHookError, IrisHookProtocolError
from iris.runtime.runtime import _normalize_run_error


@pytest.mark.parametrize(
    ("error_type", "code"),
    [(IrisHookError, "HOOK_ERROR"), (IrisHookProtocolError, "HOOK_PROTOCOL_ERROR")],
)
def test_hook_error_keeps_runtime_source_and_handler_details(
    error_type: type[IrisHookError], code: str
) -> None:
    """真实错误投影保留事件定位信息并能构造既有 RunErrorInfo。"""
    error = error_type("invalid output", event="tool.after", handler="checker")

    normalized = _normalize_run_error(error)

    assert normalized.source == "runtime"
    assert normalized.code == code
    assert normalized.message == str(error)
    assert normalized.details == {"event": "tool.after", "handler": "checker"}
