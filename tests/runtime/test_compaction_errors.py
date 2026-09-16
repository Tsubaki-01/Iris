"""压缩失败沿用 context 来源与稳定错误码。"""

import pytest

from iris.exceptions import IrisContextCompactionError
from iris.runtime.runtime import _normalize_run_error


@pytest.mark.parametrize(
    "code",
    [
        "CONTEXT_COMPACTION_UNAVAILABLE",
        "CONTEXT_COMPACTION_FAILED",
        "CONTEXT_COMPACTION_TIMEOUT",
    ],
)
def test_compaction_errors_keep_context_source(code: str) -> None:
    error = _normalize_run_error(IrisContextCompactionError("摘要无法完成", code=code))
    assert error.source == "context"
    assert error.code == code
