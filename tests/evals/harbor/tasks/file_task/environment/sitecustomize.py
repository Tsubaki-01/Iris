"""仅在指定容器 fixture 中注入 SDK 关闭失败，不修改正式 runner。"""

import json
import sys
from pathlib import Path

if (
    sys.prefix == "/installed-agent/iris/venv"
    and json.loads(Path("/fixture/scenario.json").read_text())["mode"] == "cleanup_failure"
):
    from iris.exceptions import IrisRunStateError
    from iris.harness import AgentRunner

    original_close = AgentRunner.aclose

    async def broken_close(self: AgentRunner) -> None:
        """真实资源已关闭后报告受控错误，验证 adapter 的独立错误通道。"""
        await original_close(self)
        raise IrisRunStateError("integration cleanup failure")

    AgentRunner.aclose = broken_close
