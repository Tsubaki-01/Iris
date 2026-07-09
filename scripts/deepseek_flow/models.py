"""DeepSeek live 验证脚本共享类型。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

ScenarioReport = dict[str, Any]
ScenarioRunner = Callable[[Path, int], Awaitable[ScenarioReport]]
