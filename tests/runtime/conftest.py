"""Runtime 模板测试使用独立项目来源。"""

from pathlib import Path

import pytest

from iris.prompts import PromptSnapshot, PromptSource


@pytest.fixture
def prompt_snapshot(tmp_path: Path) -> PromptSnapshot:
    """为当前用例初始化默认项目模板。"""
    return PromptSource.initialize(tmp_path).snapshot()
