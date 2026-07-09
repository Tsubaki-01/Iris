from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser: pytest.Parser) -> None:
    """注册需要显式开启的 live integration 测试参数。"""
    parser.addoption(
        "--run-live-deepseek",
        action="store_true",
        default=False,
        help="运行需要真实 DeepSeek API key 和外网的 live integration 测试。",
    )


def pytest_configure(config: pytest.Config) -> None:
    """注册项目自定义 pytest marker。"""
    config.addinivalue_line(
        "markers",
        "live_deepseek: 需要真实 DeepSeek API key 和外网的 live integration 测试",
    )
