"""性能观测测试的共享 pytest 配置。"""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """注册显式本机计时开关。"""
    parser.addoption(
        "--run-performance-timing",
        action="store_true",
        default=False,
        help="运行仅用于同机前后比较的性能计时场景",
    )


def pytest_configure(config: pytest.Config) -> None:
    """注册性能计时 marker，避免修改项目级 pytest 配置。"""
    config.addinivalue_line(
        "markers",
        "performance_timing: 仅在 --run-performance-timing 下执行的本机计时",
    )


@pytest.fixture
def require_performance_timing(request: pytest.FixtureRequest) -> None:
    """未显式启用时跳过本机计时场景。"""
    if not request.config.getoption("--run-performance-timing"):
        pytest.skip("需要 --run-performance-timing")


@pytest.fixture
def baseline_metadata() -> dict[str, str | bool]:
    """采集本次观测对应的 Git、Python 与平台事实。"""
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "git_head": head,
        "dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


@pytest.fixture
def record_observation(
    baseline_metadata: dict[str, str | bool],
) -> Callable[..., None]:
    """返回输出稳定 JSON 性能观测的记录函数。"""

    def record(
        *,
        scenario: str,
        perf_ids: tuple[str, ...],
        fixture: dict[str, int | str | bool],
        samples_ms: tuple[float, ...],
        counters: dict[str, int],
    ) -> None:
        if not scenario or not perf_ids:
            raise AssertionError("性能观测必须声明 scenario 和 perf_ids")
        if not samples_ms or any(sample < 0 for sample in samples_ms):
            raise AssertionError("性能观测必须包含非负计时样本")
        if any(value < 0 for value in counters.values()):
            raise AssertionError("性能观测计数不得为负")
        payload: dict[str, Any] = {
            "scenario": scenario,
            "perf_ids": list(perf_ids),
            **baseline_metadata,
            "fixture": fixture,
            "sample_count": len(samples_ms),
            "samples_ms": [round(sample, 3) for sample in samples_ms],
            "counters": counters,
        }
        print(f"PERF_OBSERVATION {json.dumps(payload, ensure_ascii=False, sort_keys=True)}")

    return record
