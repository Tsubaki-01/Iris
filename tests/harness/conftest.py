"""Lifecycle harness contract 共用的确定性 fixtures。"""

from datetime import UTC, datetime, timedelta

import pytest


@pytest.fixture
def lifecycle_now() -> datetime:
    """返回不会读取 wall clock 的固定 UTC 时间。"""
    return datetime(2026, 1, 2, 3, 4, tzinfo=UTC)


@pytest.fixture
def lifecycle_later(lifecycle_now: datetime) -> datetime:
    """返回固定时间之后的一秒。"""
    return lifecycle_now + timedelta(seconds=1)
