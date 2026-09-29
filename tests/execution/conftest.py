"""真实 Docker 测试显式启用；普通测试不连接引擎。"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """注册本目录的真实本地引擎验收开关。"""
    parser.addoption(
        "--run-docker",
        action="store_true",
        default=False,
        help="运行真实本地 Docker 测试；需 sandbox extra、Linux engine 和预备镜像",
    )
