"""显式启用需要 Docker 与安装网络的接入验收。"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """让普通单元测试默认不启动容器。"""
    parser.addoption("--run-harbor-docker", action="store_true", default=False)


@pytest.fixture(scope="session")
def harbor_docker(request: pytest.FixtureRequest) -> None:
    """仅在显式选择时运行真实容器场景。"""
    if not request.config.getoption("--run-harbor-docker"):
        pytest.skip("使用 --run-harbor-docker 启用容器验收")
