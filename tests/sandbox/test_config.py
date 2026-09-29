"""Docker 配置在 sandbox 的原始输入边界解析。"""

import pytest
from pydantic import ValidationError

from iris.sandbox import DockerConfig


def test_default_container_configuration() -> None:
    config = DockerConfig()
    assert config.image == "iris-command:local"
    assert config.network == "none"
    assert (config.cpus, config.memory_mb, config.pids_limit) == (2.0, 1024, 128)


@pytest.mark.parametrize(
    "endpoint", ["unix:///var/run/docker.sock", "npipe:////./pipe/docker_engine"]
)
def test_local_endpoints_are_preserved(endpoint: str) -> None:
    assert DockerConfig(endpoint=endpoint).endpoint == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "tcp://localhost:2375",
        "ssh://local/docker",
        "http://remote:2375",
        "unix://remote/docker.sock",
        "unix://",
        "npipe:////remote/pipe/docker_engine",
        "unix:///var/run/docker.sock?remote=true",
    ],
)
def test_nonlocal_or_invalid_endpoints_are_rejected(endpoint: str) -> None:
    with pytest.raises(ValidationError, match="endpoint"):
        DockerConfig(endpoint=endpoint)


@pytest.mark.parametrize(
    "payload",
    [
        {"network": "host"},
        {"cpus": 0},
        {"memory_mb": -1},
        {"pids_limit": 0},
        {"environment": {"COUNT": 1}},
    ],
)
def test_resource_constraints_are_owned_by_docker_config(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DockerConfig.model_validate(payload)
