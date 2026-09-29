"""命令执行配置在原始输入边界解析，核心导入不加载 Docker。"""

import subprocess
import sys

import pytest
from pydantic import ValidationError

from iris.command import CommandConfig, CommandMode, DockerConfig


def test_default_command_is_native_without_docker_configuration() -> None:
    config = CommandConfig()
    assert config.mode is CommandMode.NATIVE
    assert config.timeout_seconds == 120
    assert config.docker is None


def test_docker_mode_supplies_resource_defaults() -> None:
    config = CommandConfig.model_validate({"mode": "docker"})
    assert config.docker == DockerConfig()
    assert config.docker.image == "python:3.12-slim"
    assert config.docker.network == "none"
    assert (config.docker.cpus, config.docker.memory_mb, config.docker.pids_limit) == (
        2.0,
        1024,
        128,
    )


@pytest.mark.parametrize("docker", [{}, None, {"network": "bridge"}])
def test_native_rejects_any_explicit_docker_block(docker: object) -> None:
    with pytest.raises(ValidationError, match="docker"):
        CommandConfig.model_validate({"mode": "native", "docker": docker})


@pytest.mark.parametrize(
    "endpoint",
    ["unix:///var/run/docker.sock", "npipe:////./pipe/docker_engine"],
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
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"extra": "ignored"},
        {"mode": "docker", "docker": {"network": "host"}},
        {"mode": "docker", "docker": {"cpus": 0}},
        {"mode": "docker", "docker": {"memory_mb": -1}},
        {"mode": "docker", "docker": {"pids_limit": 0}},
        {"mode": "docker", "docker": {"environment": {"COUNT": 1}}},
    ],
)
def test_configuration_constraints_are_owned_by_models(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CommandConfig.model_validate(payload)


def test_core_import_does_not_load_optional_driver() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import iris.command; assert 'aiodocker' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
