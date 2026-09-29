"""命令执行配置在原始输入边界解析，核心导入不加载 Docker。"""

import subprocess
import sys

import pytest
from pydantic import ValidationError

from iris.command import CommandConfig, CommandMode
from iris.sandbox import DockerConfig


def test_default_command_is_native_without_docker_configuration() -> None:
    config = CommandConfig()
    assert config.mode is CommandMode.NATIVE
    assert config.timeout_seconds == 120
    assert config.docker is None


def test_docker_mode_supplies_resource_defaults() -> None:
    config = CommandConfig.model_validate({"mode": "docker"})
    assert config.docker == DockerConfig()


@pytest.mark.parametrize("docker", [{}, None, {"network": "bridge"}])
def test_native_rejects_any_explicit_docker_block(docker: object) -> None:
    with pytest.raises(ValidationError, match="docker"):
        CommandConfig.model_validate({"mode": "native", "docker": docker})


@pytest.mark.parametrize(
    "payload",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"extra": "ignored"},
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
            "import sys; import iris.command; import iris.sandbox; "
            "assert 'aiodocker' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
