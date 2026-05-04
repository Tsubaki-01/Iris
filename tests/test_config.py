from collections.abc import Generator
from pathlib import Path

import pytest

import iris
from iris.config import Config
from iris.exceptions import IrisConfigError


@pytest.fixture(autouse=True)
def reset_config_state() -> Generator[None, None, None]:
    iris.reset()
    yield
    iris.reset()


def test_config_does_not_expose_unimplemented_retry_field() -> None:
    assert "max_retries" not in Config.model_fields


def test_init_config_loads_values_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("IRIS_API_KEY=sk-env-file\n", encoding="utf-8")

    config = iris.init_config(env_file=str(env_file))

    assert config.api_key == "sk-env-file"
    assert iris.get_config().api_key == "sk-env-file"


def test_package_exports_minimal_config_lifecycle() -> None:
    config = iris.init_config(api_key="sk-test")

    assert config.api_key == "sk-test"
    assert iris.get_config().api_key == "sk-test"

    iris.reset()

    with pytest.raises(IrisConfigError, match="配置尚未初始化"):
        iris.get_config()
