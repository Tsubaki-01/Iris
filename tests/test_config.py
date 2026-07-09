from collections.abc import Generator
from pathlib import Path

import pytest

import iris
from iris.config import Config, ProviderConfig
from iris.exceptions import IrisConfigError


@pytest.fixture(autouse=True)
def reset_config_state(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    for name in (
        "IRIS_API_KEY",
        "IRIS_PROVIDER_API_KEYS__DEEPSEEK",
        "IRIS_PROVIDER_API_KEYS__OPENAI",
        "IRIS_PROVIDER_API_KEYS__SILICONFLOW",
        "IRIS_PROVIDERS__SILICONFLOW__LITELLM_PROVIDER",
        "IRIS_PROVIDERS__SILICONFLOW__BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    iris.reset()
    yield
    iris.reset()


def test_config_does_not_expose_unimplemented_retry_field() -> None:
    assert "max_retries" not in Config.model_fields


def test_init_config_loads_values_from_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("IRIS_API_KEY=sk-env-file\n", encoding="utf-8")
    monkeypatch.delenv("IRIS_API_KEY", raising=False)

    config = iris.init_config(env_file=str(env_file))

    assert config.api_key == "sk-env-file"
    assert iris.get_config().api_key == "sk-env-file"


def test_init_config_loads_provider_api_keys_from_nested_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "IRIS_PROVIDER_API_KEYS__DEEPSEEK=deepseek-key\n"
        "IRIS_PROVIDER_API_KEYS__OPENAI=openai-key\n"
        "IRIS_API_KEY=generic-key\n",
        encoding="utf-8",
    )
    for name in (
        "IRIS_API_KEY",
        "IRIS_PROVIDER_API_KEYS__DEEPSEEK",
        "IRIS_PROVIDER_API_KEYS__OPENAI",
    ):
        monkeypatch.delenv(name, raising=False)

    config = iris.init_config(env_file=str(env_file))

    assert config.api_key == "generic-key"
    assert config.provider_api_keys == {
        "deepseek": "deepseek-key",
        "openai": "openai-key",
    }


def test_init_config_nested_env_var_overrides_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "IRIS_PROVIDER_API_KEYS__DEEPSEEK=file-deepseek-key\n"
        "IRIS_API_KEY=file-generic-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("IRIS_PROVIDER_API_KEYS__DEEPSEEK", "env-deepseek-key")
    monkeypatch.delenv("IRIS_API_KEY", raising=False)

    config = iris.init_config(env_file=str(env_file))

    assert config.api_key == "file-generic-key"
    assert config.provider_api_keys == {"deepseek": "env-deepseek-key"}


def test_init_config_explicit_provider_api_keys_override_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "IRIS_API_KEY=file-generic-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("IRIS_PROVIDER_API_KEYS__DEEPSEEK", "env-deepseek-key")
    monkeypatch.delenv("IRIS_API_KEY", raising=False)

    config = iris.init_config(
        env_file=str(env_file),
        provider_api_keys={"deepseek": "explicit-deepseek-key"},
    )

    assert config.api_key == "file-generic-key"
    assert config.provider_api_keys == {"deepseek": "explicit-deepseek-key"}


def test_init_config_loads_provider_config_from_nested_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "IRIS_PROVIDER_API_KEYS__SILICONFLOW=sk-siliconflow\n"
        "IRIS_PROVIDERS__SILICONFLOW__LITELLM_PROVIDER=openai\n"
        "IRIS_PROVIDERS__SILICONFLOW__BASE_URL=https://api.siliconflow.cn/v1\n",
        encoding="utf-8",
    )
    for name in (
        "IRIS_PROVIDER_API_KEYS__SILICONFLOW",
        "IRIS_PROVIDERS__SILICONFLOW__LITELLM_PROVIDER",
        "IRIS_PROVIDERS__SILICONFLOW__BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    config = iris.init_config(env_file=str(env_file))

    assert config.provider_api_keys == {"siliconflow": "sk-siliconflow"}
    assert config.providers == {
        "siliconflow": ProviderConfig(
            litellm_provider="openai",
            base_url="https://api.siliconflow.cn/v1",
        )
    }


def test_init_config_allows_provider_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IRIS_API_KEY", raising=False)
    config = iris.init_config(
        providers={
            "siliconflow": ProviderConfig(
                litellm_provider="openai",
                base_url="https://api.siliconflow.cn/v1",
            )
        }
    )

    assert config.api_key is None
    assert config.provider_api_keys == {}
    assert "siliconflow" in config.providers


def test_init_config_env_file_keeps_explicit_kwargs_strict(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "IRIS_API_KEY=sk-env-file\nIRIS_UNRELATED=value\n",
        encoding="utf-8",
    )

    with pytest.raises(IrisConfigError, match="未知配置字段"):
        iris.init_config(env_file=str(env_file), api_ky="typo")


def test_init_config_normalizes_blank_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IRIS_API_KEY", raising=False)
    config = iris.init_config(api_key="   ")

    assert config.api_key is None


def test_package_exports_minimal_config_lifecycle() -> None:
    config = iris.init_config(api_key="sk-test")

    assert config.api_key == "sk-test"
    assert iris.get_config().api_key == "sk-test"

    iris.reset()

    with pytest.raises(IrisConfigError, match="配置尚未初始化"):
        iris.get_config()
