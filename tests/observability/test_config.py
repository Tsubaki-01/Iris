"""观测配置复用 Agent 与全局配置边界。"""

import builtins
import subprocess
import sys

import pytest
from pydantic import ValidationError

from iris.agents import AgentConfig
from iris.config import Config
from iris.exceptions import IrisConfigError
from iris.observability import AgentObservabilityConfig, ObservabilityExportConfig
from iris.observability.service import Observability


def test_configuration_defaults_and_existing_boundaries() -> None:
    agent = AgentConfig(name="demo", model="openai/test", system="hello")
    assert agent.observability == AgentObservabilityConfig()
    assert not agent.observability.enabled
    assert not agent.observability.capture_content
    assert agent.observability.max_content_chars == 65536
    config = Config(observability={"traces_endpoint": "http://localhost:5000/v1/traces"})
    assert config.observability.traces_endpoint == "http://localhost:5000/v1/traces"
    one, two = ObservabilityExportConfig(), ObservabilityExportConfig()
    one.headers["experiment"] = "1"
    assert two.headers == {}


@pytest.mark.parametrize("field", ["max_content_chars"])
def test_content_limits_are_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        AgentObservabilityConfig(**{field: 0})
    with pytest.raises(ValidationError):
        ObservabilityExportConfig(timeout_seconds=0)


def test_enabled_requires_explicit_export_source() -> None:
    with pytest.raises(IrisConfigError, match="traces_endpoint"):
        Observability.from_config(
            AgentObservabilityConfig(enabled=True), ObservabilityExportConfig()
        )


def test_missing_extra_is_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def without_sdk(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("opentelemetry.sdk"):
            raise ImportError("test: no SDK")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sdk)
    with pytest.raises(IrisConfigError, match="observability"):
        Observability.from_config(
            AgentObservabilityConfig(enabled=True),
            ObservabilityExportConfig(traces_endpoint="http://localhost:5000/v1/traces"),
        )


def test_disabled_import_and_configuration_do_not_require_extra() -> None:
    code = """
import sys
from importlib.abc import MetaPathFinder
class NoSDK(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("opentelemetry.sdk", "opentelemetry.exporter")):
            raise ImportError(fullname)
sys.meta_path.insert(0, NoSDK())
from iris.config import Config
from iris.observability import AgentObservabilityConfig
from iris.observability.service import Observability
obs = Observability.from_config(AgentObservabilityConfig(), Config().observability)
with obs.scope("disabled") as span:
    assert not span.is_recording()
assert not obs.enabled
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
