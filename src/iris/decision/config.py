"""独立 Decision YAML 的解析边界。"""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from ..exceptions import IrisConfigError
from .jev import JEV_DEFAULT_MODEL, JEV_DEFAULT_TIMEOUT_SECONDS


class DecisionToolsConfig(BaseModel):
    """当前已实现的工具判断接点。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    discovery: bool = Field(default=False, strict=True)


class DecisionConfig(BaseModel):
    """服务参数与按接点独立启用的构造期配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["typesafe"] = "typesafe"
    model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = (
        JEV_DEFAULT_MODEL
    )
    timeout_seconds: float = Field(
        default=JEV_DEFAULT_TIMEOUT_SECONDS, gt=0, allow_inf_nan=False, strict=True
    )
    tools: DecisionToolsConfig = Field(default_factory=DecisionToolsConfig)

    @property
    def enabled(self) -> bool:
        """任一已实现接点开启时才需要 evaluator。"""
        return self.tools.discovery


def load_decision_config(path: str | Path) -> DecisionConfig:
    """读取指定文件并将文件、YAML 和原始字段错误归入配置异常。"""
    path = Path(path)
    try:
        return DecisionConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise IrisConfigError("Decision 配置文件读取或解析失败", path=str(path)) from exc


__all__ = ["DecisionConfig", "DecisionToolsConfig", "load_decision_config"]
