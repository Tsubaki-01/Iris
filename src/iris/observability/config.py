"""采集策略与 OTLP 导出配置，不初始化观测 SDK。"""

from pydantic import BaseModel, ConfigDict, Field


class AgentObservabilityConfig(BaseModel):
    """构建期确定的 Agent 采集策略。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    capture_content: bool = False
    max_content_chars: int = Field(default=65536, gt=0)


class ObservabilityExportConfig(BaseModel):
    """宿主统一配置的 OTLP HTTP/protobuf 导出目标。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    traces_endpoint: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    service_name: str = "iris"
    timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
