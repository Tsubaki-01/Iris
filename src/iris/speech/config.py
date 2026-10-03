"""语音输入配置的唯一 YAML 与公开模型解析边界。"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

type SpeechAdapterName = Literal["doubao_asr", "dashscope_funasr"]
type _ConnectionString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SpeechConfig(BaseModel):
    """默认关闭的语音输入声明，由宿主显式装配客户端。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(default=False, strict=True)
    adapter: SpeechAdapterName | None = None
    endpoint: _ConnectionString | None = None
    model: _ConnectionString | None = None

    @model_validator(mode="after")
    def _validate_enabled_connection(self) -> Self:
        if self.enabled and None in (self.adapter, self.endpoint, self.model):
            raise ValueError("启用 speech 时必须同时配置 adapter、endpoint 和 model")
        return self


__all__ = ["SpeechConfig"]
