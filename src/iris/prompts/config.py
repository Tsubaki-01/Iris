"""项目模板目录的纯声明配置。"""

from pydantic import BaseModel, ConfigDict, Field


class PromptConfig(BaseModel):
    """相对 root workspace 解析的模板目录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: str = Field(default=".iris/prompts", min_length=1)
