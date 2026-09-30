"""Todo 工作清单的声明式开关。"""

from pydantic import BaseModel, ConfigDict


class TodoConfig(BaseModel):
    """控制当前 Agent 是否读取会话 Todo 清单。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False


__all__ = ["TodoConfig"]
