"""Runtime safe boundary 的瞬时 steering 协议。

该模块只定义 activation-scoped 输入与回执边界，不拥有 queue 或持久化。

Example:
    >>> from iris.message import Msg
    >>> item = SteeringInput(submission_id="submission-1", message=Msg.user("继续"))
    >>> item.message.text
    '继续'
"""

# region imports
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator

from ..message import Msg, Role

# endregion


class SteeringInput(BaseModel):
    """一条等待与既有 runtime commit 原子提交的 user message。"""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    submission_id: str
    message: Msg

    @field_validator("submission_id")
    @classmethod
    def _validate_submission_id(cls, value: str, info: ValidationInfo) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} 不能为空")
        return normalized

    @model_validator(mode="after")
    def _validate_message(self) -> SteeringInput:
        if self.message.role is not Role.USER:
            raise ValueError("steering input 必须包含 user message")
        return self


class RuntimeSteeringPort(Protocol):
    """在 exact activation safe boundary 提供至多一条瞬时输入。"""

    async def claim(
        self,
        run_id: str,
        activation_id: str,
    ) -> SteeringInput | None:
        """Claim 当前 boundary 可投递的一条 exact-target input。

        Args:
            run_id (str): 当前 logical run identity。
            activation_id (str): 当前 activation identity。

        Returns:
            SteeringInput | None: 一条待提交输入；当前 boundary 无输入时为 None。
        """

    def acknowledge(self, submission_id: str) -> None:
        """确认对应 input 已随 durable commit 成功投递。

        Args:
            submission_id (str): 最近一次成功 claim 的 submission identity。
        """

    def fail(self, submission_id: str, reason: str) -> None:
        """确认对应 input 未进入 durable history。

        Args:
            submission_id (str): 最近一次成功 claim 的 submission identity。
            reason (str): 未投递原因。
        """


__all__ = ["RuntimeSteeringPort", "SteeringInput"]
