"""宿主每步提供的临时运行态快照与消息渲染。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..message import Msg
from .builder import CONTEXT_SENDER


@dataclass(frozen=True, slots=True)
class ContextBuildScope:
    """一次主模型步骤的宿主采集范围。"""

    session_id: str
    run_id: str
    step_index: int
    workspace_root: Path
    run_input: str


@dataclass(frozen=True, slots=True)
class ContextContribution:
    """宿主声明的当前材料；仅显式 optional 内容可以按优先级移除。"""

    key: str
    text: str
    required: bool = True
    priority: int = 100


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """本步骤完整当前状态；key 在同一快照内由 source 保证唯一。"""

    contributions: tuple[ContextContribution, ...] = ()


class ContextSource(Protocol):
    """绑定 runner 的宿主采集接口，允许不同 session 并发调用。"""

    async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
        """返回本步骤的完整快照，不继承上一步未列出的状态。"""


def render_context_snapshot(snapshot: ContextSnapshot) -> Msg:
    """将已选快照渲染为尾部请求消息，不写入会话历史。"""
    contents = "\n\n".join(f"[{item.key}]\n{item.text}" for item in snapshot.contributions)
    return Msg.user(
        "以下是宿主为本步骤提供的当前运行态快照。它完整替换此前的运行态条目；"
        "历史叙述不自动代表当前状态，也不改变用户原始任务。\n\n"
        + (contents or "当前无已提供的运行态条目。"),
        sender=CONTEXT_SENDER,
        metadata={"context_kind": "runtime_snapshot"},
    )
