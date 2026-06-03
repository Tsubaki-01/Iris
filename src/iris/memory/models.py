"""长期记忆内核数据模型。

本模块只定义 Stage 1 的 SDK 与持久化边界模型，不包含 mirror、工具或编排器逻辑。

Example:
    scope = MemoryScope(workspace_id="workspace", agent_id="agent", collection="default")
    item = MemoryItem(scope=scope, text="用户偏好简洁回答")
"""

# region imports
from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

# endregion


def _new_id() -> str:
    """生成无外部依赖的稳定字符串 ID。"""
    return uuid.uuid4().hex


def _now_iso() -> str:
    """生成与现有 SQLite session store 风格一致的时间戳。"""
    return datetime.now().isoformat()


class MemoryVisibility(StrEnum):
    """记忆可见范围。"""

    SESSION = "session"
    AGENT = "agent"
    WORKSPACE = "workspace"


class MemoryLevel(StrEnum):
    """分层记忆级别。"""

    EPISODIC = "l1"
    SEMANTIC = "l2"


class MemoryCategory(StrEnum):
    """记忆目录类别。"""

    USER = "user"
    FEEDBACK = "feedback"
    REFERENCE = "reference"
    TASK = "task"
    SESSION = "session"


class MemorySourceType(StrEnum):
    """记忆来源类型。"""

    MESSAGE = "message"
    TOOL_EVENT = "tool_event"
    ARTIFACT = "artifact"
    TASK = "task"
    REFERENCE = "reference"
    SDK = "sdk"


class MemoryItemKind(StrEnum):
    """长期记忆条目类型。"""

    FACT = "fact"
    PREFERENCE = "preference"
    NOTE = "note"
    SUMMARY = "summary"
    TASK_STATE = "task_state"
    CORRECTION = "correction"


class MemoryItemStatus(StrEnum):
    """长期记忆条目状态。"""

    ACTIVE = "active"
    DELETED = "deleted"
    SUPERSEDED = "superseded"


class MemoryCandidateStatus(StrEnum):
    """候选记忆状态。"""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MERGED = "merged"


class MemoryEventType(StrEnum):
    """记忆审计事件类型。"""

    OBSERVE = "observe"
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    SUPERSEDE = "supersede"
    SEARCH = "search"
    CONTEXT_INCLUDE = "context_include"
    CANDIDATE_ADD = "candidate_add"
    CANDIDATE_ACCEPT = "candidate_accept"
    CANDIDATE_REJECT = "candidate_reject"
    CANDIDATE_MERGE = "candidate_merge"


class MemoryActor(StrEnum):
    """触发记忆操作的角色。"""

    SDK = "sdk"
    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"


class WorkingMemoryFrame(BaseModel):
    """运行态 L0 工作记忆帧。预留数据模型，用于agent runtime"""

    scope: MemoryScope
    task: str = ""
    messages: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryScope(BaseModel):
    """记忆隔离边界。

    `workspace_id`、`agent_id` 与 `collection` 共同组成基础隔离键；
    session 可见性额外要求 `session_id`。
    """

    workspace_id: str
    agent_id: str
    collection: str = "default"
    visibility: MemoryVisibility = MemoryVisibility.AGENT
    session_id: str | None = None

    model_config = {"use_enum_values": False}

    @field_validator("workspace_id", "agent_id", "collection", "session_id")
    @classmethod
    def _validate_non_empty_text(cls, value: str | None) -> str | None:
        """校验 scope 文本字段不能是空白。"""
        if value is not None and not value.strip():
            raise ValueError("记忆 scope 字段不能为空")
        return value

    @model_validator(mode="after")
    def _validate_session_visibility(self) -> Self:
        """校验 session 可见性必须绑定 session_id。"""
        if self.visibility == MemoryVisibility.SESSION and self.session_id is None:
            raise ValueError("session 可见记忆必须提供 session_id")
        return self


class MemoryArtifactRef(BaseModel):
    """记忆关联产物的本地相对引用。"""

    path: str
    mime_type: str = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        """拒绝绝对路径，避免 Stage 1 模型层泄露主机路径。"""
        if not value.strip():
            raise ValueError("artifact path 不能为空")
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError("artifact path 必须是相对路径")
        return value


class MemoryEpisode(BaseModel):
    """L1 片段记忆，记录一次观察到的事实来源。"""

    id: str = Field(default_factory=_new_id)
    scope: MemoryScope
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    text: str = ""
    category: MemoryCategory = MemoryCategory.SESSION
    artifacts: list[MemoryArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)

    model_config = {"use_enum_values": False}


class MemoryItem(BaseModel):
    """L2 长期记忆条目。"""

    id: str = Field(default_factory=_new_id)
    scope: MemoryScope
    text: str
    level: MemoryLevel = MemoryLevel.SEMANTIC
    category: MemoryCategory = MemoryCategory.USER
    kind: MemoryItemKind = MemoryItemKind.NOTE
    status: MemoryItemStatus = MemoryItemStatus.ACTIVE
    episode_id: str | None = None
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    reason: str = ""
    confidence: float | None = None
    importance: float | None = None
    artifacts: list[MemoryArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    deleted_at: str | None = None

    model_config = {"use_enum_values": False}

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        """校验记忆正文不能为空。"""
        if not value.strip():
            raise ValueError("记忆正文不能为空")
        return value

    @field_validator("confidence", "importance")
    @classmethod
    def _validate_score(cls, value: float | None) -> float | None:
        """校验置信度与重要性分数范围。"""
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("记忆分数必须在 0.0 到 1.0 之间")
        return value


class MemoryCandidate(BaseModel):
    """从 L1 episode 抽取出的待处理候选记忆。"""

    id: str = Field(default_factory=_new_id)
    scope: MemoryScope
    episode_ids: list[str] = Field(default_factory=list)
    category: MemoryCategory = MemoryCategory.USER
    suggested_level: MemoryLevel = MemoryLevel.SEMANTIC
    text: str
    confidence: float | None = None
    importance: float | None = None
    reason: str
    status: MemoryCandidateStatus = MemoryCandidateStatus.PENDING
    created_at: str = Field(default_factory=_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": False}

    @field_validator("episode_ids")
    @classmethod
    def _validate_episode_ids(cls, value: list[str]) -> list[str]:
        """校验候选记忆必须可追溯到至少一个 episode。"""
        if not value:
            raise ValueError("候选记忆必须包含 episode id")
        if any(not episode_id.strip() for episode_id in value):
            raise ValueError("候选记忆 episode id 不能为空")
        return value

    @field_validator("text", "reason")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        """校验候选正文与原因不能为空。"""
        if not value.strip():
            raise ValueError("候选记忆正文和原因不能为空")
        return value

    @field_validator("confidence", "importance")
    @classmethod
    def _validate_score(cls, value: float | None) -> float | None:
        """校验候选分数范围。"""
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("候选记忆分数必须在 0.0 到 1.0 之间")
        return value


class MemoryItemPatch(BaseModel):
    """长期记忆条目的部分更新。"""

    text: str | None = None
    category: MemoryCategory | None = None
    kind: MemoryItemKind | None = None
    status: MemoryItemStatus | None = None
    confidence: float | None = None
    importance: float | None = None
    artifacts: list[MemoryArtifactRef] | None = None
    metadata: dict[str, Any] | None = None

    model_config = {"use_enum_values": False}

    @field_validator("text")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        """校验更新正文不能是空白。"""
        if value is not None and not value.strip():
            raise ValueError("记忆正文不能为空")
        return value

    @field_validator("confidence", "importance")
    @classmethod
    def _validate_optional_score(cls, value: float | None) -> float | None:
        """校验可选分数范围。"""
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("记忆分数必须在 0.0 到 1.0 之间")
        return value


class MemoryEvent(BaseModel):
    """记忆审计事件。"""

    id: str = Field(default_factory=_new_id)
    scope: MemoryScope
    event_type: MemoryEventType
    actor: MemoryActor = MemoryActor.SDK
    item_id: str | None = None
    episode_id: str | None = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)

    model_config = {"use_enum_values": False}


class MemoryQuery(BaseModel):
    """长期记忆召回查询。"""

    scope: MemoryScope
    text: str = ""
    item_ids: list[str] = Field(default_factory=list)
    categories: list[MemoryCategory] = Field(default_factory=list)
    kinds: list[MemoryItemKind] = Field(default_factory=list)
    limit: int = 10
    include_deleted: bool = False

    model_config = {"use_enum_values": False}

    @field_validator("limit")
    @classmethod
    def _validate_limit(cls, value: int) -> int:
        """校验并限制召回数量上限。"""
        if value <= 0:
            raise ValueError("记忆查询 limit 必须为正数")
        return min(value, 100)


class MemorySearchResult(BaseModel):
    """一次召回命中的长期记忆。"""

    item: MemoryItem
    score: float = 0.0
    source: str = "sqlite"
    matched_text: str = ""


class MemoryWriteInput(BaseModel):
    """写入长期记忆的 SDK 输入。"""

    scope: MemoryScope
    text: str
    reason: str
    category: MemoryCategory = MemoryCategory.USER
    kind: MemoryItemKind = MemoryItemKind.NOTE
    episode_id: str | None = None
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    actor: MemoryActor = MemoryActor.SDK
    confidence: float | None = None
    importance: float | None = None
    artifacts: list[MemoryArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": False}

    @field_validator("text", "reason")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        """校验写入正文与原因不能为空。"""
        if not value.strip():
            raise ValueError("记忆写入正文和原因不能为空")
        return value

    @field_validator("confidence", "importance")
    @classmethod
    def _validate_write_score(cls, value: float | None) -> float | None:
        """校验写入分数范围。"""
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("记忆分数必须在 0.0 到 1.0 之间")
        return value


class MemoryObserveInput(BaseModel):
    """写入 L1 观察片段的 SDK 输入。"""

    scope: MemoryScope
    text: str = ""
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    actor: MemoryActor = MemoryActor.SDK
    category: MemoryCategory = MemoryCategory.SESSION
    reason: str = ""
    artifacts: list[MemoryArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": False}


class MemoryContextFragment(BaseModel):
    """构建提示上下文时包含的一条记忆片段。"""

    item_id: str
    text: str
    score: float = 0.0
    warning: str
    source: str = ""
    truncated: bool = False


class MemoryContextBundle(BaseModel):
    """召回后可交给上游 prompt builder 使用的记忆上下文。"""

    fragments: list[MemoryContextFragment] = Field(default_factory=list)
    total_chars: int = 0
    omitted_count: int = 0
    max_chars: int
