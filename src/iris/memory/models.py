"""长期记忆内核数据模型。

本模块定义 namespace 隔离的 SDK 与持久化边界模型，不包含 mirror、工具或编排器逻辑。

Example:
    item = MemoryItem(namespace="project", text="用户偏好简洁回答")
"""

# region imports
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

# endregion


def _new_id() -> str:
    """生成无外部依赖的稳定字符串 ID。"""
    return uuid.uuid4().hex


def _now_iso() -> str:
    """生成与现有 SQLite session store 风格一致的时间戳。"""
    return datetime.now().isoformat()


class MemoryLevel(StrEnum):
    """分层记忆级别。"""

    EPISODIC = "l1"
    SEMANTIC = "l2"


class MemoryCategory(StrEnum):
    """记忆目录类别。记录记忆属于哪个业务域。"""

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
    """长期记忆条目类型。记录记忆的类别和性质。"""

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
    namespace: str = Field(default="project", pattern=r"\S")
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    text: str = ""
    category: MemoryCategory = MemoryCategory.SESSION
    artifacts: list[MemoryArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)

    model_config = {"use_enum_values": False, "extra": "forbid"}


class MemoryItem(BaseModel):
    """L2 长期记忆条目。"""

    id: str = Field(default_factory=_new_id)
    namespace: str = Field(default="project", pattern=r"\S")
    text: str
    level: MemoryLevel = MemoryLevel.SEMANTIC
    category: MemoryCategory = MemoryCategory.USER
    kind: MemoryItemKind = MemoryItemKind.NOTE
    status: MemoryItemStatus = MemoryItemStatus.ACTIVE
    episode_id: str | None = None
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    reason: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    artifacts: list[MemoryArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    deleted_at: str | None = None

    model_config = {"use_enum_values": False, "extra": "forbid"}

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        """校验记忆正文不能为空。"""
        if not value.strip():
            raise ValueError("记忆正文不能为空")
        return value


@dataclass(frozen=True, slots=True)
class MemoryNamespaceState:
    """namespace 的条目版本与最后完整正文投影版本。"""

    namespace: str
    item_revision: int = 0
    projection_revision: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryNamespaceSnapshot:
    """同一读事务得到的版本状态与完整 active L2 条目。"""

    state: MemoryNamespaceState
    items: tuple[MemoryItem, ...]


class MemoryCandidate(BaseModel):
    """从 L1 episode 抽取出的待处理候选记忆。"""

    id: str = Field(default_factory=_new_id)
    namespace: str = Field(default="project", pattern=r"\S")
    episode_ids: list[str] = Field(default_factory=list)
    category: MemoryCategory = MemoryCategory.USER
    suggested_level: MemoryLevel = MemoryLevel.SEMANTIC
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str
    status: MemoryCandidateStatus = MemoryCandidateStatus.PENDING
    created_at: str = Field(default_factory=_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": False, "extra": "forbid"}

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


class MemoryItemPatch(BaseModel):
    """长期记忆条目的部分更新，省略字段不修改，仅评分允许显式 null。"""

    text: str | SkipJsonSchema[None] = Field(default_factory=lambda: None)
    category: MemoryCategory | SkipJsonSchema[None] = Field(default_factory=lambda: None)
    kind: MemoryItemKind | SkipJsonSchema[None] = Field(default_factory=lambda: None)
    status: MemoryItemStatus | SkipJsonSchema[None] = Field(default_factory=lambda: None)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    artifacts: list[MemoryArtifactRef] | SkipJsonSchema[None] = Field(default_factory=lambda: None)
    metadata: dict[str, Any] | SkipJsonSchema[None] = Field(default_factory=lambda: None)

    model_config = {"use_enum_values": False, "extra": "forbid"}

    @field_validator("text", "category", "kind", "status", "artifacts", "metadata", mode="before")
    @classmethod
    def _reject_explicit_null(cls, value: Any) -> Any:
        """必需条目字段可省略，但显式 null 不能进入可信 patch。"""
        if value is None:
            raise ValueError("更新字段不能显式设为 null；不修改时请省略该字段")
        return value

    @field_validator("text")
    @classmethod
    def _validate_optional_text(cls, value: str) -> str:
        """校验更新正文不能是空白。"""
        if not value.strip():
            raise ValueError("记忆正文不能为空")
        return value


class MemoryEvent(BaseModel):
    """记忆审计事件。"""

    id: str = Field(default_factory=_new_id)
    namespace: str = Field(default="project", pattern=r"\S")
    event_type: MemoryEventType
    actor: MemoryActor = MemoryActor.SDK
    item_id: str | None = None
    episode_id: str | None = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)

    model_config = {"use_enum_values": False, "extra": "forbid"}


class MemoryQuery(BaseModel):
    """长期记忆召回查询。"""

    namespaces: list[Annotated[str, Field(pattern=r"\S")]] = Field(
        default_factory=lambda: ["project"], min_length=1
    )
    max_query_terms: int | None = Field(default=None, gt=0)
    text: str = ""
    item_ids: list[str] = Field(default_factory=list)
    categories: list[MemoryCategory] = Field(default_factory=list)
    kinds: list[MemoryItemKind] = Field(default_factory=list)
    limit: int = Field(default=10, gt=0, le=100)
    include_deleted: bool = False

    model_config = {"use_enum_values": False, "extra": "forbid"}


class MemorySearchResult(BaseModel):
    """一次召回命中的长期记忆。"""

    item: MemoryItem
    score: float = 0.0
    source: str = "sqlite"
    matched_text: str = ""


class MemoryWriteInput(BaseModel):
    """写入长期记忆的 SDK 输入。"""

    namespace: str = Field(default="project", pattern=r"\S")
    text: str
    reason: str
    category: MemoryCategory = MemoryCategory.USER
    kind: MemoryItemKind = MemoryItemKind.NOTE
    episode_id: str | None = None
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    actor: MemoryActor = MemoryActor.SDK
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    artifacts: list[MemoryArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": False, "extra": "forbid"}

    @field_validator("text", "reason")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        """校验写入正文与原因不能为空。"""
        if not value.strip():
            raise ValueError("记忆写入正文和原因不能为空")
        return value


class MemoryObserveInput(BaseModel):
    """写入 L1 观察片段的 SDK 输入。"""

    namespace: str = Field(default="project", pattern=r"\S")
    text: str = ""
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    actor: MemoryActor = MemoryActor.SDK
    category: MemoryCategory = MemoryCategory.SESSION
    reason: str = ""
    artifacts: list[MemoryArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": False, "extra": "forbid"}


class MemoryContextFragment(BaseModel):
    """构建提示上下文时包含的一条记忆片段。"""

    item_id: str
    namespace: str = Field(default="project", pattern=r"\S")
    text: str
    category: MemoryCategory
    kind: MemoryItemKind
    level: MemoryLevel
    reason: str = ""
    confidence: float | None = None
    importance: float | None = None
    warning: str
    truncated: bool = False


class MemoryContextBundle(BaseModel):
    """召回后可交给上游 prompt builder 使用的记忆上下文。"""

    fragments: list[MemoryContextFragment] = Field(default_factory=list)
    total_chars: int = 0
    omitted_count: int = 0
    max_chars: int


class MemoryOverviewConfig(BaseModel):
    """显式概览生成与后续窗口采用的独立预算配置。"""

    model_config = ConfigDict(extra="forbid")

    input_budget_tokens: int = Field(default=96000, gt=0)
    max_tokens: int = Field(default=1024, gt=0)
    system_budget_ratio: float = Field(default=0.02, gt=0, le=1)


class MemoryOverviewContent(BaseModel):
    """模型一次生成的核心事实与覆盖全部已知主题的知识范围。"""

    model_config = ConfigDict(extra="forbid")

    core_facts: str
    knowledge_scope: str = Field(pattern=r"\S")


@dataclass(frozen=True, slots=True)
class MemoryOverviewDocument:
    """供窗口采用的完整概览、知识范围节和读后新鲜度。"""

    namespace: str
    path: Path
    source_revision: int | None
    text: str
    navigation: str
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryOverviewGenerationResult:
    """一次显式生成的实际来源、发布结果与 provider 用量。"""

    namespace: str
    path: Path
    source_revision: int
    current_revision: int
    projection_revision: int | None
    item_count: int
    published: bool
    publication_reason: str | None
    usage: dict[str, int]
    elapsed_seconds: float
