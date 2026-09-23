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
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from ._query import tokenize_text

# endregion


def _new_id() -> str:
    """生成无外部依赖的稳定字符串 ID。"""
    return uuid.uuid4().hex


def _now_iso() -> str:
    """生成与现有 SQLite session store 风格一致的时间戳。"""
    return datetime.now().isoformat()


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
    GENERATION = "generation"


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


class MemoryEventType(StrEnum):
    """记忆审计事件类型。"""

    OBSERVE = "observe"
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    SUPERSEDE = "supersede"
    SEARCH = "search"
    CONTEXT_INCLUDE = "context_include"
    FLUSH = "flush"
    DREAM = "dream"


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


class MemoryRecord(BaseModel):
    """Episode 内具有稳定标识的原始材料记录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(default_factory=_new_id)
    role: str = "sdk"
    text: str = ""
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    occurred_at: str = Field(default_factory=_now_iso)
    artifacts: tuple[MemoryArtifactRef, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryEpisode(BaseModel):
    """有明确材料边界的不可变经历，不代表已经提炼的知识。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(default_factory=_new_id)
    namespace: str = Field(default="project", pattern=r"\S")
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    records: tuple[MemoryRecord, ...] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)

    @field_validator("records")
    @classmethod
    def _unique_record_ids(cls, records: tuple[MemoryRecord, ...]) -> tuple[MemoryRecord, ...]:
        """材料边界要求原文定位无歧义。"""
        if len({record.id for record in records}) != len(records):
            raise ValueError("Episode 内原文记录 ID 必须唯一")
        return records


class MemoryEvidenceRef(BaseModel):
    """指向经历内原文片段或真实显式写入事件的证据定位。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["episode", "event"]
    source_id: str = Field(pattern=r"\S")
    record_id: str | None = None
    start: int = Field(default=0, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_locator(self) -> Self:
        """首次解析时区分原文片段和不可变写入事件定位。"""
        if self.kind == "episode":
            if not self.record_id or self.end is None or self.end <= self.start:
                raise ValueError("Episode 证据必须包含记录 ID 和非空半开区间")
        elif self.record_id is not None or self.start != 0 or self.end is not None:
            raise ValueError("Event 证据只使用事件 ID 定位")
        return self


class MemoryObservation(BaseModel):
    """从材料中提出的不可变观察，整理进度由 store 单独持有。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(default_factory=_new_id)
    namespace: str = Field(default="project", pattern=r"\S")
    text: str = Field(pattern=r"\S")
    applicability: str = ""
    category: MemoryCategory = MemoryCategory.USER
    kind: MemoryItemKind = MemoryItemKind.NOTE
    reason: str = Field(pattern=r"\S")
    evidence: tuple[MemoryEvidenceRef, ...] = Field(min_length=1)
    target_item_ids: tuple[str, ...] = ()
    generation_model: str = ""
    created_at: str = Field(default_factory=_now_iso)


class MemoryItem(BaseModel):
    """包含当前支持证据的正式长期知识。"""

    id: str = Field(default_factory=_new_id)
    namespace: str = Field(default="project", pattern=r"\S")
    text: str
    category: MemoryCategory = MemoryCategory.USER
    kind: MemoryItemKind = MemoryItemKind.NOTE
    status: MemoryItemStatus = MemoryItemStatus.ACTIVE
    superseded_by: str | None = None
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    reason: str = ""
    evidence: tuple[MemoryEvidenceRef, ...] = ()
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
    """同一读事务得到的版本状态与完整 active 正式条目。"""

    state: MemoryNamespaceState
    items: tuple[MemoryItem, ...]


class MemoryItemPatch(BaseModel):
    """长期记忆条目的部分更新，省略字段不修改，显式 null 不表示省略。"""

    text: str | SkipJsonSchema[None] = Field(default_factory=lambda: None)
    category: MemoryCategory | SkipJsonSchema[None] = Field(default_factory=lambda: None)
    kind: MemoryItemKind | SkipJsonSchema[None] = Field(default_factory=lambda: None)
    status: MemoryItemStatus | SkipJsonSchema[None] = Field(default_factory=lambda: None)
    artifacts: list[MemoryArtifactRef] | SkipJsonSchema[None] = Field(default_factory=lambda: None)
    evidence: tuple[MemoryEvidenceRef, ...] | SkipJsonSchema[None] = Field(
        default_factory=lambda: None
    )
    metadata: dict[str, Any] | SkipJsonSchema[None] = Field(default_factory=lambda: None)

    model_config = {"use_enum_values": False, "extra": "forbid"}

    @field_validator(
        "text", "category", "kind", "status", "artifacts", "metadata", "evidence", mode="before"
    )
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
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)

    model_config = {"use_enum_values": False, "extra": "forbid"}


class MemorySearchQuery(BaseModel):
    """SDK 与工具共享的搜索输入，读取范围由调用方单独绑定。"""

    model_config = ConfigDict(extra="forbid")

    query: str
    required_terms: list[str] = Field(default_factory=list)
    categories: list[MemoryCategory] = Field(default_factory=list)
    kinds: list[MemoryItemKind] = Field(default_factory=list)
    limit: int = Field(default=8, ge=1, le=100)

    @field_validator("required_terms")
    @classmethod
    def _validate_required_terms(cls, value: list[str]) -> list[str]:
        """在公共查询边界拒绝没有可索引词项的必要词组。"""
        for phrase in value:
            if not tokenize_text(phrase):
                raise ValueError("required_terms 中每个词组必须包含可索引字符")
        return value


@dataclass(frozen=True, slots=True)
class MemorySearchHit:
    """一条搜索命中的最小定位信息与原文片段。"""

    item_id: str
    namespace: str
    category: MemoryCategory
    kind: MemoryItemKind
    snippet: str
    is_complete: bool


@dataclass(frozen=True, slots=True)
class MemorySearchResponse:
    """按相关度排序的有限命中，以及是否仍有候选。"""

    items: tuple[MemorySearchHit, ...]
    has_more: bool


class MemoryWriteInput(BaseModel):
    """写入长期记忆的 SDK 输入。"""

    namespace: str = Field(default="project", pattern=r"\S")
    text: str
    reason: str
    category: MemoryCategory = MemoryCategory.USER
    kind: MemoryItemKind = MemoryItemKind.NOTE
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    actor: MemoryActor = MemoryActor.SDK
    evidence: tuple[MemoryEvidenceRef, ...] = ()
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
    """独立 SDK 的经历取材输入。"""

    namespace: str = Field(default="project", pattern=r"\S")
    text: str = ""
    source_type: MemorySourceType = MemorySourceType.SDK
    source_id: str = ""
    actor: MemoryActor = MemoryActor.SDK
    records: tuple[MemoryRecord, ...] = ()
    reason: str = ""
    artifacts: list[MemoryArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": False, "extra": "forbid"}


class MemoryOverviewConfig(BaseModel):
    """显式概览生成与后续窗口采用的独立预算配置。"""

    model_config = ConfigDict(extra="forbid")

    input_budget_tokens: int = Field(default=96000, gt=0)
    max_tokens: int = Field(default=4096, gt=0)
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
