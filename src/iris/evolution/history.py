"""项目发布的完整领域档案与只读分页投影。"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    EvolutionMaterial,
    EvolutionRange,
    EvolutionResult,
    EvolutionStatus,
    RevisionEvidence,
    RevisionItem,
    RevisionTarget,
)


class PublicationDocument(BaseModel):
    """当时读取、候选或确认写入的正文；None 明确表示文件缺失。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    text: str | None


class PublicationRecord(BaseModel):
    """一次原发布 owner 的基线、候选、真实结果及材料结算事实。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    publication_id: str = Field(default_factory=lambda: uuid4().hex)
    revision_id: str | None = None
    stage: Literal["experience", "revision"]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    outcome: EvolutionResult | None = None
    publication_state: Literal["not_published", "confirmed", "unconfirmed"] = "not_published"
    origin: Literal["host_request", "experience"]
    description: str
    evidence_refs: tuple[RevisionEvidence, ...] = ()
    consumed_ranges: tuple[EvolutionRange, ...] = ()
    targets: tuple[RevisionTarget, ...] = ()
    before_documents: tuple[PublicationDocument, ...] = ()
    candidate_documents: tuple[PublicationDocument, ...] = ()
    observed_documents: tuple[PublicationDocument, ...] = ()
    reason: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    effect: str = ""
    published_at: datetime | None = None
    settled: bool = False
    request: RevisionItem | None = None
    materials: tuple[EvolutionMaterial, ...] = ()
    proposed_issue: RevisionItem | None = None

    @property
    def after_documents(self) -> tuple[PublicationDocument, ...]:
        """仅从原发布确认投影已写正文，不再次序列化候选副本。"""
        return self.candidate_documents if self.publication_state == "confirmed" else ()


class PublicationSummary(BaseModel):
    """历史列表的短元数据，不读取或携带正文、材料与证据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    publication_id: str
    revision_id: str | None
    created_at: datetime
    stage: Literal["experience", "revision"]
    origin: Literal["host_request", "experience"]
    description: str
    targets: tuple[RevisionTarget, ...]
    status: EvolutionStatus | None
    publication_state: Literal["not_published", "confirmed", "unconfirmed"]
    reason: str
    published_at: datetime | None
    settled: bool


class RevisionRequestSummary(BaseModel):
    """修订请求列表的短元数据，status 仅表示请求的最终结算结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    created_at: datetime
    description: str
    targets: tuple[RevisionTarget, ...]
    origin: Literal["host", "experience"]
    status: EvolutionStatus | None = None


@dataclass(frozen=True, slots=True)
class EvolutionHistoryCursor:
    """按创建时刻和原始 ID 排序的数据库游标。"""

    created_at: datetime
    id: str


@dataclass(frozen=True, slots=True)
class EvolutionHistoryPage[T]:
    """一页领域历史及下一页位置。"""

    items: tuple[T, ...]
    next_cursor: EvolutionHistoryCursor | None


type PublicationPage = EvolutionHistoryPage[PublicationSummary]
type RevisionRequestPage = EvolutionHistoryPage[RevisionRequestSummary]
