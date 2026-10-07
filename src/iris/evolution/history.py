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
    after_documents: tuple[PublicationDocument, ...] = ()
    observed_documents: tuple[PublicationDocument, ...] = ()
    reason: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    effect: str = ""
    published_at: datetime | None = None
    settled: bool = False
    request: RevisionItem | None = None
    materials: tuple[EvolutionMaterial, ...] = ()
    proposed_issue: RevisionItem | None = None


@dataclass(frozen=True, slots=True)
class EvolutionHistoryCursor:
    """按创建时刻和原始 ID 排序的文件位置，只用于下一页读取。"""

    key: str


@dataclass(frozen=True, slots=True)
class EvolutionHistoryPage[T]:
    """一页领域历史及下一页位置。"""

    items: tuple[T, ...]
    next_cursor: EvolutionHistoryCursor | None


type PublicationPage = EvolutionHistoryPage[PublicationRecord]
type RevisionRequestPage = EvolutionHistoryPage[RevisionItem]
