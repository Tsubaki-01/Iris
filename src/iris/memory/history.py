"""Memory 历史分页与文件发布产物的持久模型。"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .generation_models import GenerationResult
from .models import MemoryEpisode, _new_id, _now_iso


class MemoryPublicationDocument(BaseModel):
    """发布 owner 确认写入的完整文档正文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    text: str


class MemoryPublicationRecord(BaseModel):
    """一次概览或分类发布的结果，部分文件成功不等于完整投影成功。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    publication_id: str = Field(default_factory=_new_id)
    kind: Literal["projection", "overview"]
    namespace: str
    item_revision: int
    status: Literal["published", "failed", "conflict", "unconfirmed"]
    projection_revision: int | None = None
    generation_result_id: str | None = None
    documents: tuple[MemoryPublicationDocument, ...] = ()
    error: str | None = None
    created_at: str = Field(default_factory=_now_iso)


@dataclass(frozen=True, slots=True)
class MemoryHistoryCursor:
    """按创建时间和原始 ID 升序读取的稳定位置。"""

    created_at: str
    id: str


@dataclass(frozen=True, slots=True)
class MemoryHistoryPage[T]:
    """领域历史的一页实际记录及后续位置。"""

    items: tuple[T, ...]
    next_cursor: MemoryHistoryCursor | None


type EpisodePage = MemoryHistoryPage[MemoryEpisode]
type GenerationResultPage = MemoryHistoryPage[GenerationResult]
type MemoryPublicationPage = MemoryHistoryPage[MemoryPublicationRecord]
