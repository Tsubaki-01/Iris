"""原发布 owner 共用的档案与结算收据，不执行生成或重放文件写入。"""

from datetime import UTC, datetime
from pathlib import Path

from ..exceptions import IrisEvolutionError
from .history import PublicationDocument, PublicationRecord
from .materials import EvolutionMaterialStore
from .models import EvolutionResult


class PublicationJournal:
    """保存文件发布已返回的进程内事实，并在项目锁内完成原材料结算。"""

    def __init__(self, store: EvolutionMaterialStore) -> None:
        self.store = store
        self._receipts: dict[str, PublicationRecord] = {}

    def begin(self, record: PublicationRecord) -> None:
        """文件副作用前保存基线和候选；尚未确认写入。"""
        self.store.save_publication(record.model_copy(update={"publication_state": "unconfirmed"}))

    def complete(self, record: PublicationRecord, result: EvolutionResult) -> EvolutionResult:
        """只在原文件发布调用返回后记录确认，先保留内存事实再持久结算。"""
        published = result.status == "updated"
        confirmed = record.model_copy(
            update={
                "outcome": result.model_copy(update={"consumed_ranges": ()}),
                "publication_state": "confirmed" if published else "not_published",
                "published_at": datetime.now(UTC) if published else None,
                "reason": result.reason,
                "usage": result.usage,
                "effect": result.effect,
            }
        )
        self._receipts[record.publication_id] = confirmed
        return self._finish(confirmed)

    def _finish(self, record: PublicationRecord) -> EvolutionResult:
        self.store.save_publication(record)
        result = self.store.settle_publication(record)
        self._receipts.pop(record.publication_id, None)
        return result

    def failure(self, record: PublicationRecord, result: EvolutionResult) -> None:
        """生成/发布前失败没有生效正文；已发生发布的收据不被失败状态覆盖。"""
        if record.publication_id not in self._receipts:
            self.store.save_publication(
                record.model_copy(
                    update={
                        "outcome": result,
                        "publication_state": "not_published",
                        "settled": True,
                        "reason": result.reason,
                        "usage": result.usage,
                        "effect": result.effect,
                    }
                )
            )
            self.store.record_step(result)

    def resume(self) -> EvolutionResult | None:
        """先结算真实收据；重启后的未确认候选只展示观察结果，不推断过去成功。"""
        if self._receipts:
            return self._finish(next(iter(self._receipts.values())))
        records = self.store.list_unsettled_publications()
        if not records:
            return None
        record = records[0]
        if record.publication_state != "unconfirmed":
            return self._finish(record)
        observed: list[PublicationDocument] = []
        for path in dict.fromkeys(
            document.path for document in (*record.before_documents, *record.candidate_documents)
        ):
            try:
                text = Path(path).read_text(encoding="utf-8")
            except FileNotFoundError:
                text = None
            except (OSError, UnicodeError) as exc:
                raise IrisEvolutionError("未确认发布的当前正文读取失败", path=path) from exc
            observed.append(PublicationDocument(path=path, text=text))
        result = EvolutionResult(
            stage=record.stage,
            revision_id=record.revision_id,
            publication_id=record.publication_id,
            status="failed",
            error_code="publication_unconfirmed",
            reason="publication_unconfirmed：缺少原发布确认，保留当前观察正文，未重放修改。",
            targets=record.targets,
            usage=record.usage,
        )
        self.store.save_publication(
            record.model_copy(
                update={
                    "observed_documents": tuple(observed),
                    "outcome": result,
                    "reason": result.reason,
                }
            )
        )
        self.store.record_step(result)
        return result
