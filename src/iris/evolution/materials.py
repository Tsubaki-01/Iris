"""项目经历、修订请求和发布历史的独立 SQLite 存储。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..exceptions import IrisEvolutionError
from ..utils.generation_worker import check_generation_cancelled
from ._sqlite import connection, encode, initialize
from .history import (
    EvolutionHistoryCursor,
    EvolutionHistoryPage,
    PublicationPage,
    PublicationRecord,
    PublicationSummary,
    RevisionRequestPage,
    RevisionRequestSummary,
)
from .models import (
    EvolutionCaptureBlock,
    EvolutionMaterial,
    EvolutionRange,
    EvolutionRecord,
    EvolutionResult,
    EvolutionSession,
    EvolutionSource,
    EvolutionSourceState,
    ExperienceOrigin,
    HostOrigin,
    PendingMaterials,
    RevisionItem,
    RevisionTarget,
)

_DETAIL_FIELDS = {
    "before_documents",
    "candidate_documents",
    "evidence_refs",
    "request",
    "materials",
    "proposed_issue",
}


class _Registration(BaseModel):
    """来源登记与消费位置在数据库读取边界一起校验。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: EvolutionSource
    initial_message_count: int = Field(ge=0)
    consumed_until: int = Field(ge=0)
    captured_until: int = Field(ge=0)
    observed_terminal: int | None = Field(default=None, ge=0)
    observed_outcome: str | None = None

    @model_validator(mode="after")
    def _positions(self) -> _Registration:
        if not self.initial_message_count <= self.consumed_until <= self.captured_until:
            raise ValueError("项目材料消费位置与捕获范围不一致")
        return self

    def state(self) -> EvolutionSourceState:
        """仅在连续捕获到终点后向宿主暴露封源事实。"""
        sealed = self.observed_terminal == self.captured_until
        return EvolutionSourceState(
            source=self.source,
            initial_message_count=self.initial_message_count,
            captured_until=self.captured_until,
            consumed_until=self.consumed_until,
            terminal_message_count=self.observed_terminal if sealed else None,
            outcome=self.observed_outcome if sealed else None,
        )


class _RevisionRouting(BaseModel):
    """请求调度只解析来源与目标，不携带问题正文和证据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    origin: Annotated[ExperienceOrigin | HostOrigin, Field(discriminator="kind")]
    targets: tuple[RevisionTarget, ...]


def _registration(row: sqlite3.Row) -> _Registration:
    return _Registration.model_validate(
        {
            **json.loads(row["registration_json"]),
            "consumed_until": row["consumed_until"],
            "captured_until": row["captured_until"],
            "observed_terminal": row["observed_terminal"],
            "observed_outcome": row["observed_outcome"],
        }
    )


def _source_keys(sources: frozenset[tuple[str, str]]) -> str:
    return encode([json.dumps(pair, separators=(",", ":")) for pair in sorted(sources)])


_PENDING_SOURCE = "captured_until=observed_terminal AND consumed_until < observed_terminal"


def _key(source: EvolutionSource) -> str:
    return json.dumps([source.lifecycle_source_id, source.run_id], separators=(",", ":"))


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _publication_summary(record: PublicationRecord) -> PublicationSummary:
    return PublicationSummary.model_construct(
        publication_id=record.publication_id,
        revision_id=record.revision_id,
        created_at=record.created_at,
        stage=record.stage,
        origin=record.origin,
        description=record.description,
        targets=record.targets,
        status=record.outcome.status if record.outcome is not None else None,
        publication_state=record.publication_state,
        reason=record.reason,
        published_at=record.published_at,
        settled=record.settled,
    )


def _publication(row: sqlite3.Row) -> PublicationRecord:
    return PublicationRecord.model_validate(
        {**json.loads(row["state_json"]), **json.loads(row["detail_json"])}
    )


class EvolutionMaterialStore:
    """管理独立 evolution.db；运行资格和项目锁仍由宿主与生成服务拥有。"""

    def __init__(self, workspace_root: Path) -> None:
        """绑定项目 SQLite，初始化当前结构，不迁移旧 JSON。"""
        self.root = workspace_root.resolve() / ".iris" / "evolution"
        self.path = self.root / "evolution.db"
        initialize(self.path)

    def save_publication(self, record: PublicationRecord) -> None:
        """首次插入候选档案；确认和结算只更新状态，不重写静态正文。"""
        with connection(self.path, write=True) as database:
            self._save_publication(database, record)

    @staticmethod
    def _save_publication(database: sqlite3.Connection, record: PublicationRecord) -> None:
        summary = encode(_publication_summary(record).model_dump(mode="json"))
        state = encode(record.model_dump(mode="json", exclude=_DETAIL_FIELDS))
        updated = database.execute(
            "UPDATE publications SET settled=?,summary_json=?,state_json=? WHERE id=?",
            (record.settled, summary, state, record.publication_id),
        )
        if updated.rowcount == 0:
            database.execute(
                "INSERT INTO publications VALUES (?,?,?,?,?)",
                (
                    record.publication_id,
                    _timestamp(record.created_at),
                    record.settled,
                    summary,
                    state,
                ),
            )
            database.execute(
                "INSERT INTO publication_details VALUES (?,?)",
                (
                    record.publication_id,
                    encode(record.model_dump(mode="json", include=_DETAIL_FIELDS)),
                ),
            )

    def list_publications(
        self, *, after: EvolutionHistoryCursor | None = None, limit: int = 50
    ) -> PublicationPage:
        """数据库按创建时刻分页，只读取短摘要列。"""
        return self._history_page("publications", PublicationSummary, after, limit)

    def get_publication(self, publication_id: str) -> PublicationRecord | None:
        """按主键读取完整档案，after 正文由已确认候选投影。"""
        with connection(self.path) as database:
            row = database.execute(
                "SELECT p.state_json,d.detail_json FROM publications p "
                "JOIN publication_details d ON d.publication_id=p.id WHERE p.id=?",
                (publication_id,),
            ).fetchone()
            return _publication(row) if row is not None else None

    def list_unsettled_publications(self) -> tuple[PublicationRecord, ...]:
        """仅加载尚待原 owner 收尾的完整档案。"""
        with connection(self.path) as database:
            return tuple(
                _publication(row)
                for row in database.execute(
                    "SELECT p.state_json,d.detail_json FROM publications p "
                    "JOIN publication_details d ON d.publication_id=p.id WHERE p.settled=0 "
                    "ORDER BY p.created_at,p.id"
                )
            )

    def list_revision_requests(
        self, *, after: EvolutionHistoryCursor | None = None, limit: int = 50
    ) -> RevisionRequestPage:
        """读取请求摘要，正文与证据留给详情接口。"""
        return self._history_page("requests", RevisionRequestSummary, after, limit)

    def get_revision_request(self, revision_id: str) -> RevisionItem | None:
        """按原始 ID 读取完整请求，结算后仍保留历史。"""
        with connection(self.path) as database:
            row = database.execute(
                "SELECT payload FROM requests WHERE id=?", (revision_id,)
            ).fetchone()
            return RevisionItem.model_validate_json(row[0]) if row is not None else None

    def _history_page[SummaryT: BaseModel](
        self, table: str, schema: type[SummaryT], after: EvolutionHistoryCursor | None, limit: int
    ) -> EvolutionHistoryPage[SummaryT]:
        if not 1 <= limit <= 100:
            raise IrisEvolutionError("历史分页 limit 必须在 1 到 100 之间")
        where = "" if after is None else " WHERE (created_at,id) > (?,?)"
        params = () if after is None else (_timestamp(after.created_at), after.id)
        with connection(self.path) as database:
            rows = database.execute(
                f"SELECT id,created_at,summary_json FROM {table}{where} "
                "ORDER BY created_at,id LIMIT ?",
                (*params, limit + 1),
            ).fetchall()
            page = rows[:limit]
            return EvolutionHistoryPage(
                tuple(schema.model_validate_json(row["summary_json"]) for row in page),
                EvolutionHistoryCursor(
                    datetime.fromisoformat(page[-1]["created_at"]), page[-1]["id"]
                )
                if len(rows) > limit
                else None,
            )

    def register_source(
        self, source: EvolutionSource, initial_message_count: int
    ) -> EvolutionSourceState:
        """原子登记来源，重复登记不重置消费或捕获位置。"""
        with connection(self.path, write=True) as database:
            database.execute(
                "INSERT OR IGNORE INTO sources "
                "(source_key,registration_json,consumed_until,lifecycle_source_id,captured_until) "
                "VALUES (?,?,?,?,?)",
                (
                    _key(source),
                    encode(
                        {
                            "source": source.model_dump(mode="json"),
                            "initial_message_count": initial_message_count,
                        }
                    ),
                    initial_message_count,
                    source.lifecycle_source_id,
                    initial_message_count,
                ),
            )
            return self._read_source(database, _key(source)).state()

    def list_capture_sources(self, lifecycle_source_id: str) -> tuple[EvolutionSourceState, ...]:
        """只返回同一 reader 尚未连续封源的来源。"""
        with connection(self.path) as database:
            return tuple(
                _registration(row).state()
                for row in database.execute(
                    "SELECT * FROM sources WHERE lifecycle_source_id=? AND "
                    "(observed_terminal IS NULL OR captured_until < observed_terminal) "
                    "ORDER BY source_key",
                    (lifecycle_source_id,),
                )
            )

    def commit_capture(self, block: EvolutionCaptureBlock) -> EvolutionSourceState:
        """正文和收据同事务发布，允许跨进程重叠捕获。"""
        with connection(self.path, write=True) as database:
            key = _key(block.source)
            current = self._read_source(database, key)
            database.execute(
                "INSERT INTO captures VALUES (?,?,?,?,?,?)",
                (
                    uuid4().hex,
                    key,
                    block.start_message_count,
                    block.end_message_count,
                    encode(block.model_copy(update={"records": ()}).model_dump(mode="json")),
                    encode(block.model_dump(mode="json"))
                    if block.end_message_count > current.consumed_until
                    else None,
                ),
            )
            position = current.captured_until
            for row in database.execute(
                "SELECT start_count,end_count FROM captures WHERE source_key=? AND end_count>? "
                "ORDER BY start_count,end_count",
                (key, position),
            ):
                if row["start_count"] > position:
                    break
                position = max(position, row["end_count"])
            terminal = current.observed_terminal
            outcome = current.observed_outcome
            if block.terminal_message_count is not None:
                terminal, outcome = block.terminal_message_count, block.outcome
            database.execute(
                "UPDATE sources SET captured_until=?,observed_terminal=?,observed_outcome=? "
                "WHERE source_key=?",
                (position, terminal, outcome, key),
            )
            return current.model_copy(
                update={
                    "captured_until": position,
                    "observed_terminal": terminal,
                    "observed_outcome": outcome,
                }
            ).state()

    def list_pending_sources(self) -> tuple[EvolutionSource, ...]:
        """合并已封源未消费的材料与待处理经历问题的来源。"""
        with connection(self.path) as database:
            sources = {
                row["source_key"]: _registration(row).source
                for row in database.execute(
                    f"SELECT * FROM sources WHERE {_PENDING_SOURCE} ORDER BY source_key"
                )
            }
            for _, item in self._pending_routing(database):
                if isinstance(item.origin, ExperienceOrigin):
                    sources.update((_key(source), source) for source in item.origin.sources)
            return tuple(sources.values())

    def list_pending_sessions(self) -> tuple[EvolutionSession, ...]:
        """列出未结算宿主请求的真实会话身份。"""
        with connection(self.path) as database:
            return tuple(
                dict.fromkeys(
                    item.origin.session
                    for _, item in self._pending_routing(database)
                    if isinstance(item.origin, HostOrigin) and item.origin.session is not None
                )
            )

    def enqueue_revision(self, item: RevisionItem) -> None:
        """一条记录同时承担请求队列和历史，不复制两份请求正文。"""
        with connection(self.path, write=True) as database:
            self._insert_request(database, item)

    @staticmethod
    def _insert_request(database: sqlite3.Connection, item: RevisionItem) -> None:
        summary = RevisionRequestSummary.model_construct(
            id=item.id,
            created_at=item.created_at,
            description=item.description,
            targets=item.targets,
            origin=item.origin.kind,
            status=None,
        )
        database.execute(
            "INSERT INTO requests (id,created_at,summary_json,payload,routing_json) "
            "VALUES (?,?,?,?,?)",
            (
                item.id,
                _timestamp(item.created_at),
                encode(summary.model_dump(mode="json")),
                encode(item.model_dump(mode="json")),
                encode(item.model_dump(mode="json", include={"origin", "targets"})),
            ),
        )

    @staticmethod
    def _pending_routing(
        database: sqlite3.Connection, requested_revision_id: str | None = None
    ) -> Iterator[tuple[str, _RevisionRouting]]:
        if requested_revision_id is not None:
            row = database.execute(
                "SELECT id,routing_json FROM requests WHERE id=? AND result_json IS NULL",
                (requested_revision_id,),
            ).fetchone()
            if row is not None:
                yield row["id"], _RevisionRouting.model_validate_json(row["routing_json"])
        cursor: tuple[str, str] | None = None
        while True:
            clause = "" if cursor is None else " AND (created_at,id) > (?,?)"
            rows = database.execute(
                "SELECT id,created_at,routing_json FROM requests WHERE result_json IS NULL"
                + clause
                + " ORDER BY created_at,id LIMIT 64",
                () if cursor is None else cursor,
            ).fetchall()
            if not rows:
                return
            for row in rows:
                if row["id"] != requested_revision_id:
                    yield row["id"], _RevisionRouting.model_validate_json(row["routing_json"])
            cursor = rows[-1]["created_at"], rows[-1]["id"]

    @staticmethod
    def _eligible_revisions(
        database: sqlite3.Connection,
        *,
        allowed_sources: frozenset[tuple[str, str]],
        allowed_sessions: frozenset[tuple[str, str]],
        allowed_targets: frozenset[tuple[str, str]],
        requested_revision_id: str | None = None,
    ) -> Iterator[str]:
        for identity, item in EvolutionMaterialStore._pending_routing(
            database, requested_revision_id
        ):
            origin = item.origin
            if isinstance(origin, ExperienceOrigin):
                allowed = all(
                    (source.lifecycle_source_id, source.run_id) in allowed_sources
                    for source in origin.sources
                )
            else:
                allowed = (
                    origin.session is None
                    or (
                        origin.session.lifecycle_source_id,
                        origin.session.session_id,
                    )
                    in allowed_sessions
                )
            if allowed and all(
                (target.kind, target.name) in allowed_targets for target in item.targets
            ):
                yield identity

    def read_pending_revisions(
        self,
        *,
        allowed_sources: frozenset[tuple[str, str]],
        allowed_sessions: frozenset[tuple[str, str]],
        allowed_targets: frozenset[tuple[str, str]],
        limit: int = 1,
        requested_revision_id: str | None = None,
    ) -> tuple[RevisionItem, ...]:
        """来源和目标过滤先于限额，显式请求优先。"""
        with connection(self.path) as database:
            eligible = []
            for identity in self._eligible_revisions(
                database,
                allowed_sources=allowed_sources,
                allowed_sessions=allowed_sessions,
                allowed_targets=allowed_targets,
                requested_revision_id=requested_revision_id,
            ):
                row = database.execute(
                    "SELECT payload FROM requests WHERE id=?", (identity,)
                ).fetchone()
                eligible.append(RevisionItem.model_validate_json(row[0]))
                if len(eligible) == limit:
                    break
            return tuple(eligible)

    def has_pending_revisions(
        self,
        *,
        allowed_sources: frozenset[tuple[str, str]],
        allowed_sessions: frozenset[tuple[str, str]],
        allowed_targets: frozenset[tuple[str, str]],
    ) -> bool:
        """只查询合格请求是否存在，不加载问题正文或证据。"""
        with connection(self.path) as database:
            return (
                next(
                    self._eligible_revisions(
                        database,
                        allowed_sources=allowed_sources,
                        allowed_sessions=allowed_sessions,
                        allowed_targets=allowed_targets,
                    ),
                    None,
                )
                is not None
            )

    def has_pending_materials(self, *, allowed_sources: frozenset[tuple[str, str]]) -> bool:
        """只查询已连续封源且未消费的来源，不读取材料正文。"""
        with connection(self.path) as database:
            return (
                database.execute(
                    f"SELECT 1 FROM sources WHERE {_PENDING_SOURCE} "
                    "AND source_key IN (SELECT value FROM json_each(?)) LIMIT 1",
                    (_source_keys(allowed_sources),),
                ).fetchone()
                is not None
            )

    def settle_revision(self, item_id: str, step: EvolutionResult) -> None:
        """原子记录请求结果与最近步骤，保留请求正文供历史读取。"""
        with connection(self.path, write=True) as database:
            self._settle_revision(database, item_id, step)

    def _settle_revision(
        self, database: sqlite3.Connection, item_id: str, step: EvolutionResult
    ) -> None:
        row = database.execute(
            "SELECT summary_json FROM requests WHERE id=?", (item_id,)
        ).fetchone()
        summary = RevisionRequestSummary.model_validate_json(row[0]).model_copy(
            update={"status": step.status}
        )
        database.execute(
            "UPDATE requests SET result_json=?,summary_json=? WHERE id=?",
            (
                encode(step.model_dump(mode="json")),
                encode(summary.model_dump(mode="json")),
                item_id,
            ),
        )
        self._record_step(database, step)

    def revision_result(self, item_id: str) -> EvolutionResult | None:
        """读取跨进程可见的最终请求结果。"""
        with connection(self.path) as database:
            row = database.execute(
                "SELECT result_json FROM requests WHERE id=?", (item_id,)
            ).fetchone()
            return (
                EvolutionResult.model_validate_json(row[0]) if row and row[0] is not None else None
            )

    def read_pending(
        self, *, allowed_sources: frozenset[tuple[str, str]], limit: int = 128
    ) -> PendingMaterials:
        """在同一快照按合格来源读取完整消息，重叠区间只出现一次。"""
        with connection(self.path) as database:
            items: list[EvolutionMaterial] = []
            for row in database.execute(
                f"SELECT * FROM sources WHERE {_PENDING_SOURCE} "
                "AND source_key IN (SELECT value FROM json_each(?)) ORDER BY source_key",
                (_source_keys(allowed_sources),),
            ):
                if len(items) == limit:
                    return PendingMaterials(tuple(items), True)
                state = _registration(row).state()
                key = row["source_key"]
                through = min(
                    cast(int, state.terminal_message_count),
                    state.consumed_until + limit - len(items),
                )
                messages: dict[int, tuple[EvolutionRecord, ...]] = {}
                for capture in database.execute(
                    "SELECT body_json FROM captures WHERE source_key=? AND end_count>? "
                    "AND start_count<? ORDER BY id",
                    (key, state.consumed_until, through),
                ):
                    check_generation_cancelled()
                    block = EvolutionCaptureBlock.model_validate_json(capture[0])
                    grouped: dict[int, list[EvolutionRecord]] = {}
                    for record in block.records:
                        grouped.setdefault(record.message_ordinal, []).append(record)
                    for ordinal in range(
                        max(state.consumed_until, block.start_message_count),
                        min(through, block.end_message_count),
                    ):
                        messages.setdefault(ordinal, tuple(grouped.get(ordinal, ())))
                    if len(messages) == through - state.consumed_until:
                        break
                for ordinal in sorted(messages):
                    items.append(
                        EvolutionMaterial.model_construct(
                            source=state.source,
                            start_message_count=ordinal,
                            end_message_count=ordinal + 1,
                            records=messages[ordinal],
                        )
                    )
                if through < cast(int, state.terminal_message_count):
                    return PendingMaterials(tuple(items), True)
            return PendingMaterials(tuple(items), False)

    def consume(
        self,
        selected: tuple[EvolutionMaterial, ...],
        step: EvolutionResult,
        *,
        issue: RevisionItem | None = None,
    ) -> None:
        """消费进度、提炼请求与正文清理在一个事务中完成。"""
        with connection(self.path, write=True) as database:
            self._consume(database, selected, step, issue=issue)

    def _consume(
        self,
        database: sqlite3.Connection,
        selected: tuple[EvolutionMaterial, ...],
        step: EvolutionResult,
        *,
        issue: RevisionItem | None,
    ) -> None:
        if (
            step.publication_id is not None
            and database.execute(
                "SELECT 1 FROM consumed_publications WHERE publication_id=?",
                (step.publication_id,),
            ).fetchone()
        ):
            return
        consumed: dict[str, int] = {}
        for item in selected:
            key = _key(item.source)
            if key not in consumed:
                consumed[key] = self._read_source(database, key).consumed_until
            if item.start_message_count != consumed[key]:
                raise IrisEvolutionError(
                    "项目材料消费范围不连续",
                    run_id=item.source.run_id,
                    consumed_until=consumed[key],
                )
            consumed[key] = item.end_message_count
        for key, position in consumed.items():
            database.execute(
                "UPDATE sources SET consumed_until=? WHERE source_key=?",
                (position, key),
            )
        if issue is not None:
            self._insert_request(database, issue)
        if step.publication_id is not None:
            database.execute("INSERT INTO consumed_publications VALUES (?)", (step.publication_id,))
        self._record_step(database, step)
        self._clean_consumed_bodies(database, consumed)

    def settle_publication(self, record: PublicationRecord) -> EvolutionResult:
        """把已确认发布的材料或请求结算与档案收尾原子提交，不重放文件写入。"""
        result = cast(EvolutionResult, record.outcome)
        with connection(self.path, write=True) as database:
            if result.status in {"updated", "no_change"}:
                if record.stage == "revision":
                    self._settle_revision(database, cast(str, record.revision_id), result)
                else:
                    result = result.model_copy(
                        update={
                            "consumed_ranges": tuple(
                                EvolutionRange.model_construct(
                                    source=item.source,
                                    start_message_count=item.start_message_count,
                                    end_message_count=item.end_message_count,
                                )
                                for item in record.materials
                            )
                        }
                    )
                    self._consume(database, record.materials, result, issue=record.proposed_issue)
            else:
                self._record_step(database, result)
            self._save_publication(
                database,
                record.model_copy(
                    update={
                        "settled": True,
                        "outcome": result,
                        "consumed_ranges": result.consumed_ranges,
                    }
                ),
            )
        return result

    @staticmethod
    def _clean_consumed_bodies(database: sqlite3.Connection, consumed: dict[str, int]) -> None:
        database.executemany(
            "UPDATE captures SET body_json=NULL WHERE source_key=? AND end_count<=? "
            "AND body_json IS NOT NULL",
            consumed.items(),
        )

    def record_step(self, step: EvolutionResult) -> None:
        """记录简短结果，失败不推进消费位置。"""
        with connection(self.path, write=True) as database:
            self._record_step(database, step)

    @staticmethod
    def _record_step(database: sqlite3.Connection, step: EvolutionResult) -> None:
        database.execute(
            "INSERT INTO progress VALUES (1,?) ON CONFLICT(id) DO UPDATE SET "
            "latest_step=excluded.latest_step",
            (encode(step.model_dump(mode="json")),),
        )

    @staticmethod
    def _read_source(database: sqlite3.Connection, key: str) -> _Registration:
        return _registration(
            database.execute("SELECT * FROM sources WHERE source_key=?", (key,)).fetchone()
        )
