"""项目经历的独立捕获块、封源事实与项目锁内消费进度。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..exceptions import IrisEvolutionError
from ..utils.files import atomic_write_text
from ..utils.generation_worker import check_generation_cancelled
from .models import (
    EvolutionCaptureBlock,
    EvolutionMaterial,
    EvolutionRecord,
    EvolutionResult,
    EvolutionSession,
    EvolutionSource,
    EvolutionSourceState,
    ExperienceOrigin,
    HostOrigin,
    PendingMaterials,
    RevisionItem,
)


class _Registration(BaseModel):
    """不可变来源登记，不保存共享的捕获水位。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: EvolutionSource
    initial_message_count: int = Field(ge=0)


class _Progress(BaseModel):
    """只在项目锁内替换的消费位置与最近一次简短结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    consumed: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    latest_step: EvolutionResult | None = None
    issues: dict[str, RevisionItem] = Field(default_factory=dict)
    settled_revisions: dict[str, EvolutionResult] = Field(default_factory=dict)


def _key(source: EvolutionSource) -> str:
    """编码稳定的 lifecycle/run 身份，session 保留在来源中。"""
    return json.dumps([source.lifecycle_source_id, source.run_id], separators=(",", ":"))


def _read[ModelT: BaseModel](path: Path, schema: type[ModelT]) -> ModelT:
    """持久 JSON 回到可信域时完整解析一次。"""
    try:
        return schema.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise IrisEvolutionError("项目材料读取失败", path=str(path), error=str(exc)) from exc


def _write(path: Path, model: BaseModel) -> None:
    """在持久化出口序列化一次，再完整发布。"""
    try:
        content = json.dumps(model.model_dump(mode="json"), ensure_ascii=False, allow_nan=False)
        atomic_write_text(path, content)
    except (OSError, TypeError, ValueError) as exc:
        raise IrisEvolutionError("项目材料发布失败", path=str(path), error=str(exc)) from exc


class EvolutionMaterialStore:
    """保存自有材料；资格和项目锁由宿主与生成服务拥有。"""

    def __init__(self, workspace_root: Path) -> None:
        """绑定项目固定 pending 目录，不创建额外数据库。"""
        self.root = workspace_root.resolve() / ".iris" / "evolution" / "pending"
        self._sources = self.root / "sources"
        self._captures = self.root / "captures"
        self._blocks = self.root / "blocks"
        self._requests = self.root / "requests"
        self._progress_path = self.root / "progress.json"

    def register_source(
        self, source: EvolutionSource, initial_message_count: int
    ) -> EvolutionSourceState:
        """首条原文前登记来源，重复登记不修改捕获或消费进度。"""
        states, _ = self._load_sources()
        if _key(source) in states:
            return states[_key(source)]
        _write(
            self._sources / f"{uuid4().hex}.json",
            _Registration.model_construct(
                source=source, initial_message_count=initial_message_count
            ),
        )
        states, _ = self._load_sources()
        return states[_key(source)]

    def list_capture_sources(self, lifecycle_source_id: str) -> tuple[EvolutionSourceState, ...]:
        """只返回同一 reader 仍需补采的来源，已封源项由待处理入口提供。"""
        states, _ = self._load_sources()
        return tuple(
            state
            for state in states.values()
            if state.source.lifecycle_source_id == lifecycle_source_id
            and state.terminal_message_count is None
        )

    def commit_capture(self, block: EvolutionCaptureBlock) -> EvolutionSourceState:
        """独立发布完整正文及收据；跨进程重叠由消费读取去重。"""
        name = f"{uuid4().hex}.json"
        body = self._blocks / name
        _write(body, block)
        _write(self._captures / name, block.model_copy(update={"records": ()}))
        states, _ = self._load_sources()
        state = states[_key(block.source)]
        if block.end_message_count <= state.consumed_until:
            self._remove_body(body)
        return state

    def list_pending_sources(self) -> tuple[EvolutionSource, ...]:
        """列出连续到终态且尚未全部消费的来源，不判断 lifecycle 资格。"""
        states, _ = self._load_sources()
        sources = {
            _key(state.source): state.source
            for state in states.values()
            if state.terminal_message_count is not None
            and state.consumed_until < state.terminal_message_count
        }
        for item in self._pending_revisions():
            if isinstance(item.origin, ExperienceOrigin):
                sources.update((_key(source), source) for source in item.origin.sources)
        return tuple(sources.values())

    def list_pending_sessions(self) -> tuple[EvolutionSession, ...]:
        """列出宿主请求携带的会话身份，实际状态仍由 lifecycle reader 判断。"""
        return tuple(
            dict.fromkeys(
                item.origin.session
                for item in self._pending_revisions()
                if isinstance(item.origin, HostOrigin) and item.origin.session is not None
            )
        )

    def enqueue_revision(self, item: RevisionItem) -> None:
        """独立发布宿主请求，不在项目锁外覆盖 A/B 的共享进度。"""
        _write(self._requests / f"{item.id}.json", item)

    def _pending_revisions(self) -> tuple[RevisionItem, ...]:
        """合并 A 进度中的问题与独立宿主请求，排除已经结算的 ID。"""
        progress = self._read_progress()
        items = dict(progress.issues)
        for path in sorted(self._requests.glob("*.json")):
            if path.stem in progress.settled_revisions:
                continue
            try:
                item = _read(path, RevisionItem)
            except IrisEvolutionError as exc:
                if (
                    isinstance(exc.__cause__, FileNotFoundError)
                    and path.stem in self._read_progress().settled_revisions
                ):
                    continue
                raise
            items[item.id] = item
        return tuple(item for item in items.values() if item.id not in progress.settled_revisions)

    def read_pending_revisions(
        self,
        *,
        allowed_sources: frozenset[tuple[str, str]],
        allowed_sessions: frozenset[tuple[str, str]],
        allowed_targets: frozenset[tuple[str, str]],
        limit: int = 1,
        requested_revision_id: str | None = None,
    ) -> tuple[RevisionItem, ...]:
        """先按来源与当前开放目标过滤，再优先本次显式请求并限量。"""
        eligible = []
        for item in self._pending_revisions():
            origin = item.origin
            if isinstance(origin, ExperienceOrigin):
                allowed = all(
                    (source.lifecycle_source_id, source.run_id) in allowed_sources
                    for source in origin.sources
                )
            else:
                allowed = (
                    origin.session is None
                    or (origin.session.lifecycle_source_id, origin.session.session_id)
                    in allowed_sessions
                )
            if allowed and all(
                (target.kind, target.name) in allowed_targets for target in item.targets
            ):
                eligible.append(item)
        if requested_revision_id is not None:
            eligible.sort(key=lambda item: item.id != requested_revision_id)
        return tuple(eligible[:limit])

    def settle_revision(self, item_id: str, step: EvolutionResult) -> None:
        """项目锁内结算一项 B，不重写 A 消费位置，也不保留已完成问题正文。"""
        progress = self._read_progress()
        _write(
            self._progress_path,
            progress.model_copy(
                update={
                    "issues": {
                        key: item for key, item in progress.issues.items() if key != item_id
                    },
                    "settled_revisions": {**progress.settled_revisions, item_id: step},
                    "latest_step": step,
                }
            ),
        )
        self._remove_body(self._requests / f"{item_id}.json")

    def revision_result(self, item_id: str) -> EvolutionResult | None:
        """读取指定请求的简短已结算结果，允许宿主观察另一进程的完成。"""
        return self._read_progress().settled_revisions.get(item_id)

    def read_pending(
        self, *, allowed_sources: frozenset[tuple[str, str]], limit: int = 128
    ) -> PendingMaterials:
        """项目锁内按合格来源读取完整消息，重叠区间只出现一次。"""
        states, captures = self._load_sources()
        items: list[EvolutionMaterial] = []
        for key, state in states.items():
            if (
                (state.source.lifecycle_source_id, state.source.run_id) not in allowed_sources
                or state.terminal_message_count is None
                or state.consumed_until >= state.terminal_message_count
            ):
                continue
            through = min(
                state.terminal_message_count, state.consumed_until + limit + 1 - len(items)
            )
            messages: dict[int, tuple[EvolutionRecord, ...]] = {}
            for path, capture in captures:
                check_generation_cancelled()
                if (
                    _key(capture.source) != key
                    or capture.end_message_count <= state.consumed_until
                    or capture.start_message_count >= through
                ):
                    continue
                block = _read(self._blocks / path.name, EvolutionCaptureBlock)
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
                if len(items) > limit:
                    return PendingMaterials(tuple(items[:limit]), True)
        return PendingMaterials(tuple(items), False)

    def consume(
        self,
        selected: tuple[EvolutionMaterial, ...],
        step: EvolutionResult,
        *,
        issue: RevisionItem | None = None,
    ) -> None:
        """项目锁内确认实际选中范围，先保存消费位置再清理完整已读正文。"""
        states, captures = self._load_sources()
        progress = self._read_progress()
        consumed = dict(progress.consumed)
        for item in selected:
            key = _key(item.source)
            state = states[key]
            position = consumed.get(key, state.initial_message_count)
            if item.start_message_count != position:
                raise IrisEvolutionError(
                    "项目材料消费范围不连续", run_id=item.source.run_id, consumed_until=position
                )
            consumed[key] = item.end_message_count
        issues = dict(progress.issues)
        if issue is not None:
            issues[issue.id] = issue
        _write(
            self._progress_path,
            progress.model_copy(
                update={
                    "consumed": consumed,
                    "latest_step": step,
                    "issues": issues,
                }
            ),
        )
        for path, capture in captures:
            key = _key(capture.source)
            if capture.end_message_count <= consumed.get(key, capture.initial_message_count):
                self._remove_body(self._blocks / path.name)

    def record_step(self, step: EvolutionResult) -> None:
        """项目锁内覆盖简短结果，失败记录不推进任何消费位置。"""
        progress = self._read_progress()
        _write(self._progress_path, progress.model_copy(update={"latest_step": step}))

    def _read_progress(self) -> _Progress:
        """读取当前消费位置，尚未有步骤时返回空进度。"""
        if not self._progress_path.exists():
            return _Progress()
        return _read(self._progress_path, _Progress)

    def _load_sources(
        self,
    ) -> tuple[dict[str, EvolutionSourceState], list[tuple[Path, EvolutionCaptureBlock]]]:
        """仅从不可变登记/收据推导连续捕获位置，正文清理不会改变它。"""
        progress = self._read_progress()
        captures = [
            (path, _read(path, EvolutionCaptureBlock))
            for path in sorted(self._captures.glob("*.json"))
        ]
        states: dict[str, EvolutionSourceState] = {}
        for path in sorted(self._sources.glob("*.json")):
            registration = _read(path, _Registration)
            key = _key(registration.source)
            position = registration.initial_message_count
            terminal: int | None = None
            outcome: str | None = None
            for capture in sorted(
                (capture for _, capture in captures if _key(capture.source) == key),
                key=lambda item: (item.start_message_count, item.end_message_count),
            ):
                if capture.start_message_count <= position:
                    position = max(position, capture.end_message_count)
                if capture.terminal_message_count is not None:
                    terminal = capture.terminal_message_count
                    outcome = capture.outcome
            consumed = progress.consumed.get(key, registration.initial_message_count)
            if not registration.initial_message_count <= consumed <= position:
                raise IrisEvolutionError("项目材料消费位置与捕获范围不一致", path=str(path))
            sealed = terminal is not None and position == terminal
            states[key] = EvolutionSourceState.model_construct(
                source=registration.source,
                initial_message_count=registration.initial_message_count,
                captured_until=position,
                consumed_until=consumed,
                terminal_message_count=terminal if sealed else None,
                outcome=outcome if sealed else None,
            )
        return states, captures

    @staticmethod
    def _remove_body(path: Path) -> None:
        """消费进度已经落盘，正文删除可在重试时幂等完成。"""
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise IrisEvolutionError(
                "已消费项目材料清理失败", path=str(path), error=str(exc)
            ) from exc
