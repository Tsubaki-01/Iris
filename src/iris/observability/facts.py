"""实际配置与来源采用的只读事实，不依赖 OTel 开关或持久状态机。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import get_ident
from typing import TYPE_CHECKING
from uuid import uuid4

from ..utils.sources import SourceDocument

if TYPE_CHECKING:
    from ..harness.configuration import EffectiveConfiguration

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConfigurationApplied:
    """本次真实 activation 采用的 Runner 配置实例。"""

    configuration_snapshot_id: str
    run_id: str
    session_id: str
    activation_id: str
    agent_id: str
    configuration: EffectiveConfiguration


@dataclass(frozen=True, slots=True)
class SourceAdopted:
    """消费 owner 确认的来源快照；不表示任务效果已得到验证。"""

    adoption_id: str
    owner_kind: str
    source_kind: str
    adoption_boundary: str
    adopted_at: datetime
    documents: tuple[SourceDocument, ...]
    configuration_snapshot_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    activation_id: str | None = None
    step_index: int | None = None
    preparation_id: str | None = None
    maintenance_cycle_id: str | None = None
    resource_ref: str | None = None
    source_versions: tuple[tuple[str, str | None], ...] = ()


type SourceFact = ConfigurationApplied | SourceAdopted


@dataclass(frozen=True, slots=True)
class _FactScope:
    """真实执行 owner 的进程内观察关联，线程 worker 借用同一出口。"""

    publish: Callable[[SourceFact], None]
    loop: asyncio.AbstractEventLoop
    thread_id: int
    configuration_snapshot_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    activation_id: str | None = None
    step_index: int | None = None
    preparation_id: str | None = None
    maintenance_cycle_id: str | None = None
    resource_ref: str | None = None


_SCOPE: ContextVar[_FactScope | None] = ContextVar("iris_source_adoption", default=None)


@contextmanager
def bind_fact_scope(
    publish: Callable[[SourceFact], None],
    *,
    configuration_snapshot_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    activation_id: str | None = None,
    maintenance_cycle_id: str | None = None,
    resource_ref: str | None = None,
) -> Iterator[None]:
    """绑定实际执行或维护资源身份，退出恢复外层关联。"""
    scope = _FactScope(
        publish,
        asyncio.get_running_loop(),
        get_ident(),
        configuration_snapshot_id,
        run_id,
        session_id,
        activation_id,
        maintenance_cycle_id=maintenance_cycle_id,
        resource_ref=resource_ref,
    )
    token = _SCOPE.set(scope)
    try:
        yield
    finally:
        _SCOPE.reset(token)


@contextmanager
def bind_fact_step(step_index: int, preparation_id: str | None = None) -> Iterator[None]:
    """只补充当前模型步骤，保持 Run/资源 owner。"""
    scope = _SCOPE.get()
    if scope is None:
        yield
        return
    token = _SCOPE.set(replace(scope, step_index=step_index, preparation_id=preparation_id))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def _publish(scope: _FactScope, fact: SourceAdopted) -> None:
    try:
        scope.publish(fact)
    except Exception:
        _logger.warning("来源采用事实发布失败", exc_info=True)


def record_source_adoption(
    *,
    owner_kind: str,
    source_kind: str,
    boundary: str,
    documents: tuple[SourceDocument, ...],
    source_versions: tuple[tuple[str, str | None], ...] = (),
) -> None:
    """在真实消费点记录正文版本；worker 回到绑定 event loop 发布。"""
    scope = _SCOPE.get()
    if scope is None:
        return
    fact = SourceAdopted(
        f"adoption_{uuid4().hex}",
        owner_kind,
        source_kind,
        boundary,
        datetime.now(UTC),
        documents,
        scope.configuration_snapshot_id,
        scope.run_id,
        scope.session_id,
        scope.activation_id,
        scope.step_index,
        scope.preparation_id,
        scope.maintenance_cycle_id,
        scope.resource_ref,
        source_versions,
    )
    if get_ident() == scope.thread_id:
        _publish(scope, fact)
    else:
        scope.loop.call_soon_threadsafe(_publish, scope, fact)


__all__ = [
    "ConfigurationApplied",
    "SourceAdopted",
    "bind_fact_scope",
    "bind_fact_step",
    "record_source_adoption",
]
