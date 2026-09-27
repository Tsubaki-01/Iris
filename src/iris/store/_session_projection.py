"""原文增量的纯折叠与有效历史锚点语义。"""

from collections.abc import Sequence

from ..lifecycle.history import is_ordinary_user
from ..lifecycle.models import SessionReadState, SessionToolDiscovery
from ..message import Msg, Role


def advance_session_read_state(
    state: SessionReadState, message_count: int, delta: Sequence[Msg]
) -> SessionReadState:
    """只消费新消息，构造与原文一起发布的读取状态候选。"""
    discovery = state.tool_discovery
    discovered = discovery.discovered_at
    used = discovery.used_at
    latest = discovery.latest_search_names
    protected = discovery.protected_first_names
    last_user = state.last_ordinary_user_index
    for index, message in enumerate(delta, start=message_count):
        if message.role is Role.ASSISTANT:
            protected = ()
        if is_ordinary_user(message):
            last_user = index
        for result in message.tool_results:
            if result.is_error:
                continue
            facts = result.metadata.get("extra", {})
            name = facts.get("context_tool_name")
            if name is not None:
                if used is discovery.used_at:
                    used = used.copy()
                used[name] = index
            if "context_revealed_tools" in facts:
                latest = tuple(facts["context_revealed_tools"])
                if latest:
                    if discovered is discovery.discovered_at:
                        discovered = discovered.copy()
                    discovered.update((name, index) for name in latest)
                    protected = tuple(sorted({*protected, latest[0]}))
    if (
        discovered is not discovery.discovered_at
        or used is not discovery.used_at
        or latest != discovery.latest_search_names
        or protected != discovery.protected_first_names
    ):
        discovery = SessionToolDiscovery.model_construct(
            discovered_at=discovered,
            used_at=used,
            latest_search_names=latest,
            protected_first_names=protected,
        )
    if discovery is state.tool_discovery and last_user == state.last_ordinary_user_index:
        return state
    return SessionReadState.model_construct(
        tool_discovery=discovery, last_ordinary_user_index=last_user
    )


def protected_run_indices(
    initial_count: int,
    message_count: int,
    first_message: Msg | None,
    last_ordinary_user_index: int | None,
) -> tuple[int, ...]:
    """按当前输入起点、BCI 与最近普通 steer 确定保护位置。"""
    if first_message is None:
        return ()
    indices = {initial_count}
    if first_message.metadata.get("context_kind") == "before_current_input":
        indices.add(initial_count + 1)
    if (
        last_ordinary_user_index is not None
        and initial_count <= last_ordinary_user_index < message_count
    ):
        indices.add(last_ordinary_user_index)
    return tuple(sorted(indices))
