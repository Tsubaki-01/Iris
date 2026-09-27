"""消息增量维护发现与普通用户位置，保持全量重放语义。"""

from iris.lifecycle import SessionReadState, SessionToolDiscovery
from iris.message import Msg, Role
from iris.store._session_projection import advance_session_read_state


def _result(*names: str, used: str = "tool_search", error: bool = False) -> Msg:
    return Msg.tool_result(
        tool_use_id="search",
        name=used,
        content="result",
        is_error=error,
        metadata={"extra": {"context_tool_name": used, "context_revealed_tools": list(names)}},
    )


def _replay(messages: list[Msg]) -> SessionReadState:
    discovered: dict[str, int] = {}
    used: dict[str, int] = {}
    latest: tuple[str, ...] = ()
    protected: set[str] = set()
    last_user: int | None = None
    for index, message in enumerate(messages):
        if message.role is Role.ASSISTANT:
            protected.clear()
        if message.role is Role.USER and message.sender != "context" and not message.tool_results:
            last_user = index
        for result in message.tool_results:
            if result.is_error:
                continue
            extra = result.metadata.get("extra", {})
            if "context_tool_name" in extra:
                used[extra["context_tool_name"]] = index
            if "context_revealed_tools" in extra:
                latest = tuple(extra["context_revealed_tools"])
                discovered.update((name, index) for name in latest)
                if latest:
                    protected.add(latest[0])
    return SessionReadState(
        tool_discovery=SessionToolDiscovery(
            discovered_at=discovered,
            used_at=used,
            latest_search_names=latest,
            protected_first_names=tuple(sorted(protected)),
        ),
        last_ordinary_user_index=last_user,
    )


def test_incremental_state_matches_replay_and_preserves_old_state() -> None:
    messages = [
        Msg.user("input"),
        _result("a", "b"),
        _result("c"),
        _result(),
        _result("wrong", error=True),
        Msg.user("context", sender="context"),
        Msg.assistant("done"),
        _result("b", "a", used="a"),
        Msg.user("steer"),
    ]
    initial = SessionReadState()
    expected = _replay(messages)
    assert advance_session_read_state(initial, 0, messages) == expected
    for chunk_size in (1, 2, 4):
        state = initial
        for start in range(0, len(messages), chunk_size):
            previous = state.model_copy(deep=True)
            old = state
            state = advance_session_read_state(state, start, messages[start : start + chunk_size])
            assert old == previous
        assert state == expected
    assert initial == SessionReadState()


def test_no_relevant_delta_reuses_state_and_user_only_reuses_discovery() -> None:
    state = SessionReadState()
    assert advance_session_read_state(state, 0, []) is state
    assert advance_session_read_state(state, 0, [Msg.user("context", sender="context")]) is state
    updated = advance_session_read_state(state, 3, [Msg.user("input")])
    assert updated.last_ordinary_user_index == 3
    assert updated.tool_discovery is state.tool_discovery


def test_search_blocks_in_one_message_keep_order_and_share_absolute_index() -> None:
    first, second = _result("a", "b"), _result("c", "a")
    message = Msg.user([*first.content, *second.content])
    state = advance_session_read_state(SessionReadState(), 8, [message])
    assert state.tool_discovery.discovered_at == {"a": 8, "b": 8, "c": 8}
    assert state.tool_discovery.latest_search_names == ("c", "a")
    assert state.tool_discovery.protected_first_names == ("a", "c")
