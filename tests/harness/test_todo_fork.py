"""公开 SessionHistory 分支后只根据新身份读取当前 Todo 文件。"""

from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.harness import AgentRunner, SessionHistory
from iris.lifecycle import AgentRunOptions, AgentRunRequest, RunLimits, RunStopReason
from iris.message import LLMRequest, ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore

from .fakes import StaticProvider, text_response, tool_response


def _snapshot(request: LLMRequest) -> str:
    return "\n".join(
        message.text
        for message in request.messages
        if message.metadata.get("context_kind") == "runtime_snapshot"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("existing_branch_file", [False, True])
async def test_fork_uses_only_new_session_file_and_never_rehydrates_deleted_todo(
    tmp_path: Path, persistent: bool, existing_branch_file: bool
) -> None:
    """保留来源真实工具历史，分支显示新路径、读取自己文件且删除后保持为空。"""
    store = SQLiteStore(tmp_path / "fork.db") if persistent else InMemoryLifecycleStore()
    provider = StaticProvider()
    runner = AgentRunner.from_config(
        AgentConfig(
            name="todo-fork",
            model="openai/test",
            system="维护当前会话清单。",
            todo={"enabled": True},
            permissions={"workspace": str(tmp_path), "writes": "allow"},
            tools={"builtin": ["file.read", "file.write"]},
        ),
        provider=provider,
        store=store,
    )
    try:
        source_path = (await runner.get_todo("source")).path
        source_path.parent.mkdir(parents=True)
        source_path.write_text("- [ ] source-private-plan\n", encoding="utf-8")
        provider.responses.extend(
            [
                tool_response(
                    ToolUseBlock(
                        id="read-source", name="read_file", input={"file_path": str(source_path)}
                    )
                ),
                tool_response(
                    ToolUseBlock(
                        id="write-source",
                        name="write_file",
                        input={
                            "file_path": str(source_path),
                            "content": "- [x] source-private-plan\n",
                        },
                    )
                ),
                text_response("来源任务结束"),
            ]
        )
        source = await runner.start(
            AgentRunRequest(input="处理来源任务", session_id="source", run_id="source-run"),
            options=AgentRunOptions(limits=RunLimits(max_model_steps=3)),
        )
        assert source.run.stop_reason is RunStopReason.COMPLETED
        source_session = runner.get_session("source")
        branch = SessionHistory(store).fork("source-run")
        assert branch.session_id != "source"
        assert branch.messages == source_session.messages
        assert len(provider.requests) == 3
        assert "source-private-plan" in "\n".join(
            message.model_dump_json() for message in branch.messages
        )
        branch_todo = await runner.get_todo(branch.session_id)
        assert branch_todo.path != source_path
        assert branch_todo.items == () and not branch_todo.path.exists()
        if existing_branch_file:
            branch_todo.path.write_text("- [x] branch-own-plan\n", encoding="utf-8")
            provider.responses.extend(
                [
                    tool_response(
                        ToolUseBlock(
                            id="read-branch",
                            name="read_file",
                            input={"file_path": str(branch_todo.path)},
                        )
                    ),
                    text_response("分支读取完毕"),
                ]
            )
        else:
            provider.responses.append(text_response("分支当前没有清单"))
        await runner.start(
            AgentRunRequest(input="继续分支", session_id=branch.session_id, run_id="branch-run"),
            options=AgentRunOptions(limits=RunLimits(max_model_steps=2)),
        )
        current = _snapshot(provider.requests[3])
        assert str(branch_todo.path) in current
        assert str(source_path) not in current
        assert "source-private-plan" not in current
        if existing_branch_file:
            assert "branch-own-plan" in current
            assert (await runner.get_todo(branch.session_id)).items[0].content == "branch-own-plan"
            assert "branch-own-plan" in "\n".join(
                message.model_dump_json()
                for message in runner.get_session(branch.session_id).messages
            )
            branch_todo.path.unlink()
        else:
            assert "当前清单为空" in current
            assert not branch_todo.path.exists()

        provider.responses.append(text_response("仍然没有清单"))
        await runner.start(
            AgentRunRequest(input="再次查看分支", session_id=branch.session_id),
            options=AgentRunOptions(limits=RunLimits(max_model_steps=1)),
        )
        deleted = _snapshot(provider.requests[-1])
        assert str(branch_todo.path) in deleted and "当前清单为空" in deleted
        assert "branch-own-plan" not in deleted and "source-private-plan" not in deleted
        assert not branch_todo.path.exists()
        assert (await runner.get_todo(branch.session_id)).items == ()
        assert source_path.read_text(encoding="utf-8") == "- [x] source-private-plan\n"
        assert runner.get_session("source") == source_session
    finally:
        await runner.aclose()
