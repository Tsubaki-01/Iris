"""Decision 工具发现的共享契约、单次评价与披露事实。"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from iris.decision import ChoiceAnswer, DecisionRequest, DecisionResponse, DecisionUsage
from iris.exceptions import IrisDecisionError, IrisToolValidationError
from iris.message import ToolUseBlock
from iris.prompts import PromptSnapshot, PromptSource
from iris.tools import (
    CallableTool,
    ToolCapability,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
)
from iris.tools.discovery import ToolSearchInput, ToolSearchTool


@pytest.fixture
def prompt_snapshot(tmp_path: Path) -> PromptSnapshot:
    return PromptSource.initialize(tmp_path).snapshot()


class FakeDecision:
    """只实现 evaluate 的借用对象，按题号返回乱序答案。"""

    def __init__(self, choices: tuple[str, ...], error: IrisDecisionError | None = None) -> None:
        self.choices = choices
        self.error = error
        self.requests: list[DecisionRequest] = []

    async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return DecisionResponse(
            provider="typesafe",
            model="test-actual-model",
            answers={
                f"q{i}": ChoiceAnswer(choice=choice, probabilities={choice: 1.0}, confidence=1.0)
                for i, choice in reversed(list(enumerate(self.choices)))
            },
            usage=DecisionUsage(input_tokens=42, output_tokens=9),
        )


def _register(
    registry: ToolRegistry, name: str, *, group: str = "docs", deferred: bool = True
) -> None:
    registry.register(
        CallableTool(
            lambda: "ok",
            name=name,
            description="共享工具描述",
            group=group,
            deferred=deferred,
            tags=["not-sent"],
            capabilities={ToolCapability.READ},
        )
    )


@pytest.mark.asyncio
async def test_decision_uses_full_current_allowed_catalog_once_and_maps_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt_snapshot: PromptSnapshot
) -> None:
    registry = ToolRegistry()
    for name in ("first", "none", "denied"):
        _register(registry, name)
    _register(registry, "outside", group="other")
    _register(registry, "eager", deferred=False)
    evaluator = FakeDecision(("c1", "none", "c1", "c2"))
    search = ToolSearchTool(
        registry.view(deny={"denied"}),
        decision_client=evaluator,
        prompt_snapshot=prompt_snapshot,
    )
    registry.register(search)
    _register(registry, "published_later")

    def no_local_index(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("Decision 模式不运行 BM25 预筛选或回退")

    monkeypatch.setattr(registry, "search_deferred", no_local_index)
    queries = ["按名字选择 none", "不匹配", "按名字选择 none", "选择新发布工具"]
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(
            id="search", name="tool_search", input={"queries": queries, "include_groups": ["docs"]}
        ),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert not result.is_error
    assert len(evaluator.requests) == 1
    request = evaluator.requests[0]
    assert request.state == {
        "queries": queries,
        "tools": {
            "c0": {"name": "first", "description": "共享工具描述"},
            "c1": {"name": "none", "description": "共享工具描述"},
            "c2": {"name": "published_later", "description": "共享工具描述"},
        },
    }
    assert list(request.questions) == ["q0", "q1", "q2", "q3"]
    for i, question in enumerate(request.questions.values()):
        assert question.type == "choice"
        assert set(question.options) == {"c0", "c1", "c2", "none"}
        assert all(question.options[key] is None for key in ("c0", "c1", "c2"))
        assert question.options["none"]
        assert f"queries[{i}]" in question.instructions
        assert "共享工具描述" not in question.instructions
    assert result.data["selections"] == [
        {"query": queries[0], "tool": "none"},
        {"query": queries[1], "tool": None},
        {"query": queries[2], "tool": "none"},
        {"query": queries[3], "tool": "published_later"},
    ]
    assert [tool["name"] for tool in result.data["tools"]] == ["none", "published_later"]
    assert json.loads(result.model_content) == result.data
    extra = result.to_block_metadata()["extra"]
    assert extra["context_revealed_tools"] == ["none", "published_later"]
    assert extra["decision"] == {
        "feature": "tools.discovery",
        "provider": "typesafe",
        "model": "test-actual-model",
        "question_count": 4,
        "input_tokens": 42,
        "output_tokens": 9,
    }


@pytest.mark.parametrize(
    "params",
    [
        {"query": "docs"},
        {"queries": ["docs"], "limit": 1},
        {"queries": ["docs"], "extra": True},
        {"queries": []},
        {"queries": [""]},
        {"queries": [" "]},
        {"queries": "docs"},
        {"queries": [1]},
    ],
)
def test_both_backends_reject_invalid_or_previous_input(params: dict[str, Any]) -> None:
    registry = ToolRegistry()
    for client in (None, FakeDecision(())):
        search = ToolSearchTool(registry.view(), decision_client=client)
        with pytest.raises(IrisToolValidationError):
            search.validate_input(params)


def test_backends_share_the_same_input_schema_description_and_example() -> None:
    registry = ToolRegistry()
    local = ToolSearchTool(registry.view())
    remote = ToolSearchTool(registry.view(), decision_client=FakeDecision(()))
    assert local.input_model is remote.input_model is ToolSearchInput
    assert local.definition.input_schema == remote.definition.input_schema
    assert local.definition.description == remote.definition.description
    assert local.definition.metadata["examples"] == remote.definition.metadata["examples"]
    schema = local.definition.input_schema
    assert set(schema["properties"]) == {"queries", "include_groups"}
    assert schema["required"] == ["queries"]
    assert schema["additionalProperties"] is False
    assert local.validate_input(
        {"queries": [" docs ", "docs"], "include_groups": None}
    ).queries == ["docs", "docs"]
    assert local.definition.capabilities == {ToolCapability.READ}
    assert remote.definition.capabilities == {ToolCapability.READ, ToolCapability.NETWORK}
    assert local.is_read_only({})
    assert not remote.is_read_only({})


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_groups", [False, True])
async def test_empty_catalog_or_groups_returns_equal_length_nulls_without_evaluation(
    tmp_path: Path, empty_groups: bool
) -> None:
    registry = ToolRegistry()
    if empty_groups:
        _register(registry, "candidate")
    evaluator = FakeDecision(())
    search = ToolSearchTool(registry.view(), decision_client=evaluator)
    result = await search.arun(
        ToolSearchInput(queries=["one", "one"], include_groups=[] if empty_groups else None),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert result.data == {
        "selections": [{"query": "one", "tool": None}, {"query": "one", "tool": None}],
        "tools": [],
    }
    assert evaluator.requests == []
    assert "decision" not in result.metadata


@pytest.mark.asyncio
async def test_single_candidate_still_competes_with_none(
    tmp_path: Path, prompt_snapshot: PromptSnapshot
) -> None:
    registry = ToolRegistry()
    _register(registry, "only")
    evaluator = FakeDecision(("none",))
    result = await ToolSearchTool(
        registry.view(), decision_client=evaluator, prompt_snapshot=prompt_snapshot
    ).arun(ToolSearchInput(queries=["unrelated"]), ToolExecutionContext(workspace_root=tmp_path))
    assert len(evaluator.requests) == 1
    assert set(evaluator.requests[0].questions["q0"].options) == {"c0", "none"}
    assert result.data["selections"] == [{"query": "unrelated", "tool": None}]


@pytest.mark.asyncio
async def test_catalog_is_not_truncated_before_the_jev_boundary(
    tmp_path: Path, prompt_snapshot: PromptSnapshot
) -> None:
    registry = ToolRegistry()
    for i in range(255):
        _register(registry, f"tool{i}")
    evaluator = FakeDecision(("none",))
    await ToolSearchTool(
        registry.view(), decision_client=evaluator, prompt_snapshot=prompt_snapshot
    ).arun(ToolSearchInput(queries=["pick"]), ToolExecutionContext(workspace_root=tmp_path))
    assert len(evaluator.requests[0].questions["q0"].options) == 256


@pytest.mark.asyncio
async def test_decision_error_is_execution_error_without_disclosure_or_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt_snapshot: PromptSnapshot
) -> None:
    registry = ToolRegistry()
    _register(registry, "candidate")
    evaluator = FakeDecision((), IrisDecisionError("network failed"))
    registry.register(
        ToolSearchTool(registry.view(), decision_client=evaluator, prompt_snapshot=prompt_snapshot)
    )

    def no_fallback(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("远程失败不可回退")

    monkeypatch.setattr(registry, "search_deferred", no_fallback)
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="search", name="tool_search", input={"queries": ["candidate"]}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert result.is_error and result.error.code == "EXECUTION_ERROR"
    assert len(evaluator.requests) == 1
    assert "context_revealed_tools" not in result.to_block_metadata()["extra"]


@pytest.mark.asyncio
async def test_outer_cancellation_reaches_borrowed_evaluator(
    tmp_path: Path, prompt_snapshot: PromptSnapshot
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingDecision:
        async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            raise AssertionError("等待必须取消")

    registry = ToolRegistry()
    _register(registry, "candidate")
    tool = ToolSearchTool(
        registry.view(), decision_client=BlockingDecision(), prompt_snapshot=prompt_snapshot
    )
    task = asyncio.create_task(
        tool.arun(
            ToolSearchInput(queries=["candidate"]), ToolExecutionContext(workspace_root=tmp_path)
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_decision_instructions_use_frozen_source_and_current_query_indices(
    tmp_path: Path,
) -> None:
    """已构造工具固定项目正文，新请求仍逐条传入当前 query_index。"""
    source = PromptSource.initialize(tmp_path)
    prompt = source.root / "tool_discovery_instruction.j2"
    prompt.write_text("旧指令 state.queries[{{ query_index }}]", encoding="utf-8")
    registry = ToolRegistry()
    _register(registry, "candidate")
    evaluator = FakeDecision(("c0",))
    tool = ToolSearchTool(
        registry.view(), decision_client=evaluator, prompt_snapshot=source.snapshot()
    )
    context = ToolExecutionContext(workspace_root=tmp_path)
    await tool.arun(ToolSearchInput(queries=["first"]), context)
    prompt.write_text("新指令 state.queries[{{ query_index }}]", encoding="utf-8")
    evaluator.choices = ("c0", "none")
    await tool.arun(ToolSearchInput(queries=["current", "unmatched"]), context)

    current = evaluator.requests[-1]
    assert current.state["queries"] == ["current", "unmatched"]
    assert [question.instructions for question in current.questions.values()] == [
        "旧指令 state.queries[0]",
        "旧指令 state.queries[1]",
    ]
    replacement = ToolSearchTool(
        registry.view(), decision_client=evaluator, prompt_snapshot=source.snapshot()
    )
    await replacement.arun(ToolSearchInput(queries=["current", "unmatched"]), context)
    assert [question.instructions for question in evaluator.requests[-1].questions.values()] == [
        "新指令 state.queries[0]",
        "新指令 state.queries[1]",
    ]
