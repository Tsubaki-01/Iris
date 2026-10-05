"""工具发现两种后端的选择映射与结果投影。"""

import json
from typing import Any, cast

from ..decision import ChoiceAnswer, ChoiceQuestion, DecisionEvaluator, DecisionRequest
from ..exceptions import IrisDecisionError, IrisTemplateError, IrisToolExecutionError
from ..message import TextBlock
from ..prompts import PromptSnapshot
from .base import ToolDefinition, ToolResult


async def select_with_decision(
    evaluator: DecisionEvaluator,
    queries: list[str],
    candidates: list[ToolDefinition],
    *,
    prompt_snapshot: PromptSnapshot | None,
) -> tuple[list[ToolDefinition | None], dict[str, Any]]:
    """把全部允许候选与独立意图一次提交，再按题号回映工具。"""
    if not candidates:
        return [None for _ in queries], {}
    if prompt_snapshot is None:
        raise IrisToolExecutionError("Decision 工具发现需要项目 prompt 快照")
    try:
        instructions = [
            prompt_snapshot.render("tool_discovery_instruction", {"query_index": i}).strip()
            for i in range(len(queries))
        ]
    except IrisTemplateError as exc:
        raise IrisToolExecutionError("工具发现指令模板渲染失败", **exc.context) from exc
    candidate_map = {f"c{i}": definition for i, definition in enumerate(candidates)}
    options: dict[str, str | None] = {key: None for key in candidate_map}
    options["none"] = "No available tool matches this query."
    request = DecisionRequest.model_construct(
        state={
            "queries": queries,
            "tools": {
                key: {"name": definition.name, "description": definition.description}
                for key, definition in candidate_map.items()
            },
        },
        questions={
            f"q{i}": ChoiceQuestion.model_construct(
                instructions=instructions[i],
                options=options,
            )
            for i in range(len(queries))
        },
    )
    try:
        response = await evaluator.evaluate(request)
    except IrisDecisionError as exc:
        raise IrisToolExecutionError("工具发现 Decision 调用失败", error=str(exc)) from exc
    selected: list[ToolDefinition | None] = []
    for i in range(len(queries)):
        answer = cast(ChoiceAnswer, response.answers[f"q{i}"])
        selected.append(None if answer.choice == "none" else candidate_map[answer.choice])
    return selected, {
        "feature": "tools.discovery",
        "provider": response.provider,
        "model": response.model,
        "question_count": len(queries),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def selection_result(
    tool_name: str,
    queries: list[str],
    selected: list[ToolDefinition | None],
    decision_metadata: dict[str, Any],
) -> ToolResult:
    """保留逐意图结果，摘要和披露按首次命中顺序去重。"""
    unique = {definition.name: definition for definition in selected if definition is not None}
    data = {
        "selections": [
            {"query": query, "tool": definition.name if definition is not None else None}
            for query, definition in zip(queries, selected, strict=True)
        ],
        "tools": [
            {
                "name": definition.name,
                "description": definition.description[:240],
                "group": definition.group,
            }
            for definition in unique.values()
        ],
    }
    metadata: dict[str, Any] = {"context_revealed_tools": list(unique)}
    if decision_metadata:
        metadata["decision"] = decision_metadata
    return ToolResult(
        tool_use_id="",
        tool_name=tool_name,
        content=[TextBlock(text=json.dumps(data, ensure_ascii=False, separators=(",", ":")))],
        data=data,
        metadata=metadata,
    )
