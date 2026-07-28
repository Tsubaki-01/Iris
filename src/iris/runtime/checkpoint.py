"""Runtime HITL checkpoint 的构建与纯验证。"""

from __future__ import annotations

import json
from typing import Any, Literal

from ..exceptions import HITLCheckpointInvalidError
from ..hitl import HumanInteraction
from ..message import LLMResponse, Msg
from ..tools import ToolResult
from .models import (
    ProviderResponseSnapshot,
    RuntimeHITLCheckpoint,
    RuntimeOptions,
    RuntimeOptionsSnapshot,
)


def build_hitl_checkpoint(
    *,
    run_mode: Literal["turn", "loop"],
    agent_name: str,
    runtime_options: RuntimeOptions,
    assistant_message: Msg,
    response: LLMResponse,
    step_index: int,
    next_tool_index: int,
    batch_results: list[dict[str, Any]],
    all_tool_results: list[ToolResult],
    read_state: Any | None,
) -> dict[str, Any]:
    """构造并验证等待恢复所需的 JSON-safe checkpoint。"""
    try:
        read_state_json: Any | None = None
        if read_state is not None:
            read_state_json = (
                read_state.model_dump(mode="json")
                if hasattr(read_state, "model_dump")
                else read_state
            )
        checkpoint = RuntimeHITLCheckpoint(
            run_mode=run_mode,
            agent_name=agent_name,
            session_id=runtime_options.session_id,
            run_id=runtime_options.run_id,
            step_index=step_index,
            runtime_options=RuntimeOptionsSnapshot(options=runtime_options.model_dump(mode="json")),
            assistant_message={
                "role": "assistant",
                "content": [block.model_dump(mode="json") for block in response.content],
                "metadata": {
                    "provider": response.provider,
                    "id": response.id,
                    "model": response.model,
                    "finish_reason": response.finish_reason,
                },
            },
            provider_response=ProviderResponseSnapshot(
                provider=response.provider,
                response_id=response.id,
                model=response.model,
                content=[block.model_dump(mode="json") for block in response.content],
                finish_reason=response.finish_reason,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
                reasoning=response.reasoning,
            ),
            tool_calls=[call.model_dump(mode="json") for call in assistant_message.tool_calls],
            next_tool_index=next_tool_index,
            batch_results=batch_results,
            all_tool_results=[result.model_dump(mode="json") for result in all_tool_results],
            read_state=read_state_json,
        ).model_dump(mode="json")
        json.dumps(checkpoint, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise HITLCheckpointInvalidError("HITL checkpoint 必须是 JSON-safe 数据") from exc
    return checkpoint


def validate_hitl_checkpoint(
    interaction: HumanInteraction,
    *,
    agent_name: str,
) -> tuple[dict[str, Any], RuntimeOptions]:
    """验证 checkpoint v2 的结构、runtime 身份与调用选项。"""
    checkpoint = interaction.checkpoint
    try:
        snapshot = RuntimeHITLCheckpoint.model_validate(checkpoint)
        options = RuntimeOptions.model_validate(snapshot.runtime_options.options)
    except (TypeError, ValueError) as exc:
        raise HITLCheckpointInvalidError("HITL checkpoint 结构无效") from exc
    if (
        snapshot.agent_name != agent_name
        or snapshot.session_id != interaction.session_id
        or snapshot.run_id != interaction.run_id
        or options.session_id != interaction.session_id
        or options.run_id != interaction.run_id
    ):
        raise HITLCheckpointInvalidError("HITL checkpoint 与当前 runtime 不匹配")
    return checkpoint, options


__all__ = ["build_hitl_checkpoint", "validate_hitl_checkpoint"]
