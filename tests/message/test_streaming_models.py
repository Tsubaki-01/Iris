"""Provider-neutral streaming 消息模型测试。"""

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from iris.message import (
    LLMResponse,
    ModelBlockRef,
    ModelResponseCompleted,
    ModelStreamEvent,
    ModelStreamFinalization,
    ModelStreamScope,
    TextBlock,
)

_OCCURRED_AT = datetime(2026, 8, 31, 12, tzinfo=UTC)


def _scope() -> ModelStreamScope:
    return ModelStreamScope(
        model_stream_id="stream-1",
        provider="openai",
        model="gpt-4o",
        attempt=1,
    )


def _text_block() -> ModelBlockRef:
    return ModelBlockRef(index=0, block_id="block-0", kind="text")


def test_model_stream_event_union_round_trips_completed_response() -> None:
    event = ModelResponseCompleted(
        scope=_scope(),
        sequence=3,
        occurred_at=_OCCURRED_AT,
        response=LLMResponse(
            provider="openai",
            id="response-1",
            model="gpt-4o",
            content=[TextBlock(text="你好")],
            finish_reason="stop",
            input_tokens=2,
            output_tokens=1,
            total_tokens=3,
        ),
        semantic_output_emitted=True,
    )

    restored = TypeAdapter(ModelStreamEvent).validate_json(
        TypeAdapter(ModelStreamEvent).dump_json(event)
    )

    assert restored == event


def test_model_stream_finalization_requires_response_only_for_success() -> None:
    response = LLMResponse(provider="openai", content=[TextBlock(text="完成")])

    completed = ModelStreamFinalization(
        status="completed",
        response=response,
        last_provider_sequence=4,
        semantic_output_emitted=True,
    )

    assert completed.response == response
    assert completed.error is None

    with pytest.raises(ValidationError):
        ModelStreamFinalization(
            status="failed",
            response=response,
            last_provider_sequence=4,
            semantic_output_emitted=True,
        )
