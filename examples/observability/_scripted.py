"""两个观测示例共用的离线模型与命令行导出配置。"""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from uuid import uuid4

from iris.config import init_config
from iris.exceptions import IrisProviderError
from iris.message import (
    LLMRequest,
    LLMResponse,
    ModelBlockCompleted,
    ModelBlockDelta,
    ModelBlockRef,
    ModelBlockStarted,
    ModelResponseCompleted,
    ModelResponseFailed,
    ModelResponseStarted,
    ModelStreamEvent,
    ModelStreamScope,
    ModelUsageSnapshot,
    ModelUsageUpdated,
    ProviderStreamError,
    TextBlock,
)
from iris.observability import AgentObservabilityConfig, ObservabilityExportConfig


class ScriptedProvider:
    """每次请求消费一条固定响应；真实工具和生命周期仍由 Iris 执行。"""

    def __init__(self, steps: Sequence[LLMResponse | IrisProviderError]) -> None:
        """保存每次调用恰好消费一条的演示脚本。"""
        self._steps = iter(steps)
        self.requests: list[LLMRequest] = []

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """小型离线脚本只需固定估算，不访问 tokenizer 或网络。"""
        return 1

    def _next(self, request: LLMRequest) -> LLMResponse | IrisProviderError:
        """保留请求并消费唯一下一步，脚本耗尽直接暴露示例错误。"""
        self.requests.append(request)
        return next(self._steps)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """返回完整 typed 响应或抛出本次实际模拟的 provider 异常。"""
        step = self._next(request)
        if isinstance(step, IrisProviderError):
            raise step
        return step

    async def stream(self, request: LLMRequest) -> AsyncIterator[ModelStreamEvent]:
        """文本分两段交付；工具调用随完整终态交付，不重建业务响应。"""
        step = self._next(request)
        scope = ModelStreamScope(
            model_stream_id=f"demo-stream-{len(self.requests)}",
            provider="scripted",
            model=request.model,
            attempt=1,
        )
        sequence = count(1)
        occurred_at = datetime.now(UTC)
        yield ModelResponseStarted(
            scope=scope,
            sequence=next(sequence),
            occurred_at=occurred_at,
            response_id=(
                f"demo-error-{len(self.requests)}"
                if isinstance(step, IrisProviderError)
                else step.id
            ),
        )
        if isinstance(step, IrisProviderError):
            yield ModelUsageUpdated(
                scope=scope,
                sequence=next(sequence),
                occurred_at=occurred_at,
                usage=ModelUsageSnapshot(**step.context["usage"]),
            )
            yield ModelResponseFailed(
                scope=scope,
                sequence=next(sequence),
                occurred_at=occurred_at,
                error=ProviderStreamError(
                    code="PROVIDER_ERROR", message=step.message, retryable=False
                ),
                semantic_output_emitted=False,
            )
            return
        emitted = False
        for index, block in enumerate(step.content):
            if not isinstance(block, TextBlock) or not block.text:
                continue
            ref = ModelBlockRef(index=index, block_id=f"text-{index}", kind="text")
            yield ModelBlockStarted(
                scope=scope, sequence=next(sequence), occurred_at=occurred_at, block=ref
            )
            midpoint = max(1, len(block.text) // 2)
            snapshot = ""
            for chunk in (block.text[:midpoint], block.text[midpoint:]):
                if chunk:
                    snapshot += chunk
                    emitted = True
                    yield ModelBlockDelta(
                        scope=scope,
                        sequence=next(sequence),
                        occurred_at=occurred_at,
                        block=ref,
                        channel="text",
                        delta=chunk,
                        snapshot=snapshot,
                    )
            yield ModelBlockCompleted(
                scope=scope, sequence=next(sequence), occurred_at=occurred_at, block=ref
            )
        counts = {
            "input_tokens": step.input_tokens,
            "output_tokens": step.output_tokens,
            "total_tokens": step.total_tokens,
        }
        usage = {key: value for key, value in counts.items() if key in step.model_fields_set}
        if usage:
            yield ModelUsageUpdated(
                scope=scope,
                sequence=next(sequence),
                occurred_at=occurred_at,
                usage=ModelUsageSnapshot(**usage, complete=True),
            )
        yield ModelResponseCompleted(
            scope=scope,
            sequence=next(sequence),
            occurred_at=occurred_at,
            response=step,
            semantic_output_emitted=emitted,
        )


def configure_cli(
    name: str, description: str, argv: Sequence[str] | None
) -> tuple[Path, AgentObservabilityConfig]:
    """把示例参数交给唯一全局配置入口，工作文件默认位于项目 tmp。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--experiment-id", required=True, help="MLflow 中已创建的 experiment ID")
    parser.add_argument("--endpoint", default="http://127.0.0.1:5000/v1/traces")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--disabled", action="store_true", help="关闭观测，仍执行相同业务流程")
    parser.add_argument("--no-content", action="store_true", help="只记录元数据，不采集正文")
    parser.add_argument("--max-content-chars", type=int, default=65536)
    args = parser.parse_args(argv)
    capture = AgentObservabilityConfig(
        enabled=not args.disabled,
        capture_content=not args.no_content,
        max_content_chars=args.max_content_chars,
    )
    init_config(
        observability=ObservabilityExportConfig(
            traces_endpoint=args.endpoint,
            headers={"x-mlflow-experiment-id": args.experiment_id},
            service_name="iris-observability-examples",
        )
    )
    workspace = args.workspace or (
        Path(__file__).resolve().parents[2] / "tmp" / f"observability-{name}-{uuid4().hex[:8]}"
    )
    return workspace.resolve(), capture
