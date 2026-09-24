"""显式概览的无工具模型请求与双字段 JSON 响应边界。"""

from __future__ import annotations

import json
from itertools import groupby

from pydantic import ValidationError

from ..exceptions import IrisMemoryError
from ..message import LLMRequest, LLMResponse, Msg
from ..utils import TemplateRenderer
from ._generation_worker import check_generation_cancelled
from ._prompts import render_memory_prompt
from .models import MemoryNamespaceSnapshot, MemoryOverviewConfig, MemoryOverviewContent


def build_overview_request(
    snapshot: MemoryNamespaceSnapshot,
    model: str,
    config: MemoryOverviewConfig,
    renderer: TemplateRenderer,
) -> LLMRequest:
    """将完整 active 正式知识按分类和类型组织成一次无工具生成请求。"""
    groups: list[dict[str, object]] = []
    for (category, kind), items in groupby(
        snapshot.items, key=lambda item: (item.category, item.kind)
    ):
        check_generation_cancelled()
        groups.append(
            {
                "category": category.value,
                "kind": kind.value,
                "items": [
                    {
                        "id": item.id,
                        "text": item.text,
                        "updated_at": item.updated_at,
                        "created_at": item.created_at,
                        "metadata": item.metadata,
                        "source_type": item.source_type.value,
                        "source_id": item.source_id,
                        "artifacts": [
                            artifact.model_dump(mode="json") for artifact in item.artifacts
                        ],
                    }
                    for item in items
                ],
            }
        )
    source = json.dumps(
        {"namespace": snapshot.state.namespace, "groups": groups}, ensure_ascii=False
    )
    return LLMRequest(
        model=model,
        messages=[
            Msg.system(render_memory_prompt(renderer, "memory_overview.j2", {})),
            Msg.user(source),
        ],
        max_tokens=config.max_tokens,
    )


def complete_overview_content(response: LLMResponse) -> MemoryOverviewContent:
    """仅接受正常完成的双字段 JSON，在响应边界完整解析一次。"""
    if response.finish_reason != "stop":
        raise IrisMemoryError(
            "memory 概览没有完整响应，保留最后完整产物",
            finish_reason=response.finish_reason,
        )
    try:
        return MemoryOverviewContent.model_validate_json(response.to_msg().text)
    except ValidationError as exc:
        raise IrisMemoryError("memory 概览必须返回完整的双字段 JSON，保留最后完整产物") from exc
