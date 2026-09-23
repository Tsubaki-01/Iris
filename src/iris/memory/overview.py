"""显式概览的无工具模型请求与双字段 JSON 响应边界。"""

from __future__ import annotations

import json
from itertools import groupby

from pydantic import ValidationError

from ..exceptions import IrisMemoryError
from ..message import LLMRequest, LLMResponse, Msg
from .models import MemoryNamespaceSnapshot, MemoryOverviewConfig, MemoryOverviewContent

OVERVIEW_PROMPT = """根据以下有效长期记忆生成简短概览，只输出一个 JSON 对象。
对象必须且只能包含 core_facts 和 knowledge_scope 两个字符串字段，字段内容使用 Markdown 正文。
core_facts：选择经常影响回答的偏好、约定和当前状态；保留提供的精确标识符、路径和数值。
没有可提炼的核心事实时，该字段可以是空字符串，但不能省略。
knowledge_scope：概括输入实际包含的全部知识主题，包括核心事实对应的主题，不能为空。
压缩各主题的表述，不为省长度主动省略主题；可用已提供的 category/kind 辅助后续筛选。
不逐条复述记忆，不列出全部条目 ID，不生成文件目录、工具协议或输入中不存在的主题。
仅使用输入信息，保留冲突和不确定性，不推断未提供的事实。
字段内不添加程序节标题；JSON 外不附文字，也不使用代码围栏。
"""


def build_overview_request(
    snapshot: MemoryNamespaceSnapshot, model: str, config: MemoryOverviewConfig
) -> LLMRequest:
    """将完整 active 正式知识按分类和类型组织成一次无工具生成请求。"""
    groups: list[dict[str, object]] = []
    for (category, kind), items in groupby(
        snapshot.items, key=lambda item: (item.category, item.kind)
    ):
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
        messages=[Msg.system(OVERVIEW_PROMPT), Msg.user(source)],
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
