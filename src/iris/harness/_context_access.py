"""复用 lifecycle 原文和工具 artifact 的有界上下文读取。"""

import json
from pathlib import Path

from ..exceptions import IrisToolExecutionError, IrisToolValidationError
from ..lifecycle import LifecycleStore
from ..message import Msg, TextBlock, ToolResultBlock, ToolUseBlock
from ..tools import ToolArtifact, WorkspacePolicy
from ..tools.context_access import (
    ContextReadInput,
    ContextReadPage,
    ContextSearchHit,
    ContextSearchInput,
    ContextSearchPage,
)


class ContextAccess:
    """绑定 runner 的 exact store；每次读取使用工具执行的 session 身份。"""

    def __init__(self, store: LifecycleStore) -> None:
        """复用 runner 的权威 store，不创建新的持久化资源。"""
        self.store = store

    def read(
        self, session_id: str, params: ContextReadInput, workspace_root: Path
    ) -> ContextReadPage:
        """读取原始消息或某次工具最终文本，不重新执行来源工具。"""
        parts = params.ref.split(":")
        index = int(parts[1])
        page = self.store.read_session_messages(session_id, start=index, limit=1)
        if not page.items:
            raise IrisToolExecutionError(
                "原文引用不在当前会话中", code="CONTEXT_SOURCE_UNAVAILABLE"
            )
        message = page.items[0][1]
        if parts[0] == "message":
            if params.representation == "raw":
                raise IrisToolExecutionError(
                    "message 引用没有 raw 表示", code="CONTEXT_REPRESENTATION_UNAVAILABLE"
                )
            text = _message_text(message)
        else:
            block_index = int(parts[2])
            blocks = message.blocks
            if block_index >= len(blocks) or not isinstance(blocks[block_index], ToolResultBlock):
                raise IrisToolExecutionError(
                    "引用不对应工具结果块", code="CONTEXT_SOURCE_UNAVAILABLE"
                )
            block = blocks[block_index]
            artifact = (
                ToolArtifact.model_validate(block.metadata["artifact"])
                if "artifact" in block.metadata
                else None
            )
            path = (
                artifact.path
                if params.representation == "raw" and artifact
                else artifact.text_path
                if artifact
                else None
            )
            if params.representation == "raw" and path is None:
                raise IrisToolExecutionError(
                    "该结果没有原生 artifact", code="CONTEXT_REPRESENTATION_UNAVAILABLE"
                )
            if path is None:
                text = block.content
            else:
                try:
                    resolved = WorkspacePolicy().resolve_path(
                        str(path), workspace_root=workspace_root
                    )
                    with resolved.open("r", encoding="utf-8", newline="") as source:
                        remaining = params.offset
                        while remaining:
                            skipped = source.read(min(remaining, 8192))
                            if not skipped:
                                break
                            remaining -= len(skipped)
                        fragment = source.read(params.limit + 1)
                except (OSError, UnicodeError, IrisToolValidationError) as exc:
                    raise IrisToolExecutionError(
                        "已保存的工具材料不可读取", code="CONTEXT_SOURCE_UNAVAILABLE"
                    ) from exc
                return ContextReadPage(
                    params.ref,
                    params.representation,
                    params.offset,
                    params.offset + len(fragment[: params.limit]),
                    len(fragment) > params.limit,
                    fragment[: params.limit],
                )
        fragment = text[params.offset : params.offset + params.limit]
        end = params.offset + len(fragment)
        return ContextReadPage(
            params.ref,
            params.representation,
            params.offset,
            end,
            end < len(text),
            fragment,
        )

    def search(self, session_id: str, params: ContextSearchInput) -> ContextSearchPage:
        """最多扫描 200 条消息，每条消息仅返回首个命中。"""
        page = self.store.read_session_messages(session_id, start=params.after, limit=200)
        matches: list[ContextSearchHit] = []
        next_after = params.after
        query = params.query.casefold()
        for index, message in page.items:
            next_after = index + 1
            for ref, name, text in _search_texts(index, message):
                position = text.casefold().find(query)
                if position >= 0:
                    # 前缀 casefold 长度把匹配位置映回原始 Unicode 字符位置。
                    start = 0
                    folded = 0
                    while start < len(text) and folded < position:
                        folded += len(text[start].casefold())
                        start += 1
                    matches.append(
                        ContextSearchHit(
                            ref,
                            message.role.value,
                            name,
                            text[max(0, start - 40) : max(0, start - 40) + 240],
                        )
                    )
                    break
            if len(matches) == params.limit:
                break
        return ContextSearchPage(tuple(matches), next_after, next_after < page.total_count)


def _message_text(message: Msg) -> str:
    lines = [f"role={message.role.value} sender={message.sender}"]
    for index, block in enumerate(message.blocks):
        lines.append(f"[block {index} {block.type}]")
        if isinstance(block, TextBlock):
            lines.append(block.text)
        elif isinstance(block, ToolUseBlock):
            lines.append(
                f"{block.name} call_id={block.id}\n"
                f"{json.dumps(block.input, ensure_ascii=False, sort_keys=True)}"
            )
        else:
            lines.append(
                f"{block.name} call_id={block.tool_use_id} is_error={block.is_error}\n"
                f"{block.content}"
            )
    return "\n".join(lines)


def _search_texts(index: int, message: Msg) -> list[tuple[str, str, str]]:
    texts: list[tuple[str, str, str]] = []
    for block_index, block in enumerate(message.blocks):
        if isinstance(block, ToolResultBlock):
            texts.append((f"result:{index}:{block_index}", block.name, block.content))
        elif isinstance(block, TextBlock):
            texts.append((f"message:{index}", "", block.text))
        else:
            texts.append(
                (
                    f"message:{index}",
                    block.name,
                    json.dumps(block.input, ensure_ascii=False, sort_keys=True),
                )
            )
    return texts
