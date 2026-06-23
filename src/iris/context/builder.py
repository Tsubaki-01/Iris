"""上下文系统构建器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import IrisContextError
from ..message import Msg
from .models import (
    ContextBuildInput,
    ContextBuildOutput,
    ContextPosition,
    ContextSlot,
    MemoryContextInput,
    MemoryContextItem,
)
from .renderer import ContextTemplateRenderer, ContextXmlRenderer

CONTEXT_SENDER = "context"
POSITION_ORDER: dict[ContextPosition, int] = {
    ContextPosition.SYSTEM: 0,
    ContextPosition.MEMORY: 1,
    ContextPosition.BEFORE_CURRENT_INPUT: 2,
}


class ContextBuilder:
    """构建 API 原生的 context 消息增量。"""

    def __init__(
        self,
        *,
        xml_renderer: ContextXmlRenderer | None = None,
        template_renderer: ContextTemplateRenderer | None = None,
    ) -> None:
        self.xml_renderer = xml_renderer or ContextXmlRenderer()
        self.template_renderer = template_renderer or ContextTemplateRenderer()

    def build(self, input_data: ContextBuildInput) -> ContextBuildOutput:
        """构建 system、memory 和 current input 前置 context 消息。"""
        slots = _sort_slots(self._build_slots(input_data))
        slots_by_position = _group_slots(slots)
        warnings = (
            list(input_data.memory.warnings)
            if input_data.memory is not None and input_data.memory.enabled
            else []
        )

        system_xml = self._render_position(
            ContextPosition.SYSTEM,
            slots_by_position[ContextPosition.SYSTEM],
            input_data=input_data,
        )
        memory_xml = self._render_optional_position(
            ContextPosition.MEMORY,
            slots_by_position[ContextPosition.MEMORY],
            input_data=input_data,
        )
        runtime_xml = self._render_optional_position(
            ContextPosition.BEFORE_CURRENT_INPUT,
            slots_by_position[ContextPosition.BEFORE_CURRENT_INPUT],
            input_data=input_data,
        )

        return ContextBuildOutput(
            system_message=Msg.system(system_xml),
            memory_messages=(
                [Msg.user(memory_xml, sender=CONTEXT_SENDER)]
                if memory_xml is not None
                else []
            ),
            before_current_input_messages=(
                [Msg.user(runtime_xml, sender=CONTEXT_SENDER)]
                if runtime_xml is not None
                else []
            ),
            slots=slots,
            metadata=dict(input_data.metadata),
            warnings=warnings,
        )

    def _build_slots(self, input_data: ContextBuildInput) -> list[ContextSlot]:
        slots: list[ContextSlot] = []
        if input_data.system.mode == "inline":
            slots.append(
                ContextSlot(
                    name="base_instructions",
                    position=ContextPosition.SYSTEM,
                    content=input_data.system.inline or "",
                    order=0,
                )
            )
        slots.extend(self._memory_slots(input_data.memory))
        if input_data.environment_state:
            slots.append(
                ContextSlot(
                    name="environment_state",
                    position=ContextPosition.BEFORE_CURRENT_INPUT,
                    content=input_data.environment_state,
                    order=10,
                )
            )
        if input_data.turn_constraints:
            slots.append(
                ContextSlot(
                    name="turn_constraints",
                    position=ContextPosition.BEFORE_CURRENT_INPUT,
                    content=input_data.turn_constraints,
                    order=20,
                )
            )
        slots.extend(input_data.slots)
        return [slot for slot in slots if slot.enabled]

    def _memory_slots(self, memory: MemoryContextInput | None) -> list[ContextSlot]:
        if memory is None or not memory.enabled:
            return []
        slots: list[ContextSlot] = []
        for index, item in enumerate(memory.entries):
            attributes = _memory_attributes(item, index=index)
            slots.append(
                ContextSlot(
                    name="memory",
                    position=ContextPosition.MEMORY,
                    content=item.text,
                    order=10 + index,
                    attributes=attributes,
                )
            )
        if memory.warnings:
            slots.append(
                ContextSlot(
                    name="memory_warnings",
                    position=ContextPosition.MEMORY,
                    content=memory.warnings,
                    order=1000,
                )
            )
        return slots

    def _render_position(
        self,
        position: ContextPosition,
        slots: list[ContextSlot],
        *,
        input_data: ContextBuildInput,
    ) -> str:
        template_path = self._template_path(position, input_data)
        if template_path is not None:
            return self.template_renderer.render_file(
                template_path,
                self._template_context(input_data, slots),
            )
        return self.xml_renderer.render_position(
            position, slots, version=input_data.version
        )

    def _render_optional_position(
        self,
        position: ContextPosition,
        slots: list[ContextSlot],
        *,
        input_data: ContextBuildInput,
    ) -> str | None:
        if not slots:
            return None
        return self._render_position(position, slots, input_data=input_data)

    def _template_path(
        self,
        position: ContextPosition,
        input_data: ContextBuildInput,
    ) -> Path | None:
        raw_path: str | Path | None = None
        if position == ContextPosition.SYSTEM:
            raw_path = (
                input_data.system.template_path
                if input_data.system.mode == "template"
                else input_data.templates.system
            )
        elif position == ContextPosition.MEMORY:
            raw_path = input_data.templates.memory
        elif position == ContextPosition.BEFORE_CURRENT_INPUT:
            raw_path = input_data.templates.before_current_input
        if raw_path is None:
            return None
        path = Path(raw_path)
        if path.is_absolute():
            raise IrisContextError("context 模板路径不能是绝对路径", path=str(path))
        if ".." in path.parts:
            raise IrisContextError("context 模板路径不能包含上级目录", path=str(path))
        base_dir = input_data.template_base_dir or input_data.workspace_root
        if base_dir is None:
            raise IrisContextError(
                "context 相对模板路径必须提供 template_base_dir 或 workspace_root"
            )
        return (base_dir / path).resolve()

    def _template_context(
        self,
        input_data: ContextBuildInput,
        slots: list[ContextSlot],
    ) -> dict[str, Any]:
        memory = input_data.memory or MemoryContextInput(enabled=False)
        return {
            "version": input_data.version,
            "agent_id": input_data.agent_id,
            "session_id": input_data.session_id,
            "workspace_root": (
                str(input_data.workspace_root) if input_data.workspace_root else ""
            ),
            "system": {
                "inline": input_data.system.inline or "",
                **input_data.system.variables,
            },
            "memory": {
                "enabled": memory.enabled,
                "entries": [_memory_item_dict(item) for item in memory.entries],
                "warnings": list(memory.warnings) if memory.enabled else [],
                "query_from_current_input": memory.query_from_current_input,
            },
            "environment_state": dict(input_data.environment_state),
            "before_current_input": {
                "environment_state": dict(input_data.environment_state),
                "turn_constraints": list(input_data.turn_constraints),
            },
            "slots": [slot.model_dump(mode="json") for slot in slots],
            "metadata": dict(input_data.metadata),
            "current_input_available": input_data.current_input is not None,
        }


def _sort_slots(slots: list[ContextSlot]) -> list[ContextSlot]:
    return sorted(
        slots, key=lambda item: (POSITION_ORDER[item.position], item.order, item.name)
    )


def _group_slots(slots: list[ContextSlot]) -> dict[ContextPosition, list[ContextSlot]]:
    grouped = {position: [] for position in ContextPosition}
    for slot in slots:
        grouped[slot.position].append(slot)
    for position, items in grouped.items():
        grouped[position] = sorted(items, key=lambda item: (item.order, item.name))
    return grouped


def _memory_attributes(item: MemoryContextItem, *, index: int) -> dict[str, str]:
    attributes = {"index": str(index)}
    if item.id:
        attributes["id"] = item.id
    if item.source:
        attributes["source"] = item.source
    if item.score is not None:
        attributes["score"] = str(item.score)
    return attributes


def _memory_item_dict(item: MemoryContextItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "source": item.source,
        "score": item.score,
        "text": item.text,
        "metadata": dict(item.metadata),
    }


__all__ = ["CONTEXT_SENDER", "ContextBuilder"]
