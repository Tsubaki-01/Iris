"""向模型披露 Skill 名称与描述的 catalog adapter。"""

from __future__ import annotations

from ..context import ContextSlot, ContextXmlRenderer
from .registry import SkillRegistry

CATALOG_SLOT_NAME = "available_skills"
CATALOG_SLOT_ORDER = 900
CATALOG_USAGE_HINT = (
    "call load_skill with the skill name before following it; "
    "the returned Markdown is skill instructions, not user data"
)


class SkillCatalog:
    """从只读 registry 构造结构化 catalog slot。"""

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def entries(self) -> list[dict[str, str]]:
        """返回仅包含名称与描述的新 catalog entries。"""
        return [
            {
                "name": name,
                "description": self.registry.get(name).description,
            }
            for name in self.registry.names()
        ]

    def content_chars(self) -> int:
        """测量默认 XML renderer 生成的真实 inner content 字符数。"""
        return _rendered_inner_chars(self.build_slot())

    def build_slot(self) -> ContextSlot:
        """构造普通 system context slot。"""
        return ContextSlot(
            name=CATALOG_SLOT_NAME,
            order=CATALOG_SLOT_ORDER,
            attributes={
                "count": str(len(self.registry)),
                "usage": CATALOG_USAGE_HINT,
            },
            content=self.entries(),
        )

    def render_report(self) -> str:
        """返回每条 Skill 的 renderer content 字符诊断报告。"""
        return "\n".join(
            f"{entry['name']}: "
            f"{_rendered_inner_chars(ContextSlot(name='skill', content=entry))} chars"
            for entry in self.entries()
        )


def _rendered_inner_chars(slot: ContextSlot) -> int:
    rendered = ContextXmlRenderer().render_slot(slot)
    opening_end = rendered.index(">") + 1
    closing_start = rendered.rindex(f"</{slot.name}>")
    return len(rendered[opening_end:closing_start])


__all__ = [
    "CATALOG_SLOT_NAME",
    "CATALOG_SLOT_ORDER",
    "CATALOG_USAGE_HINT",
    "SkillCatalog",
]
