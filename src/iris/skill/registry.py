"""Skill discovery 结果的只读 registry。"""

from __future__ import annotations

from collections.abc import Iterable

from ..exceptions import IrisSkillError, IrisSkillNotFoundError
from .models import (
    SkillDiagnostic,
    SkillDiscoveryResult,
    SkillMetadata,
)


class SkillRegistry:
    """保存一次 discovery 快照的稳定只读索引。"""

    def __init__(self, result: SkillDiscoveryResult) -> None:
        self._skills = result.skills
        self._diagnostics = result.diagnostics
        self._by_name: dict[str, SkillMetadata] = {}
        for skill in self._skills:
            if skill.name in self._by_name:
                raise IrisSkillError(
                    "Skill discovery 结果包含重复名称",
                    name=skill.name,
                )
            self._by_name[skill.name] = skill
        self._names = tuple(skill.name for skill in self._skills)

    def get(self, name: str) -> SkillMetadata:
        """按精确名称返回 metadata。"""
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise IrisSkillNotFoundError(
                f"Skill 不存在: {name}",
                name=name,
                available=self.names(),
            ) from exc

    def has(self, name: str) -> bool:
        """返回 registry 是否包含精确名称。"""
        return name in self._by_name

    def names(self) -> tuple[str, ...]:
        """返回 discovery 顺序下的稳定名称元组。"""
        return self._names

    def missing(self, names: Iterable[str]) -> tuple[str, ...]:
        """按输入首次出现顺序返回去重后的缺失名称。"""
        missing: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name not in self._by_name and name not in seen:
                missing.append(name)
                seen.add(name)
        return tuple(missing)

    @property
    def diagnostics(self) -> tuple[SkillDiagnostic, ...]:
        """返回 discovery 产生的原始诊断元组。"""
        return self._diagnostics

    def __len__(self) -> int:
        return len(self._skills)


__all__ = ["SkillRegistry"]
