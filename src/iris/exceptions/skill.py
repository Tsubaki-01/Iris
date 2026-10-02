"""Skill 发现、格式与路径异常。"""

from .base import IrisError


class IrisSkillError(IrisError, ValueError):
    """Skill 子系统错误的基类。"""


class IrisSkillFormatError(IrisSkillError):
    """Skill 文件格式无效。"""


class IrisSkillPathError(IrisSkillError):
    """Skill 路径不满足 workspace 边界。"""


class IrisSkillNotFoundError(IrisSkillError):
    """按名称找不到 Skill。"""
