"""模板加载与渲染异常。"""

from .base import IrisError


class IrisTemplateError(IrisError):
    """模板相关错误的基类。"""


class IrisTemplateNotFoundError(IrisTemplateError):
    """找不到所需模板时抛出。"""
