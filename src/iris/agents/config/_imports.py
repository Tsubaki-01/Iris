"""配置中 Python 可调用引用的唯一导入边界。"""

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ...exceptions import IrisConfigError


def import_ref(ref: str) -> Callable[..., Any]:
    """导入 module:attribute 引用，并在配置边界确认目标可调用。"""
    module_name, separator, attribute_name = ref.partition(":")
    if not separator or not module_name or not attribute_name:
        raise IrisConfigError("Python 引用必须使用 module:attribute 格式", ref=ref)
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise IrisConfigError("Python 引用模块无法导入", ref=ref) from exc
    try:
        target = getattr(module, attribute_name)
    except AttributeError as exc:
        raise IrisConfigError("Python 引用属性不存在", ref=ref) from exc
    if not callable(target):
        raise IrisConfigError("Python 引用目标不可调用", ref=ref)
    return target
