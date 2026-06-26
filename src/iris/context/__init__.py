"""上下文系统对外 API。"""

from .builder import CONTEXT_SENDER, ContextBuilder
from .config import load_context_build_input
from .models import (
    ContextBuildInput,
    ContextBuildOutput,
    ContextSection,
    ContextSlot,
)
from .renderer import ContextTemplateRenderer, ContextXmlRenderer

__all__ = [
    "CONTEXT_SENDER",
    "ContextBuildInput",
    "ContextBuildOutput",
    "ContextBuilder",
    "ContextSection",
    "ContextSlot",
    "ContextTemplateRenderer",
    "ContextXmlRenderer",
    "load_context_build_input",
]
