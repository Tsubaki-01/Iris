"""上下文系统对外 API。"""

from .builder import CONTEXT_SENDER, ContextBuilder
from .config import load_context_build_input
from .models import (
    ContextBuildInput,
    ContextBuildOutput,
    ContextSection,
    ContextSlot,
)
from .renderer import ContextXmlRenderer
from .source import ContextBuildScope, ContextContribution, ContextSnapshot, ContextSource

__all__ = [
    "CONTEXT_SENDER",
    "ContextBuildInput",
    "ContextBuildOutput",
    "ContextBuildScope",
    "ContextBuilder",
    "ContextContribution",
    "ContextSection",
    "ContextSlot",
    "ContextSnapshot",
    "ContextSource",
    "ContextXmlRenderer",
    "load_context_build_input",
]
