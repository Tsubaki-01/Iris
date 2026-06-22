"""上下文系统对外 API。"""

from .builder import CONTEXT_SENDER, ContextBuilder
from .config import (
    BeforeCurrentInputConfig,
    ContextYamlConfig,
    SystemContentConfig,
    load_context_build_input,
    load_context_config,
)
from .models import (
    ContextBuildInput,
    ContextBuildOutput,
    ContextPosition,
    ContextSlot,
    ContextTemplateSpec,
    MemoryContextInput,
    MemoryContextItem,
    SystemPromptSpec,
)
from .renderer import ContextTemplateRenderer, ContextXmlRenderer

__all__ = [
    "CONTEXT_SENDER",
    "BeforeCurrentInputConfig",
    "ContextBuildInput",
    "ContextBuildOutput",
    "ContextBuilder",
    "ContextPosition",
    "ContextSlot",
    "ContextTemplateRenderer",
    "ContextTemplateSpec",
    "ContextXmlRenderer",
    "ContextYamlConfig",
    "MemoryContextInput",
    "MemoryContextItem",
    "SystemContentConfig",
    "SystemPromptSpec",
    "load_context_build_input",
    "load_context_config",
]
