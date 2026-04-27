from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .exceptions import (
        IrisConfigError,
        IrisError,
        IrisExecutionError,
        IrisParserError,
        IrisToolError,
        IrisValidationError,
    )


__all__ = [
    "IrisError",
    "IrisConfigError",
    "IrisToolError",
    "IrisValidationError",
    "IrisParserError",
    "IrisExecutionError",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import exceptions as _exceptions

        return getattr(_exceptions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
