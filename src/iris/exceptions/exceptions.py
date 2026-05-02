from __future__ import annotations


class IrisError(Exception):
    """Base exception for all Iris-specific errors."""


class IrisConfigError(IrisError, ValueError):
    """Raised when there is a configuration issue, such as missing required parameters or invalid values."""


class IrisToolError(IrisError):
    """Raised when a tool fails during execution."""


class IrisValidationError(IrisError):
    """Raised when input or configuration validation fails."""


class IrisParserError(IrisError):
    """Raised when parsing structured content fails."""


class IrisExecutionError(IrisError):
    """Raised when an error occurs during the execution of a task."""
