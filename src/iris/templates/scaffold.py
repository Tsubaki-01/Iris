"""官方模板 scaffold 工具。"""

from __future__ import annotations

from pathlib import Path
from shutil import copy2

from ..exceptions import IrisTemplateError, IrisTemplateNotFoundError

_BUILTIN_TEMPLATE_ROOT = Path(__file__).parent / "builtin"


def scaffold_template(
    template_name: str,
    target_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """复制官方模板到目标目录。

    Args:
        template_name (str): 官方模板名称，例如 `file-agent`。
        target_dir (str | Path): 要写入模板文件的目标目录。
        overwrite (bool): 是否允许覆盖已有文件，默认不允许。

    Returns:
        list[Path]: 实际写入的目标文件路径。

    Raises:
        IrisTemplateNotFoundError: 模板不存在时抛出。
        IrisTemplateError: 目标文件已存在且不允许覆盖时抛出。
    """
    template_dir = _resolve_builtin_template(template_name)
    target_path = Path(target_dir)
    files = [path for path in sorted(template_dir.rglob("*")) if path.is_file()]
    writes = [target_path / path.relative_to(template_dir) for path in files]
    conflicts = [path for path in writes if path.exists()]
    if conflicts and not overwrite:
        raise IrisTemplateError(
            "目标目录已存在模板文件",
            template=template_name,
            conflicts=[str(path) for path in conflicts],
        )

    target_path.mkdir(parents=True, exist_ok=True)
    for source, destination in zip(files, writes, strict=True):
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, destination)
    return writes


def _resolve_builtin_template(template_name: str) -> Path:
    """解析官方模板目录。"""
    template_dir = _BUILTIN_TEMPLATE_ROOT / template_name
    if not template_dir.is_dir():
        available = sorted(
            path.name for path in _BUILTIN_TEMPLATE_ROOT.glob("*") if path.is_dir()
        )
        raise IrisTemplateNotFoundError(
            "官方模板不存在",
            template=template_name,
            available=available,
        )
    return template_dir


__all__ = ["scaffold_template"]
