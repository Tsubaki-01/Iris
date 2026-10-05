"""共享的单文件完整文本发布，不拥有路径策略或跨进程调度。"""

import tempfile
from pathlib import Path


def atomic_write_text(path: Path, content: str) -> None:
    """通过同目录完整临时文件原子替换目标。

    Args:
        path: 调用领域已确定的目标路径。
        content: 要发布的 UTF-8 文本。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
