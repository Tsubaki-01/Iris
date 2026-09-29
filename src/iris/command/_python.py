"""命令后端共用的 Python 启动器文本与本地源码准备。"""

import tempfile
from importlib.resources import files
from pathlib import Path

PYTHON_LOADER_SOURCE = (
    files("iris.command").joinpath("_python_loader.py").read_text(encoding="utf-8")
)


def write_python_source(code: str) -> Path:
    """写入并关闭调用专属源码，避免 Windows 仍持有临时文件句柄。"""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as handle:
        path = Path(handle.name)
        try:
            handle.write(code)
        except OSError:
            handle.close()
            path.unlink(missing_ok=True)
            raise
    return path
