"""由两个后端直接执行的同源 Python 启动器，只依赖标准库。"""

from __future__ import annotations

import linecache
import os
import sys
import traceback
from pathlib import Path
from types import ModuleType


def main() -> None:
    """加载一次源码，以真实 __main__ 模块执行并保留原始源码回溯。"""
    code = Path(sys.argv[1]).read_text(encoding="utf-8")
    filename = "<iris-python>"
    linecache.cache[filename] = (len(code), None, code.splitlines(keepends=True), filename)
    sys.path[0] = os.getcwd()
    sys.argv = [filename]
    sys.excepthook = traceback.print_exception
    module = ModuleType("__main__")
    sys.modules["__main__"] = module
    exec(compile(code, filename, "exec", dont_inherit=True), module.__dict__)


if __name__ == "__main__":
    main()
