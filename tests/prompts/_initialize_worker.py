"""为真实初始化进程设置发布屏障，不替代文件系统操作。"""

import sys
import time
from pathlib import Path

import iris.prompts.source as prompt_module


def main() -> None:
    """等待父测试准许后，发布带进程标记的第一个种子。"""
    workspace, coordination, label = sys.argv[1:]
    barrier = Path(coordination)
    original = prompt_module._publish_seed

    def publish_seed(target: Path, content: bytes) -> None:
        if target.name == "compaction.j2":
            content = label.encode() + b"\n" + content
            (barrier / f"{label}.ready").touch()
            deadline = time.monotonic() + 20
            while not (barrier / f"{label}.release").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("初始化测试发布屏障超时")
                time.sleep(0.01)
        original(target, content)

    prompt_module._publish_seed = publish_seed
    prompt_module.PromptSource.initialize(Path(workspace))


if __name__ == "__main__":
    main()
