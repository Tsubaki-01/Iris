"""导入本地图片，通过 YAML 配置的模型完成一次看图对话并保存到 SQLite。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from iris.config import init_config, is_config_initialized
from iris.harness import AgentRunner, AgentRunOptions, AgentRunRequest, RunLimits, RunResult
from iris.lifecycle import RunStopReason
from iris.message import ImageBlock, TextBlock

CONFIG_PATH = Path(__file__).with_name("agent.yaml")
DEFAULT_PROMPT = "请描述这张图片中的内容，并说明你能看清的文字。"


async def run_agent(
    *, config_path: Path, image_path: Path, prompt: str, session_id: str
) -> RunResult:
    """导入图片并完成一次 run，结束时关闭 runner。

    Args:
        config_path: Agent YAML 的路径。
        image_path: 图片来源路径；相对路径以当前工作目录解析。
        prompt: 与图片一起提交的文字；空字符串表示纯图片输入。
        session_id: 图片缓存和持久化会话共用的标识。

    Returns:
        持久化的运行结果，包含状态、最终回答或错误。
    """
    runner = AgentRunner.from_config_path(config_path)
    try:
        image = await runner.import_image(
            image_path.resolve(), session_id=session_id, name=image_path.name
        )
        content: list[TextBlock | ImageBlock] = [TextBlock(text=prompt)] if prompt else []
        content.append(image)
        return await runner.start(
            AgentRunRequest(input=content, session_id=session_id),
            options=AgentRunOptions(limits=RunLimits(max_model_steps=8)),
        )
    finally:
        await runner.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """提交本地图片，输出完整 RunResult JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--session-id", default=f"image-{uuid4().hex}")
    args = parser.parse_args(argv)
    if not is_config_initialized():
        init_config(env_file=str(args.env_file) if args.env_file is not None else None)
    result = asyncio.run(
        run_agent(
            config_path=args.config,
            image_path=args.image,
            prompt=args.prompt,
            session_id=args.session_id,
        )
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.run.stop_reason is RunStopReason.COMPLETED else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
