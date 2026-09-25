"""工具有限本地 IO 的执行与确定结果收口。"""

import asyncio
from collections.abc import Callable


async def run_tool_io[T](operation: Callable[[], T]) -> T:
    """在线程完成一次本地操作，取消等待后仍收回结果，不清除调用方取消意图。"""
    job = asyncio.create_task(asyncio.to_thread(operation))
    while True:
        try:
            return await asyncio.shield(job)
        except asyncio.CancelledError:
            # 外层可被重复取消；作业自己的取消或异常仍由 result 原样传播。
            if job.done():
                return job.result()
