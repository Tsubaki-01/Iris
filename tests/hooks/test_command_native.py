"""真实 Native 进程通过命令 Hook 协议派发，不依赖模型或公共配置入口。"""

from __future__ import annotations

import asyncio
import json
import platform
import sys
from pathlib import Path

import pytest
from PIL import Image

from iris.command import CommandBinding, CommandConfig, CommandEnvironment, CommandMode
from iris.command.native import NativeCommandService
from iris.hooks import RunStartedEvent, ToolAfterEvent, ToolBeforeEvent, event_to_dict
from iris.hooks._dispatch_types import CommandHookRegistration
from iris.hooks.command import CommandHookAdapter
from iris.hooks.dispatcher import HookDispatcher
from iris.message import TextBlock, image_block_from_saved
from iris.tools import ToolResult
from iris.utils.images import save_image

from .test_models import _run


def _binding(service: NativeCommandService) -> CommandBinding:
    """使用明显更短的普通命令期限，证明 adapter 采用自己的期限。"""
    host = platform.system()
    return CommandBinding(
        CommandConfig(timeout_seconds=0.001),
        service,
        CommandEnvironment(
            host, CommandMode.NATIVE, host, "cmd.exe" if sys.platform == "win32" else "/bin/sh"
        ),
    )


def _adapter(
    binding: CommandBinding, workspace: Path, name: str, timeout: float = 3
) -> CommandHookAdapter:
    """使用测试环境解释器执行工作区内的真实脚本。"""
    return CommandHookAdapter(
        binding=binding,
        workspace=workspace,
        command=f'"{sys.executable}" "{name}"',
        timeout_seconds=timeout,
    )


def _after(workspace: Path) -> ToolAfterEvent:
    """构造已知结果事件，模型反馈只有命令 JSON 可产生。"""
    return ToolAfterEvent(
        agent_id="agent",
        session_id="session",
        run_id="run",
        workspace=str(workspace),
        call_id="tool-call",
        tool_name="exec_command",
        arguments={"value": "中文"},
        result=ToolResult(tool_use_id="tool-call", tool_name="exec_command"),
        body_status="success",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("event_name", ["run.started", "tool.after"])
async def test_native_hook_receives_utf8_event_and_returns_typed_feedback(
    tmp_path: Path, event_name: str
) -> None:
    """图片输入及结果经过真实 stdin，脚本能读取两份图片引用并按事件返回。"""
    source = tmp_path / "原图.png"
    with Image.new("RGB", (8, 5), "red") as image:
        image.save(source)
    imported = image_block_from_saved(
        save_image(source, cache_dir=tmp_path / "cache"), name="原图.png"
    )
    content = [TextBlock(text="中文"), imported]
    event = _after(tmp_path)
    event.result.content = content
    if event_name == "run.started":
        event = RunStartedEvent(
            agent_id="agent",
            session_id="session",
            run_id="run",
            workspace=str(tmp_path),
            run=_run(),
            input=content,
        )
    (tmp_path / "check.py").write_text(
        "import json, pathlib, sys, time\n"
        "event = json.loads(sys.stdin.buffer.read())\n"
        "started = event['event'] == 'run.started'\n"
        "content = event['input'] if started else event['result']['content']\n"
        "for kind in ('original', 'model'):\n"
        "    data = pathlib.Path(content[1][kind]['path']).read_bytes()\n"
        "    pathlib.Path('received-' + kind + '.png').write_bytes(data)\n"
        "pathlib.Path('received.json').write_text(json.dumps(event), encoding='utf-8')\n"
        "time.sleep(0.04)\n"
        "print('diagnostic only', file=sys.stderr)\n"
        "print(json.dumps({} if started else {'feedback': '已检查：' + content[0]['text']}))\n",
        encoding="utf-8",
    )
    service = NativeCommandService(tmp_path)
    try:
        dispatcher = HookDispatcher(
            [
                CommandHookRegistration(
                    event.event, "check", _adapter(_binding(service), tmp_path, "check.py")
                ),
            ]
        )
        outcome = await dispatcher.dispatch(event)
        assert outcome.control is None and outcome.rejection is None
        assert outcome.feedback == (() if event_name == "run.started" else ("已检查：中文",))
        received = json.loads((tmp_path / "received.json").read_text(encoding="utf-8"))
        assert received == event_to_dict(event)
        assert (tmp_path / "received-original.png").read_bytes() == source.read_bytes()
        assert (tmp_path / "received-model.png").read_bytes() == imported.model.path.read_bytes()
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_native_before_denial_stops_remaining_handlers(tmp_path: Path) -> None:
    """真正的命令拒绝结果被解释为本次工具拒绝，不执行下一段脚本。"""
    (tmp_path / "deny.py").write_text(
        "import json, sys\njson.loads(sys.stdin.buffer.read())\n"
        "print(json.dumps({'deny_reason': '需要先检查输入'}))\n",
        encoding="utf-8",
    )
    (tmp_path / "later.py").write_text(
        "from pathlib import Path\nPath('unexpected').touch()\nprint('{}')\n",
        encoding="utf-8",
    )
    service = NativeCommandService(tmp_path)
    binding = _binding(service)
    try:
        dispatcher = HookDispatcher(
            [
                CommandHookRegistration(
                    "tool.before", "deny", _adapter(binding, tmp_path, "deny.py")
                ),
                CommandHookRegistration(
                    "tool.before", "later", _adapter(binding, tmp_path, "later.py")
                ),
            ]
        )
        outcome = await dispatcher.dispatch(
            ToolBeforeEvent(
                agent_id="agent",
                session_id="session",
                run_id="run",
                workspace=str(tmp_path),
                call_id="tool-call",
                tool_name="exec_command",
                arguments={},
            )
        )
        assert outcome.rejection is not None
        assert outcome.rejection.code == "HOOK_REJECTED"
        assert outcome.rejection.reason == "需要先检查输入"
        assert not (tmp_path / "unexpected").exists()
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_native_timeout_drains_before_next_after_handler(tmp_path: Path) -> None:
    """脚本自身 timeout 已收口后，后续 after 处理器仍可正常运行。"""
    (tmp_path / "slow.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    (tmp_path / "next.py").write_text('print(\'{"feedback": "next"}\')\n', encoding="utf-8")
    service = NativeCommandService(tmp_path)
    binding = _binding(service)
    try:
        dispatcher = HookDispatcher(
            [
                CommandHookRegistration(
                    "tool.after", "slow", _adapter(binding, tmp_path, "slow.py", 0.1)
                ),
                CommandHookRegistration(
                    "tool.after", "next", _adapter(binding, tmp_path, "next.py")
                ),
            ]
        )
        outcome = await asyncio.wait_for(dispatcher.dispatch(_after(tmp_path)), 5)
        assert outcome.control is None
        assert outcome.feedback == ("next",)
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_native_cancel_stops_process_before_returning_control(tmp_path: Path) -> None:
    """取消实际脚本，停止事实交回且不会启动剩余处理器。"""
    (tmp_path / "slow.py").write_text(
        "import json, pathlib, sys, time\n"
        "json.loads(sys.stdin.buffer.read())\n"
        "pathlib.Path('ready').touch()\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    (tmp_path / "later.py").write_text(
        "from pathlib import Path\nPath('unexpected').touch()\nprint('{}')\n",
        encoding="utf-8",
    )
    service = NativeCommandService(tmp_path)
    binding = _binding(service)
    dispatcher = HookDispatcher(
        [
            CommandHookRegistration(
                "tool.after", name, _adapter(binding, tmp_path, name + ".py", 30)
            )
            for name in ("slow", "later")
        ]
    )
    task = asyncio.create_task(dispatcher.dispatch(_after(tmp_path)))
    try:
        async with asyncio.timeout(5):
            while not (tmp_path / "ready").exists():
                await asyncio.sleep(0.01)
        task.cancel()
        outcome = await asyncio.wait_for(task, 5)
        assert outcome.control is not None
        assert outcome.control.origin == "task_cancelled"
        assert outcome.control.stop_slot is not None
        receipt = outcome.control.stop_slot.receipt
        assert receipt is not None
        await service.wait_drained(receipt)
        assert not service._calls
        assert outcome.feedback == ()
        assert not (tmp_path / "unexpected").exists()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await service.aclose()
