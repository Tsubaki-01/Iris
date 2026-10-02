"""显式启用的真实 Docker 命令 Hook 协议与取消验收。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image

from iris.command import CommandBinding, CommandConfig, CommandEnvironment, CommandMode
from iris.command.docker import DockerCommandService
from iris.hooks import ToolAfterEvent, event_to_dict
from iris.hooks._dispatch_types import CommandHookRegistration
from iris.hooks.command import CommandHookAdapter
from iris.hooks.dispatcher import HookDispatcher
from iris.message import TextBlock, image_block_from_saved
from iris.sandbox import DockerConfig
from iris.tools import ToolResult
from iris.utils.images import save_image


@pytest.fixture(autouse=True)
def require_real_docker(request: pytest.FixtureRequest) -> None:
    """只有显式开关才创建容器；失败不能作为跳过掩盖。"""
    if not request.config.getoption("--run-docker"):
        pytest.skip("真实 Docker 需显式 --run-docker")


def _dispatcher(
    service: DockerCommandService, workspace: Path, names: tuple[str, ...]
) -> HookDispatcher:
    """adapter 使用独立期限与同一个已准备的 Docker 命令服务。"""
    binding = CommandBinding(
        CommandConfig(mode=CommandMode.DOCKER, timeout_seconds=0.001),
        service,
        CommandEnvironment("Windows", CommandMode.DOCKER, "Linux", "/bin/sh"),
    )
    return HookDispatcher(
        [
            CommandHookRegistration(
                "tool.after",
                name,
                CommandHookAdapter(
                    binding=binding,
                    workspace=workspace,
                    command=f"python {name}.py",
                    timeout_seconds=30,
                ),
            )
            for name in names
        ]
    )


def _event(workspace: Path) -> ToolAfterEvent:
    """真实命令消费的工具完成事件。"""
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
async def test_real_docker_hook_json_and_independent_timeout(tmp_path: Path) -> None:
    """图片事件保留宿主路径，脚本按 workspace 映射读取两份图片并返回反馈。"""
    source = tmp_path / "原图.png"
    with Image.new("RGB", (8, 5), "red") as image:
        image.save(source)
    imported = image_block_from_saved(
        save_image(source, cache_dir=tmp_path / "cache"), name="原图.png"
    )
    event = _event(tmp_path)
    event.result.content = [TextBlock(text="中文"), imported]
    (tmp_path / "check.py").write_text(
        "import json, pathlib, sys, time\n"
        "event = json.loads(sys.stdin.buffer.read())\n"
        "path_type = (pathlib.PureWindowsPath\n"
        "    if pathlib.PureWindowsPath(event['workspace']).drive else pathlib.PurePosixPath)\n"
        "for kind in ('original', 'model'):\n"
        "    host_path = path_type(event['result']['content'][1][kind]['path'])\n"
        "    relative = host_path.relative_to(path_type(event['workspace']))\n"
        "    data = pathlib.Path('/workspace', *relative.parts).read_bytes()\n"
        "    pathlib.Path('received-' + kind + '.png').write_bytes(data)\n"
        "pathlib.Path('received.json').write_text(json.dumps(event), encoding='utf-8')\n"
        "time.sleep(0.04)\n"
        "print('diagnostic only', file=sys.stderr)\n"
        "print(json.dumps({'feedback': '已检查：' + event['arguments']['value']}))\n",
        encoding="utf-8",
    )
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    try:
        await service.prepare()
        outcome = await _dispatcher(service, tmp_path, ("check",)).dispatch(event)
        assert outcome.control is None
        assert outcome.feedback == ("已检查：中文",)
        received = json.loads((tmp_path / "received.json").read_text(encoding="utf-8"))
        assert received == event_to_dict(event)
        assert (tmp_path / "received-original.png").read_bytes() == source.read_bytes()
        assert (tmp_path / "received-model.png").read_bytes() == imported.model.path.read_bytes()
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_real_docker_hook_cancel_does_not_start_next_script(tmp_path: Path) -> None:
    """SDK 取消穿过 Dispatcher/adapter，收据排空后容器保持停止。"""
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
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    await service.prepare()
    task = asyncio.create_task(
        _dispatcher(service, tmp_path, ("slow", "later")).dispatch(_event(tmp_path))
    )
    try:
        async with asyncio.timeout(10):
            while not (tmp_path / "ready").exists():
                await asyncio.sleep(0.02)
        task.cancel()
        outcome = await asyncio.wait_for(task, 15)
        assert outcome.control is not None
        assert outcome.control.origin == "task_cancelled"
        assert outcome.control.stop_slot is not None
        receipt = outcome.control.stop_slot.receipt
        assert receipt is not None
        await service.wait_drained(receipt)
        assert not (await service._sandbox.container.show())["State"]["Running"]
        assert outcome.feedback == ()
        assert not (tmp_path / "unexpected").exists()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await service.aclose()
