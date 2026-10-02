"""真实 YAML 工厂与 Native/Docker 集成共用的小型文件证据工具。"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from iris.hooks import HookEvent, HookHandler, HookResult, ToolAfterEvent, ToolAfterResult
from iris.tools import ToolCall, ToolMiddleware, ToolNext, ToolResult


def record(path: str | Path, value: str) -> None:
    """写一条可跨宿主/子进程观察的顺序证据。"""
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")


def records(path: Path) -> list[str]:
    """读取测试明确创建的 JSONL 证据。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def create_hook(*, path: str, label: str, feedback: str | None = None) -> HookHandler:
    """同步构造一个 async handler；构造次数与实际调用分别记录。"""
    record(path + ".created", label)

    async def handler(event: HookEvent) -> HookResult:
        record(path, f"{label}:{event.event}:{event.run_id}")
        if isinstance(event, ToolAfterEvent) and feedback is not None:
            return ToolAfterResult(feedback=feedback)
        return None

    return handler


class RecordingMiddleware(ToolMiddleware):
    """记录公开包装链的进入与返回，不改变业务结果。"""

    def __init__(self, path: str, label: str) -> None:
        self.path = path
        self.label = label

    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        """用实际工具名区分每次调用。"""
        record(self.path, f"{self.label}:before:{call.tool_name}")
        result = await call_next()
        record(self.path, f"{self.label}:after:{call.tool_name}")
        return result


def create_middleware(*, path: str, label: str) -> ToolMiddleware:
    """仅构造包装对象，留运行时执行 continuation。"""
    record(path + ".created", label)
    return RecordingMiddleware(path, label)


def python_hook(
    path: Path,
    event: str,
    label: str,
    *,
    feedback: str | None = None,
    tools: list[str] | None = None,
) -> dict[str, Any]:
    """生成进入真实 YAML 解析器的工厂声明。"""
    value: dict[str, Any] = {
        "name": label,
        "event": event,
        "handler": {
            "type": "python",
            "factory": "tests.hooks.config_extensions:create_hook",
            "options": {"path": str(path), "label": label, "feedback": feedback},
        },
    }
    if tools is not None:
        value["tools"] = tools
    return value


def script_command(script: Path, *, docker: bool = False) -> str:
    """Native 使用当前解释器，Docker 使用镜像解释器和 workspace 相对路径。"""
    arguments = ["python" if docker else sys.executable, "-u", script.name]
    return (
        shlex.join(arguments) if docker or os.name != "nt" else subprocess.list2cmdline(arguments)
    )


def write_hook_script(workspace: Path, *, gate: bool = False) -> Path:
    """实际读取 stdin 至 EOF，记录完整事件并返回唯一 JSON 协议。"""
    script = workspace / "hook.py"
    script.write_text(
        "import json, pathlib, sys, time\n"
        "event = json.loads(sys.stdin.buffer.read())\n"
        "with pathlib.Path('events.jsonl').open('a', encoding='utf-8') as output:\n"
        "    output.write(json.dumps(event) + '\\n')\n"
        "with pathlib.Path('order.jsonl').open('a', encoding='utf-8') as output:\n"
        "    output.write(json.dumps('script:' + event['event'] + ':' + event['run_id']) + '\\n')\n"
        + (
            "pathlib.Path('entered').touch()\n"
            "while not pathlib.Path('release').exists():\n    time.sleep(0.01)\n"
            if gate
            else ""
        )
        + "print('diagnostic-only', file=sys.stderr)\n"
        "print(json.dumps({'feedback': '脚本反馈'} if event['event'] == 'tool.after' else {}))\n",
        encoding="utf-8",
    )
    return script


def command_hook(event: str, command: str, *, tools: list[str] | None = None) -> dict[str, Any]:
    """脚本使用独立 Hook 期限。"""
    result: dict[str, Any] = {
        "name": "script-" + event,
        "event": event,
        "timeout_seconds": 20,
        "handler": {"type": "command", "command": command},
    }
    if tools is not None:
        result["tools"] = tools
    return result


def write_config(workspace: Path, hooks: list[dict[str, Any]], **extra: Any) -> Path:
    """写实际 YAML 文件，所有调用从公开路径加载。"""
    path = workspace / "agent.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "configured-agent",
                "model": "openai/scripted",
                "system": "test",
                "context_policy": {"enabled": False},
                "permissions": {"workspace": str(workspace), "execute": "allow", "writes": "allow"},
                "hooks": hooks,
                **extra,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path
