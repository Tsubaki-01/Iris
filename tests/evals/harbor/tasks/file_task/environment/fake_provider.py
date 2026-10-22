"""容器内确定性 OpenAI 协议服务，仅供接入测试。"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SCENARIO = json.loads(Path("/fixture/scenario.json").read_text())


def actions() -> list[tuple[str, dict[str, Any]]]:
    """通过真正的 Iris 文件/命令工具完成或故意做错任务。"""
    if SCENARIO["mode"] == "timeout":
        script = (
            "import time\nfrom pathlib import Path\n"
            "while True:\n"
            "    with Path('ticks.txt').open('a') as output:\n"
            "        output.write('tick\\n')\n"
            "    time.sleep(0.05)\n"
        )
    else:
        delta = 1 if SCENARIO["mode"] == "wrong" else 0
        script = (
            "import csv, json, sys\nfrom pathlib import Path\n"
            "assert sys.version_info[:2] == (3, 10), sys.version\n"
            "total = sum(int(row['value']) for row in csv.DictReader(open('input.csv')))\n"
            f"Path('answer.json').write_text(json.dumps({{'total': total + {delta}}}))\n"
            "Path('task-python.txt').write_text(sys.executable + '\\n' + sys.version)\n"
            "print('task complete')\n"
        )
    sequence = [
        ("write_file", {"file_path": "solve.py", "content": script}),
        ("exec_command", {"command": "python solve.py", "cwd": "."}),
        ("read_file", {"file_path": "pixel.png"}),
        ("exec_command", {"command": "python -c \"print('archive-text-' * 6000)\"", "cwd": "."}),
    ]
    if SCENARIO["mode"] == "partial":
        sequence.append(
            (
                "exec_command",
                {
                    "command": "find .iris/tool-results -type f -delete",
                    "cwd": ".",
                },
            )
        )
    return sequence


class Handler(BaseHTTPRequestHandler):
    """只实现测试使用的非流式 chat completion。"""

    step = 0

    def do_POST(self) -> None:  # noqa: N802
        """每次请求返回一个预定工具动作，最终返回正常结束。"""
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if SCENARIO["mode"] == "provider_failure":
            self._reply(
                401,
                {
                    "error": {
                        "message": "fixture provider failure",
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }
                },
            )
            return
        available = actions()
        step = type(self).step
        type(self).step += 1
        message: dict[str, Any] = {"role": "assistant", "content": "完成"}
        finish = "stop"
        if step < len(available):
            name, arguments = available[step]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{step}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        self._reply(
            200,
            {
                "id": f"response-{step}",
                "object": "chat.completion",
                "created": 1,
                "model": request["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        )

    def _reply(self, status: int, value: dict[str, Any]) -> None:
        """写入确定性的完整 HTTP JSON 响应。"""
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
