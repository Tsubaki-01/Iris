"""仅在示例容器运行的离线依赖与后台服务控制脚本。"""

import json
import shutil
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.request import urlopen

PORT = 8765
DEPENDENCY = Path("/tmp/iris_example_stats.py")


def fetch() -> dict[str, int]:
    """从当前容器内的受控服务读取 JSON。"""
    with urlopen(f"http://127.0.0.1:{PORT}/", timeout=2) as response:
        return json.load(response)


def serve() -> None:
    """后台进程借用容器文件层中的离线模块。"""
    sys.path.insert(0, "/tmp")
    from iris_example_stats import total

    class Handler(BaseHTTPRequestHandler):
        """只返回示例固定数据。"""

        def do_GET(self) -> None:  # noqa: N802
            """响应本地请求。"""
            body = json.dumps({"total": total([3, 4, 5])}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


def main() -> None:
    """准备依赖、启动服务、请求数据，或确认整体停止后的状态。"""
    action = sys.argv[1]
    if action == "serve":
        serve()
        return
    if action == "start":
        shutil.copyfile("offline_stats.py", DEPENDENCY)
        subprocess.Popen(
            [sys.executable, __file__, "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 5
        while True:
            try:
                payload = fetch()
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
    elif action == "check":
        payload = fetch()
    else:
        with socket.socket() as connection:
            connection.settimeout(1)
            running = connection.connect_ex(("127.0.0.1", PORT)) == 0
        payload = {"dependency_retained": DEPENDENCY.is_file(), "service_running": running}
    Path(sys.argv[2]).write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
