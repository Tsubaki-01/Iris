"""固定 provider 的显式环境保存与原路径恢复演示。"""

import argparse
import asyncio
import json
import zipfile
from pathlib import Path

from iris.agents import AgentConfig
from iris.exceptions import IrisRunStateError
from iris.harness import AgentRunner
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import AgentRunRequest, RunPhase

from ..command._scripted import ScriptedProvider, done, require_completed, tool


def build_demo_config(workspace: Path, database: Path, image: str) -> AgentConfig:
    """声明独立 SQLite、Docker 与显式环境恢复，只注册演示所需工具。"""
    return AgentConfig.model_validate(
        {
            "name": "environment-demo",
            "model": "openai/offline-example",
            "system": "按脚本执行。",
            "permissions": {
                "workspace": str(workspace.resolve()),
                "writes": "allow",
                "execute": "allow",
            },
            "session": {"backend": "sqlite", "path": str(database.resolve())},
            "command": {"mode": "docker", "docker": {"image": image}},
            "environment_recovery": {"enabled": True},
            "tools": {"builtin": ["exec.python", "human.ask"]},
        }
    )


def _probe_wheel(workspace: Path) -> Path:
    """生成无需联网或构建工具的微型 wheel，安装到容器用户目录。"""
    path = workspace / "iris_snapshot_probe-0.0.1-py3-none-any.whl"
    metadata = "iris_snapshot_probe-0.0.1.dist-info"
    entries = {
        "iris_snapshot_probe.py": "VALUE = 'saved dependency'\n",
        f"{metadata}/METADATA": (
            "Metadata-Version: 2.1\nName: iris-snapshot-probe\nVersion: 0.0.1\n"
        ),
        f"{metadata}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = f"{metadata}/RECORD"
    entries[record] = "".join(f"{name},,\n" for name in (*entries, record))
    with zipfile.ZipFile(path, "w") as wheel:
        for name, content in entries.items():
            wheel.writestr(name, content)
    return path


async def save_demo(workspace: Path, database: Path, directory: Path, image: str) -> None:
    """执行离线安装与文件写入，保留一个待回答 Run，再显式保存。"""
    workspace.mkdir(parents=True, exist_ok=True)
    wheel = _probe_wheel(workspace)
    provider = ScriptedProvider()
    provider.steps.extend(
        (
            tool(
                "install",
                "run_python",
                code=(
                    "import subprocess, sys\nfrom pathlib import Path\n"
                    "subprocess.run([sys.executable, '-m', 'pip', 'install', "
                    "'--user', '--no-index', "
                    f"'--no-deps', '--no-compile', '/workspace/{wheel.name}'], check=True)\n"
                    f"Path({wheel.name!r}).unlink()\n"
                    "Path('snapshot-value.txt').write_text('saved')"
                ),
            ),
            tool(
                "check-install",
                "run_python",
                code=("import iris_snapshot_probe\nprint(iris_snapshot_probe.VALUE)"),
            ),
            done("文件与依赖已保存"),
            tool("confirm", "ask_question", question="恢复后继续吗？"),
        )
    )
    runner = AgentRunner.from_config(
        build_demo_config(workspace, database, image), provider=provider
    )
    try:
        result = await runner.start(
            AgentRunRequest(input="安装测试依赖并写文件", session_id="files")
        )
        require_completed(runner, result)
        waiting = await runner.start(
            AgentRunRequest(input="等待回答", session_id="approval", run_id="environment-waiting")
        )
        if waiting.run.phase is not RunPhase.WAITING:
            raise IrisRunStateError("示例未进入人工等待")
        manifest = await runner.save_environment_snapshot(directory)
        print(
            json.dumps(
                {
                    "snapshot_id": str(manifest.snapshot_id),
                    "run_id": waiting.run.run_id,
                    "interaction_id": waiting.pending_interaction.interaction_id,
                },
                ensure_ascii=False,
            )
        )
    finally:
        await runner.aclose()


async def restore_demo(workspace: Path, database: Path, directory: Path, image: str) -> None:
    """从最新恢复点还原文件和依赖，再回答原 interaction。"""
    provider = ScriptedProvider()
    provider.steps.extend(
        (
            tool(
                "verify",
                "run_python",
                code=(
                    "from pathlib import Path\nimport iris_snapshot_probe\n"
                    "print(Path('snapshot-value.txt').read_text())\nprint(iris_snapshot_probe.VALUE)"
                ),
            ),
            done("恢复完成"),
        )
    )
    runner = await AgentRunner.from_environment_snapshot(
        build_demo_config(workspace, database, image),
        directory,
        provider=provider,
    )
    try:
        waiting = runner.get_result("environment-waiting")
        if waiting is None or waiting.pending_interaction is None:
            raise IrisRunStateError("恢复点没有示例等待事项")
        result = await runner.resume(
            waiting.run.run_id,
            interaction_id=waiting.pending_interaction.interaction_id,
            response=QuestionInteractionResponse(answer="继续"),
        )
        require_completed(runner, result)
        print(
            json.dumps(
                {
                    "run_id": result.run.run_id,
                    "stop_reason": result.run.stop_reason,
                    "file": (workspace / "snapshot-value.txt").read_text(),
                },
                ensure_ascii=False,
            )
        )
    finally:
        await runner.aclose()


async def main() -> None:
    """显式选择 save 或 restore；目标腾空由运行示例的宿主完成。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("save", "restore"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--snapshot-directory", type=Path, required=True)
    parser.add_argument("--image", default="iris-command:local")
    parser.add_argument("--run-docker", action="store_true")
    args = parser.parse_args()
    if not args.run_docker:
        parser.error("真实 Docker 示例需要显式 --run-docker")
    operation = save_demo if args.operation == "save" else restore_demo
    await operation(
        args.workspace.resolve(),
        args.database.resolve(),
        args.snapshot_directory.resolve(),
        args.image,
    )


if __name__ == "__main__":
    asyncio.run(main())
