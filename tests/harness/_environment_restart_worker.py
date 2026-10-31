"""独立解释器中的恢复验收入口，provider 始终固定。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from iris.agents import load_agent_config
from iris.command.models import CommandRequest, CommandScope, PythonCode
from iris.harness import AgentRunner
from iris.hitl import QuestionInteractionResponse
from iris.lifecycle import RunPhase

from .fakes import StaticProvider, text_response


async def main() -> None:
    """记录恢复后的事实，再继续原 HITL，真实引擎场景另验证安装依赖。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("run_id")
    parser.add_argument("--fake-docker", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.fake_docker:
        from ..command.test_docker import FakeClient, FakeDockerError

        client = FakeClient()
        sys.modules["aiodocker"] = SimpleNamespace(Docker=lambda **kwargs: client)
        sys.modules["aiodocker.exceptions"] = SimpleNamespace(DockerError=FakeDockerError)
    provider = StaticProvider(text_response("restarted"))
    runner = await AgentRunner.from_environment_snapshot(
        load_agent_config(args.config),
        args.directory,
        provider=provider,
    )
    try:
        assert not provider.requests
        waiting = runner.get_result(args.run_id)
        assert waiting is not None and waiting.run.phase is RunPhase.WAITING
        assert waiting.pending_interaction is not None
        report = {
            "source_id": runner.store.source_id,
            "interaction_id": waiting.pending_interaction.interaction_id,
            "steps_before": waiting.run.usage.model_steps_committed,
            "file": (runner.runtime.environment.workspace_root / "snapshot-value.txt").read_text(),
            "sessions": [item.session_id for item in runner.list_sessions().items],
        }
        service = runner.runtime.environment.command_binding.service
        if args.probe:
            probe = await service.execute(
                CommandScope("probe", "files"),
                CommandRequest(
                    "probe",
                    PythonCode(
                        "import json, os, iris_snapshot_probe\n"
                        "print(json.dumps({'value': iris_snapshot_probe.VALUE, "
                        "'path': iris_snapshot_probe.__file__, "
                        "'user': f'{os.getuid()}:{os.getgid()}', "
                        "'env': os.environ.get('SNAPSHOT_TEST')}))"
                    ),
                    runner.runtime.environment.workspace_root,
                    10,
                ),
            )
            assert probe.exit_code == 0, probe.stderr
            report["probe"] = json.loads(probe.stdout)
        result = await runner.resume(
            args.run_id,
            interaction_id=waiting.pending_interaction.interaction_id,
            response=QuestionInteractionResponse(answer="continue"),
        )
        assert result.run.phase is RunPhase.TERMINAL
        report["steps_after"] = result.run.usage.model_steps_committed
        if args.probe:
            original = runner._restored_manifest
            fourth = await runner.save_environment_snapshot(args.directory)
            fifth = await runner.save_environment_snapshot(args.directory)
            assert not (args.directory / str(fourth.snapshot_id)).exists()
            await service._sandbox._client.images.inspect(original.docker.image_ref)
            report["latest"] = str(fifth.snapshot_id)
            report["latest_image"] = fifth.docker.image_ref
            report["deferred_image"] = original.docker.image_ref
        print(json.dumps(report, ensure_ascii=False))
    finally:
        await runner.aclose()


if __name__ == "__main__":
    asyncio.run(main())
