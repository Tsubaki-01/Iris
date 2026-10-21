"""真实 Docker、Harbor Trial、Iris SDK 与独立 verifier 的闭环验收。"""

# ruff: noqa: E402
from __future__ import annotations

import asyncio
import json
import os
import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("harbor")

from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).parent / "tasks" / "file_task"


def tiny_png() -> bytes:
    """用标准 PNG 编码生成测试像素，不依赖宿主 Pillow。"""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


@pytest.fixture(scope="session")
def iris_wheel(harbor_docker: None, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """构建当前工作树 wheel，测试每个 trial 安装同一份明确产物。"""
    target = tmp_path_factory.mktemp("iris-wheel")
    env = {**os.environ, "UV_CACHE_DIR": str(ROOT / "tmp" / "uv-cache")}
    subprocess.run(["uv", "build", str(ROOT), "--out-dir", str(target)], check=True, env=env)
    return next(target.glob("iris-*.whl"))


async def run_trial(
    tmp_path: Path,
    wheel: Path,
    mode: str,
    *,
    seed: int = 10,
) -> tuple[TrialResult, Path]:
    """通过 Harbor 自身的 Trial 生命周期创建、执行、验收并回收容器。"""
    task_path = tmp_path / f"task-{mode}-{seed}"
    shutil.copytree(FIXTURE, task_path, ignore=shutil.ignore_patterns("__pycache__"))
    environment = task_path / "environment"
    (environment / "scenario.json").write_text(json.dumps({"mode": mode}), encoding="utf-8")
    (environment / "input.csv").write_text(f"value\n{seed}\n{seed + 1}\n", encoding="utf-8")
    (environment / "pixel.png").write_bytes(tiny_png())
    (task_path / "tests/test.sh").write_text(
        "python /tests/verify.py\n", encoding="utf-8", newline="\n"
    )
    trial_name = f"iris-{mode}-{uuid4().hex[:8]}"
    config = TrialConfig(
        task=TaskConfig(path=task_path),
        trial_name=trial_name,
        trials_dir=tmp_path / "trials",
        agent=AgentConfig(
            import_path="evaluation.harbor.agent:IrisHarborAgent",
            model_name="openai/fixture",
            override_timeout_sec=8 if mode == "timeout" else 90,
            override_setup_timeout_sec=600,
            kwargs={"iris_wheel": str(wheel), "iris_revision": "integration-worktree"},
            env={
                "IRIS_PROVIDER_API_KEYS__OPENAI": "fixture-key",
                "IRIS_PROVIDERS__OPENAI__BASE_URL": "http://127.0.0.1:8765/v1",
            },
        ),
        environment=EnvironmentConfig(type="docker", delete=True),
    )
    trial = await Trial.create(config)
    result = await trial.run()
    # Harbor 用 session_id 作为 compose project，确认回收了本 trial 的容器。
    containers = subprocess.run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label=com.docker.compose.project={trial.agent_environment.session_id}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not containers.stdout.strip()
    return result, config.trials_dir / trial_name


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,reward", [("correct", 1), ("wrong", 0), ("partial", 1)])
async def test_file_task_reward_and_offline_artifacts(
    harbor_docker: None,
    iris_wheel: Path,
    tmp_path: Path,
    mode: str,
    reward: int,
) -> None:
    result, folder = await run_trial(tmp_path, iris_wheel, mode)
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards["reward"] == reward
    record = json.loads((folder / "agent/iris/result.json").read_text(encoding="utf-8"))
    assert record["cleanup_completed"]
    assert record["trace_status"] == ("partial" if mode == "partial" else "complete")
    assert result.agent_result.n_input_tokens > 0
    bundle = folder / "agent/iris"
    trace = json.loads((bundle / "trace.json").read_text(encoding="utf-8"))
    assert trace["configuration"]["workspace"] == "/task-project"
    copied = [bundle / item["path"] for item in trace["assets"] if item["path"]]
    assert any(
        path.suffix == ".png" and path.read_bytes().startswith(b"\x89PNG") for path in copied
    )
    if mode != "partial":
        assert any(path.suffix == ".txt" and "archive-text-" in path.read_text() for path in copied)
    else:
        assert any(asset["error"] is not None for asset in trace["assets"])


@pytest.mark.asyncio
async def test_parallel_trials_are_isolated(
    harbor_docker: None, iris_wheel: Path, tmp_path: Path
) -> None:
    outcomes = await asyncio.gather(
        run_trial(tmp_path, iris_wheel, "correct", seed=20),
        run_trial(tmp_path, iris_wheel, "correct", seed=100),
    )
    observed = []
    sessions = []
    for result, folder in outcomes:
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards["reward"] == 1
        observed.append(json.loads((folder / "verifier/observed.json").read_text())["actual"])
        sessions.append(json.loads((folder / "agent/iris/result.json").read_text())["session_id"])
    assert sorted(observed) == [41, 201]
    assert len(set(sessions)) == 2


@pytest.mark.asyncio
async def test_timeout_stops_writes_before_verification(
    harbor_docker: None,
    iris_wheel: Path,
    tmp_path: Path,
) -> None:
    result, folder = await run_trial(tmp_path, iris_wheel, "timeout")
    assert result.exception_info.exception_type == "AgentTimeoutError"
    assert result.verifier_result.rewards["reward"] == 1
    record = json.loads((folder / "agent/iris/result.json").read_text())
    assert record["cleanup_completed"]
    assert record["stop_reason"] == "cancelled"


@pytest.mark.asyncio
async def test_cleanup_failure_skips_verifier(
    harbor_docker: None,
    iris_wheel: Path,
    tmp_path: Path,
) -> None:
    result, folder = await run_trial(tmp_path, iris_wheel, "cleanup_failure")
    assert result.exception_info.exception_type == "RuntimeError"
    assert result.verifier_result is None
    assert not (folder / "verifier/started.txt").exists()
    record = json.loads((folder / "agent/iris/result.json").read_text())
    assert not record["cleanup_completed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,exception",
    [
        ("provider_failure", "NonZeroAgentExitCodeError"),
        ("missing_reward", "RewardFileNotFoundError"),
    ],
)
async def test_error_paths_keep_provider_and_missing_reward_distinct(
    harbor_docker: None,
    iris_wheel: Path,
    tmp_path: Path,
    mode: str,
    exception: str,
) -> None:
    result, folder = await run_trial(tmp_path, iris_wheel, mode)
    assert result.exception_info.exception_type == exception
    record = json.loads((folder / "agent/iris/result.json").read_text(encoding="utf-8"))
    assert record["cleanup_completed"]
    assert (folder / "verifier/started.txt").is_file()
    if mode == "provider_failure":
        assert record["stop_reason"] == "failed"
        assert record["result"]["error"]["source"] == "provider"
        assert result.verifier_result.rewards["reward"] == 0
    else:
        assert record["stop_reason"] == "completed"
        assert result.verifier_result is None
