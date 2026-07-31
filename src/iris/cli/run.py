"""Config-first、一次性的 Agent lifecycle CLI 编排。"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..agents import AgentConfig
from ..exceptions import IrisRunStateError
from ..harness import (
    AgentRunner,
    AgentRunOptions,
    AgentRunRequest,
    RunLimits,
    RuntimeExecutionOptions,
)
from ..hitl import PermissionInteractionResponse, QuestionInteractionResponse
from ..message import LLMRequest, LLMResponse
from ..providers import create_provider_client
from ..runtime import RuntimeProvider
from ._config import load_cli_agent
from .run_output import (
    RunCommandName,
    RunCommandRenderer,
    project_events_exception_output,
    project_events_interrupted_output,
    project_events_output,
    project_exception_output,
    project_interrupted_output,
    project_run_output,
)

Decision = Literal["approve", "reject"]


@dataclass(frozen=True, slots=True)
class RunStartOptions:
    """``iris run start`` 的已解析选项。"""

    config_path: Path
    input: str
    session_id: str = "cli"
    run_id: str | None = None
    max_steps: int = 20
    include_tools: bool = True
    env_file: Path | None = None
    json_output: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "input", _required_text(self.input, field_name="input"))
        object.__setattr__(
            self,
            "session_id",
            _required_text(self.session_id, field_name="session_id"),
        )
        object.__setattr__(
            self,
            "run_id",
            _optional_text(self.run_id, field_name="run_id"),
        )
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")


@dataclass(frozen=True, slots=True)
class RunStatusOptions:
    """``iris run status`` 的已解析选项。"""

    config_path: Path
    run_id: str
    env_file: Path | None = None
    json_output: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            _required_text(self.run_id, field_name="run_id"),
        )


@dataclass(frozen=True, slots=True)
class RunEventsOptions:
    """``iris run events`` 的已解析选项。"""

    config_path: Path
    run_id: str
    after_sequence: int = 0
    env_file: Path | None = None
    json_output: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            _required_text(self.run_id, field_name="run_id"),
        )
        if self.after_sequence < 0:
            raise ValueError("after_sequence 必须大于等于 0")


@dataclass(frozen=True, slots=True)
class RunResumeOptions:
    """``iris run resume`` 的已解析选项。"""

    config_path: Path
    run_id: str
    interaction_id: str
    decision: Decision | None = None
    answer: str | None = None
    env_file: Path | None = None
    json_output: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            _required_text(self.run_id, field_name="run_id"),
        )
        object.__setattr__(
            self,
            "interaction_id",
            _required_text(self.interaction_id, field_name="interaction_id"),
        )
        if (self.decision is None) == (self.answer is None):
            raise ValueError("decision 和 answer 必须且只能提供一个")
        if self.decision is not None and self.decision not in {"approve", "reject"}:
            raise ValueError("decision 必须是 approve 或 reject")
        object.__setattr__(
            self,
            "answer",
            _optional_text(self.answer, field_name="answer"),
        )


@dataclass(frozen=True, slots=True)
class RunCancelOptions:
    """``iris run cancel`` 的已解析选项。"""

    config_path: Path
    run_id: str
    reason: str | None = None
    settlement_timeout: float | None = None
    env_file: Path | None = None
    json_output: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            _required_text(self.run_id, field_name="run_id"),
        )
        object.__setattr__(
            self,
            "reason",
            _optional_text(self.reason, field_name="reason"),
        )
        if self.settlement_timeout is not None and (
            not math.isfinite(self.settlement_timeout) or self.settlement_timeout <= 0
        ):
            raise ValueError("settlement_timeout 必须是正有限数")


@dataclass(frozen=True, slots=True)
class RunRecoverOptions:
    """``iris run recover`` 的已解析选项。"""

    config_path: Path
    run_id: str
    activation_id: str
    env_file: Path | None = None
    json_output: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            _required_text(self.run_id, field_name="run_id"),
        )
        object.__setattr__(
            self,
            "activation_id",
            _required_text(self.activation_id, field_name="activation_id"),
        )


class _NonExecutingProvider:
    """为 provider-free lifecycle 命令提供不可执行的 runtime 依赖。"""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        raise IrisRunStateError("当前 CLI 命令不能调用 provider")


def run_start(
    options: RunStartOptions,
    *,
    renderer: RunCommandRenderer | None = None,
) -> int:
    """创建 run 并推进一次 start activation。"""
    command: RunCommandName = "start"
    output = renderer or _default_renderer(options)
    try:
        runner = _build_real_runner(options)
        result = asyncio.run(
            runner.start(
                AgentRunRequest(
                    input=options.input,
                    session_id=options.session_id,
                    run_id=options.run_id,
                ),
                options=AgentRunOptions(
                    limits=RunLimits(max_model_steps=options.max_steps),
                    runtime=RuntimeExecutionOptions(include_tools=options.include_tools),
                ),
            )
        )
    except KeyboardInterrupt:
        output.render(project_interrupted_output(command))
        return 130
    except Exception as exc:
        return output.render(project_exception_output(command, exc))
    return output.render(project_run_output(command, result))


def run_status(
    options: RunStatusOptions,
    *,
    renderer: RunCommandRenderer | None = None,
) -> int:
    """读取 run snapshot，并在可用时读取 durable result。"""
    command: RunCommandName = "status"
    output = renderer or _default_renderer(options)
    try:
        runner = _build_non_executing_runner(options)
        snapshot = runner.get_run(options.run_id)
        result = runner.get_result(options.run_id)
    except KeyboardInterrupt:
        output.render(project_interrupted_output(command))
        return 130
    except Exception as exc:
        return output.render(project_exception_output(command, exc))
    return output.render(project_run_output(command, result or snapshot))


def run_events(
    options: RunEventsOptions,
    *,
    renderer: RunCommandRenderer | None = None,
) -> int:
    """从 exclusive sequence cursor 读取一次 durable event timeline。"""
    output = renderer or _default_renderer(options)
    try:
        runner = _build_non_executing_runner(options)
        events = runner.list_events(
            options.run_id,
            after_sequence=options.after_sequence,
        )
    except KeyboardInterrupt:
        output.render(
            project_events_interrupted_output(
                run_id=options.run_id,
                after_sequence=options.after_sequence,
            )
        )
        return 130
    except Exception as exc:
        return output.render(
            project_events_exception_output(
                run_id=options.run_id,
                after_sequence=options.after_sequence,
                error=exc,
            )
        )
    return output.render(
        project_events_output(
            run_id=options.run_id,
            after_sequence=options.after_sequence,
            events=events,
        )
    )


def run_resume(
    options: RunResumeOptions,
    *,
    renderer: RunCommandRenderer | None = None,
) -> int:
    """以显式 interaction identity 和 typed response 恢复 waiting run。"""
    command: RunCommandName = "resume"
    output = renderer or _default_renderer(options)
    response = (
        PermissionInteractionResponse(decision=options.decision)
        if options.decision is not None
        else QuestionInteractionResponse(answer=options.answer or "")
    )
    try:
        runner = _build_real_runner(options)
        result = asyncio.run(
            runner.resume(
                options.run_id,
                interaction_id=options.interaction_id,
                response=response,
            )
        )
    except KeyboardInterrupt:
        output.render(project_interrupted_output(command))
        return 130
    except Exception as exc:
        return output.render(project_exception_output(command, exc))
    return output.render(project_run_output(command, result))


def run_cancel(
    options: RunCancelOptions,
    *,
    renderer: RunCommandRenderer | None = None,
) -> int:
    """请求取消并等待 durable terminal settlement。"""
    command: RunCommandName = "cancel"
    output = renderer or _default_renderer(options)
    try:
        runner = _build_non_executing_runner(options)
        result = asyncio.run(
            runner.cancel(
                options.run_id,
                reason=options.reason,
                settlement_timeout=options.settlement_timeout,
            )
        )
    except KeyboardInterrupt:
        output.render(project_interrupted_output(command))
        return 130
    except Exception as exc:
        return output.render(project_exception_output(command, exc))
    return output.render(project_run_output(command, result))


def run_recover(
    options: RunRecoverOptions,
    *,
    renderer: RunCommandRenderer | None = None,
) -> int:
    """使用精确 activation fence 显式恢复 active run。"""
    command: RunCommandName = "recover"
    output = renderer or _default_renderer(options)
    try:
        runner = _build_real_runner(options)
        result = asyncio.run(
            runner.recover(
                options.run_id,
                expected_activation_id=options.activation_id,
            )
        )
    except KeyboardInterrupt:
        output.render(project_interrupted_output(command))
        return 130
    except Exception as exc:
        return output.render(project_exception_output(command, exc))
    return output.render(project_run_output(command, result))


def _build_real_runner(
    options: RunStartOptions | RunResumeOptions | RunRecoverOptions,
) -> AgentRunner:
    config = load_cli_agent(options.config_path, env_file=options.env_file)
    provider = create_provider_client(
        config.to_model_route(),
        base_url=config.model.base_url,
        timeout=config.model.timeout,
    )
    return _runner_from_config(config, options.config_path, provider=provider)


def _build_non_executing_runner(
    options: RunStatusOptions | RunEventsOptions | RunCancelOptions,
) -> AgentRunner:
    config = load_cli_agent(options.config_path, env_file=options.env_file)
    return _runner_from_config(
        config,
        options.config_path,
        provider=_NonExecutingProvider(),
    )


def _runner_from_config(
    config: AgentConfig,
    config_path: Path,
    *,
    provider: RuntimeProvider,
) -> AgentRunner:
    return AgentRunner.from_config(
        config,
        config_path=config_path,
        provider=provider,
    )


def _default_renderer(
    options: RunStartOptions
    | RunStatusOptions
    | RunEventsOptions
    | RunResumeOptions
    | RunCancelOptions
    | RunRecoverOptions,
) -> RunCommandRenderer:
    return RunCommandRenderer(
        json_output=options.json_output,
        config_path=options.config_path,
        env_file=options.env_file,
    )


def _required_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


def _optional_text(value: str | None, *, field_name: str) -> str | None:
    return None if value is None else _required_text(value, field_name=field_name)


__all__ = [
    "RunCancelOptions",
    "RunEventsOptions",
    "RunRecoverOptions",
    "RunResumeOptions",
    "RunStartOptions",
    "RunStatusOptions",
    "run_cancel",
    "run_events",
    "run_recover",
    "run_resume",
    "run_start",
    "run_status",
]
