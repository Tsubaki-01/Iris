"""一次性 lifecycle CLI 的稳定输出投影与渲染。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..exceptions import IrisError
from ..hitl import HumanInteraction, PermissionPrompt, QuestionPrompt
from ..lifecycle import (
    RunErrorInfo,
    RunEvent,
    RunPhase,
    RunResult,
    RunSnapshot,
    RunStopReason,
)
from ..message import Msg

RunCommandName = Literal["start", "status", "resume", "cancel", "recover"]
RunEventsCommandName = Literal["events"]


class RunCommandError(BaseModel):
    """CLI 对外稳定的结构化错误。"""

    code: str
    source: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunCommandOutput(BaseModel):
    """一次 lifecycle CLI 调用的稳定输出。"""

    ok: bool
    command: RunCommandName
    run: RunSnapshot | None = None
    assistant_message: Msg | None = None
    pending_interaction: HumanInteraction | None = None
    error: RunCommandError | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunEventsOutput(BaseModel):
    """一次 durable event timeline 读取的稳定输出。"""

    ok: bool
    command: RunEventsCommandName = "events"
    run_id: str
    after_sequence: int
    next_after_sequence: int
    events: list[RunEvent]
    error: RunCommandError | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


def project_run_output(
    command: RunCommandName,
    subject: RunSnapshot | RunResult,
) -> RunCommandOutput:
    """把 durable snapshot/result 投影为 CLI 输出。"""
    if isinstance(subject, RunResult):
        error = _project_run_error(subject.error)
        failed = subject.run.stop_reason in {
            RunStopReason.FAILED,
            RunStopReason.OUTCOME_UNKNOWN,
        }
        return RunCommandOutput(
            ok=not failed,
            command=command,
            run=subject.run,
            assistant_message=subject.assistant_message,
            pending_interaction=subject.pending_interaction,
            error=error,
        )
    return RunCommandOutput(ok=True, command=command, run=subject)


def project_exception_output(
    command: RunCommandName,
    error: Exception,
) -> RunCommandOutput:
    """把 CLI 边界捕获的异常归一化为稳定输出。"""
    return RunCommandOutput(
        ok=False,
        command=command,
        error=_normalize_exception(error),
    )


def project_interrupted_output(command: RunCommandName) -> RunCommandOutput:
    """构造当前 CLI 进程被用户中断的输出。"""
    return RunCommandOutput(
        ok=False,
        command=command,
        error=RunCommandError(
            code="INTERRUPTED",
            source="runtime",
            message="当前 CLI 进程已被用户中断",
        ),
    )


def project_events_output(
    *,
    run_id: str,
    after_sequence: int,
    events: list[RunEvent],
) -> RunEventsOutput:
    """把 durable events 投影为带增量 cursor 的稳定输出。"""
    next_after_sequence = events[-1].sequence if events else after_sequence
    return RunEventsOutput(
        ok=True,
        run_id=run_id,
        after_sequence=after_sequence,
        next_after_sequence=next_after_sequence,
        events=events,
    )


def project_events_exception_output(
    *,
    run_id: str,
    after_sequence: int,
    error: Exception,
) -> RunEventsOutput:
    """把 event 读取异常归一化为稳定 event 输出。"""
    return RunEventsOutput(
        ok=False,
        run_id=run_id,
        after_sequence=after_sequence,
        next_after_sequence=after_sequence,
        events=[],
        error=_normalize_exception(error),
    )


def project_events_interrupted_output(
    *,
    run_id: str,
    after_sequence: int,
) -> RunEventsOutput:
    """构造 event 读取被用户中断的稳定输出。"""
    return RunEventsOutput(
        ok=False,
        run_id=run_id,
        after_sequence=after_sequence,
        next_after_sequence=after_sequence,
        events=[],
        error=RunCommandError(
            code="INTERRUPTED",
            source="runtime",
            message="当前 CLI 进程已被用户中断",
        ),
    )


def classify_exit_code(output: RunCommandOutput | RunEventsOutput) -> int:
    """根据稳定输出分类普通退出码。"""
    return 0 if output.ok else 1


class RunCommandRenderer:
    """渲染一次性 lifecycle 命令的人类摘要或紧凑 JSON。"""

    def __init__(
        self,
        *,
        json_output: bool = False,
        config_path: Path | None = None,
        env_file: Path | None = None,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        """初始化输出模式和用于下一步命令的配置路径。"""
        self.json_output = json_output
        self.config_path = config_path
        self.env_file = env_file
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.console = Console(file=self.stdout)
        self.error_console = Console(file=self.stderr)

    def render(self, output: RunCommandOutput | RunEventsOutput) -> int:
        """恰好渲染一次输出并返回对应退出码。"""
        if self.json_output:
            return self._render_json(output)
        self._render_human(output)
        return classify_exit_code(output)

    def _render_json(self, output: RunCommandOutput | RunEventsOutput) -> int:
        try:
            payload = json.dumps(
                output.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except Exception as exc:
            fallback = (
                project_events_exception_output(
                    run_id=output.run_id,
                    after_sequence=output.after_sequence,
                    error=exc,
                )
                if isinstance(output, RunEventsOutput)
                else project_exception_output(output.command, exc)
            )
            payload = json.dumps(
                fallback.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            print(payload, file=self.stderr)
            return 1

        stream = (
            self.stderr
            if output.error is not None
            and (isinstance(output, RunEventsOutput) or output.run is None)
            else self.stdout
        )
        print(payload, file=stream)
        return classify_exit_code(output)

    def _render_human(self, output: RunCommandOutput | RunEventsOutput) -> None:
        if isinstance(output, RunEventsOutput):
            self._render_events_human(output)
            return
        body = [f"command: {output.command}"]
        run = output.run
        if run is not None:
            body.extend(
                [
                    f"run_id: {run.run_id}",
                    f"session_id: {run.session_id}",
                    f"phase: {_enum_value(run.phase)}",
                    f"revision: {run.revision}",
                    f"activation_id: {run.current_activation_id or '-'}",
                    f"interaction_id: {run.pending_interaction_id or '-'}",
                ]
            )
            self._append_interaction(body, output)
            self._append_terminal(body, output)
            self._append_active_next_step(body, output)
        elif output.error is not None and output.error.code == "INTERRUPTED":
            body.append(output.error.message)

        if output.error is not None and output.error.code != "INTERRUPTED":
            body.extend(
                [
                    f"error: {output.error.source}:{output.error.code}",
                    f"message: {output.error.message}",
                ]
            )
            if output.error.details:
                body.append(
                    "details: "
                    + json.dumps(output.error.details, ensure_ascii=False, default=str)
                )

        panel = Panel(
            Text("\n".join(body)),
            title="Iris Run" if output.ok else "Iris Run Error",
            border_style="cyan" if output.ok else "red",
        )
        console = self.error_console if output.error is not None and run is None else self.console
        console.print(panel)

    def _render_events_human(self, output: RunEventsOutput) -> None:
        body = [
            f"command: {output.command}",
            f"run_id: {output.run_id}",
            f"after_sequence: {output.after_sequence}",
            f"next_after_sequence: {output.next_after_sequence}",
            f"event_count: {len(output.events)}",
        ]
        if output.events:
            body.append("events:")
            for event in output.events:
                body.extend(
                    [
                        f"  sequence: {event.sequence}",
                        f"  kind: {_enum_value(event.kind)}",
                        f"  occurred_at: {_iso_utc(event.occurred_at)}",
                        f"  session_id: {event.session_id}",
                    ]
                )
                if event.activation_id is not None:
                    body.append(f"  activation_id: {event.activation_id}")
                if event.step_index is not None:
                    body.append(f"  step_index: {event.step_index}")
                if event.correlation_id is not None:
                    body.append(f"  correlation_id: {event.correlation_id}")
                if event.payload:
                    body.append(
                        "  payload: "
                        + json.dumps(
                            event.payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    )
        else:
            body.append("events: 无")

        if output.error is not None:
            body.extend(
                [
                    f"error: {output.error.source}:{output.error.code}",
                    f"message: {output.error.message}",
                ]
            )
            if output.error.details:
                body.append(
                    "details: "
                    + json.dumps(output.error.details, ensure_ascii=False, default=str)
                )

        panel = Panel(
            Text("\n".join(body)),
            title="Iris Run Events" if output.ok else "Iris Run Events Error",
            border_style="cyan" if output.ok else "red",
        )
        console = self.error_console if output.error is not None else self.console
        console.print(panel)

    def _append_interaction(
        self,
        body: list[str],
        output: RunCommandOutput,
    ) -> None:
        interaction = output.pending_interaction
        if interaction is None:
            return
        prompt = interaction.request.prompt
        body.extend(
            [
                f"interaction_kind: {_enum_value(prompt.kind)}",
                f"tool_name: {interaction.request.tool_call.tool_name}",
            ]
        )
        if isinstance(prompt, PermissionPrompt):
            body.append(f"prompt: {prompt.reason}")
            response_arg = "--decision approve"
        elif isinstance(prompt, QuestionPrompt):
            body.append(f"prompt: {prompt.question}")
            if prompt.options:
                body.append(f"options: {', '.join(prompt.options)}")
            response_arg = '--answer "TEXT"'
        else:  # pragma: no cover - Pydantic discriminator 保证不可达
            return
        body.append(
            "next: "
            f"iris run resume {_quoted(self.config_path)} "
            f'--run-id "{interaction.run_id}" '
            f'--interaction-id "{interaction.interaction_id}" {response_arg}'
            f"{self._env_file_argument()}"
        )

    def _append_terminal(
        self,
        body: list[str],
        output: RunCommandOutput,
    ) -> None:
        run = output.run
        if run is None or run.phase is not RunPhase.TERMINAL:
            return
        body.append(f"stop_reason: {_enum_value(run.stop_reason)}")
        if output.assistant_message is not None:
            body.append(f"assistant: {output.assistant_message.text}")

    def _append_active_next_step(
        self,
        body: list[str],
        output: RunCommandOutput,
    ) -> None:
        run = output.run
        if (
            output.command != "status"
            or run is None
            or run.phase is not RunPhase.ACTIVE
            or run.current_activation_id is None
        ):
            return
        body.append(
            "next: "
            f"iris run recover {_quoted(self.config_path)} "
            f'--run-id "{run.run_id}" '
            f'--activation-id "{run.current_activation_id}"'
            f"{self._env_file_argument()}"
        )

    def _env_file_argument(self) -> str:
        if self.env_file is None:
            return ""
        return f" --env-file {_quoted(self.env_file)}"


def _project_run_error(error: RunErrorInfo | None) -> RunCommandError | None:
    if error is None:
        return None
    return RunCommandError(
        code=error.code,
        source=_enum_value(error.source),
        message=error.message,
        details=error.details,
    )


def _normalize_exception(error: Exception) -> RunCommandError:
    if isinstance(error, IrisError):
        return RunCommandError(
            code=error.runtime_code,
            source=error.runtime_source,
            message=error.message,
            details=error.context,
        )
    return RunCommandError(
        code="CLI_ERROR",
        source="runtime",
        message=str(error) or error.__class__.__name__,
        details={"exception_type": error.__class__.__name__},
    )


def _iso_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _enum_value(value: Enum | str | None) -> str:
    if value is None:
        return "-"
    return str(value.value) if isinstance(value, Enum) else str(value)


def _quoted(path: Path | None) -> str:
    value = "AGENT_CONFIG" if path is None else str(path)
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


__all__ = [
    "RunCommandError",
    "RunCommandName",
    "RunCommandOutput",
    "RunCommandRenderer",
    "RunEventsCommandName",
    "RunEventsOutput",
    "classify_exit_code",
    "project_events_exception_output",
    "project_events_interrupted_output",
    "project_events_output",
    "project_exception_output",
    "project_interrupted_output",
    "project_run_output",
]
