"""查询已保留的执行树，导出不依赖评测框架的轨迹与引用产物。"""

from __future__ import annotations

import json
import shutil
from collections import deque
from collections.abc import Iterable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..lifecycle import RunPhase
from ..lifecycle.history import RunCursor
from ..message import ContentBlock, ImageBlock, ToolResultBlock

if TYPE_CHECKING:
    from .runner import AgentRunner


def _image_paths(blocks: Iterable[ContentBlock]) -> Iterator[Path]:
    """只遍历 typed 内容中的图片引用，不扫描正文路径。"""
    for block in blocks:
        if isinstance(block, ImageBlock):
            yield block.original.path
            yield block.model.path
        elif isinstance(block, ToolResultBlock):
            yield from _image_paths(block.content)


def collect_trajectory(runner: AgentRunner, session_id: str) -> dict[str, Any]:
    """读取根 session 的全部 run 及递归 child，不修改运行状态。

    Args:
        runner: 拥有该执行树 exact store 的 Runner，允许已经关闭。
        session_id: 本次样本独占的根 session 身份。

    Returns:
        dict[str, Any]: JSON 可序列化的原生记录；附件保留源路径，尚未复制。
            ACTIVE 运行标记 partial；每个 run 的事件序号仅在自身范围有序。
    """
    root_runs = []
    cursor: RunCursor | None = None
    while True:
        page = runner.list_runs(session_id, after=cursor)
        root_runs.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            break

    sessions: dict[str, Any] = {}
    runs: dict[str, Any] = {}
    assets: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []

    def add_asset(path: Path) -> None:
        source = str(path)
        assets.setdefault(source, {"source_path": source, "path": None, "error": None})

    def add_session(identifier: str) -> None:
        if identifier in sessions:
            return
        snapshot = runner.get_session(identifier)
        sessions[identifier] = snapshot.model_dump(mode="json")
        for message in snapshot.messages:
            for path in _image_paths(message.content):
                add_asset(path)

    add_session(session_id)
    pending = deque((run, None) for run in root_runs)
    while pending:
        run, parent = pending.popleft()
        add_session(run.session_id)
        result = runner.get_result(run.run_id)
        tool_calls = runner.list_tool_calls(run.run_id)
        runs[run.run_id] = {
            "run": run.model_dump(mode="json"),
            "parent": parent,
            "result": result.model_dump(mode="json") if result is not None else None,
            "tool_calls": [call.model_dump(mode="json") for call in tool_calls],
            "events": [event.model_dump(mode="json") for event in runner.list_events(run.run_id)],
        }
        if run.phase is RunPhase.ACTIVE:
            errors.append({"stage": "collect", "run_id": run.run_id, "message": "运行仍在推进"})
        for call in tool_calls:
            if call.result is None:
                continue
            for path in _image_paths(call.result.content):
                add_asset(path)
            artifact = call.result.artifact
            if artifact is not None:
                add_asset(artifact.path)
                if artifact.text_path is not None:
                    add_asset(artifact.text_path)
        cursor = None
        while True:
            children = runner.list_child_runs(run.run_id, after=cursor)
            for child in children.items:
                pending.append(
                    (
                        child.run,
                        {
                            "parent_run_id": child.parent_run_id,
                            "parent_tool_call_id": child.parent_tool_call_id,
                            "agent_selector": child.agent_selector,
                        },
                    )
                )
            cursor = children.next_cursor
            if cursor is None:
                break

    configuration = runner.describe_configuration()
    return {
        "schema_version": 1,
        "root_session_id": session_id,
        "root_run_ids": [run.run_id for run in root_runs],
        "configuration": {
            "snapshot_id": configuration.configuration_snapshot_id,
            "agent_name": configuration.agent_config.name,
            "model": configuration.agent_config.model.model_dump(
                mode="json", exclude={"metadata", "provider_options"}
            ),
            "workspace": str(configuration.workspace_root),
        },
        "sessions": sessions,
        "runs": runs,
        "assets": list(assets.values()),
        "export": {"status": "partial" if errors else "complete", "errors": errors},
    }


def write_trajectory(document: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """复制明确引用的文件并写出 trace.json，返回带相对索引的独立投影。

    Args:
        document: collect_trajectory 返回的可信文档，允许宿主补充 JSON 元数据。
        output_dir: 宿主为本次样本指定的归档目录。

    Returns:
        dict[str, Any]: 写出后的文档；附件复制失败为 partial，目录或 JSON
            文件写入失败为 failed。原文档与业务停止原因不变。
    """
    exported = deepcopy(document)
    destination = Path(output_dir)
    status = exported["export"]
    try:
        destination.mkdir(parents=True, exist_ok=True)
        if exported["assets"]:
            (destination / "assets").mkdir(exist_ok=True)
        for index, asset in enumerate(exported["assets"], start=1):
            source = Path(asset["source_path"])
            relative = Path("assets") / f"{index:06d}{source.suffix}"
            try:
                shutil.copyfile(source, destination / relative)
            except OSError as error:
                asset["error"] = str(error)
                status["status"] = "partial"
                status["errors"].append(
                    {
                        "stage": "asset",
                        "source_path": str(source),
                        "message": str(error),
                    }
                )
            else:
                asset["path"] = relative.as_posix()
        (destination / "trace.json").write_text(
            json.dumps(exported, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    except OSError as error:
        status["status"] = "failed"
        status["errors"].append({"stage": "write", "message": str(error)})
    return exported


__all__ = ["collect_trajectory", "write_trajectory"]
