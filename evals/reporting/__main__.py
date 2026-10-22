"""命令行入口：只读取已有 Harbor job，不发起模型或工具调用。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import summarize_job, write_report


def main() -> None:
    """读取固定清单，生成一个 job 的离线报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attempts", type=int, required=True)
    parser.add_argument("--task-id", action="append", help="只汇总清单中指定任务，可重复传入")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    task_ids = args.task_id or manifest["task_ids"]
    if args.attempts < 1:
        parser.error("--attempts 必须大于零")
    if not task_ids or len(task_ids) != len(set(task_ids)):
        parser.error("任务清单必须非空且 ID 不重复")
    if not set(task_ids) <= set(manifest["task_ids"]):
        parser.error("--task-id 必须属于 manifest 的 task_ids")
    report = summarize_job(args.job_dir, task_ids=task_ids, attempts=args.attempts)
    report["dataset"] = {key: manifest[key] for key in ("name", "dataset_id", "revision")}
    write_report(report, args.output)
    print(json.dumps(report["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
