"""独立读取文件副作用判分，不使用模型回答或 adapter 的退出码。"""

import csv
import json
import time
from pathlib import Path

root = Path("/task-project")
logs = Path("/logs/verifier")
logs.mkdir(parents=True, exist_ok=True)
(logs / "started.txt").write_text("verifier started")
scenario = json.loads(Path("/fixture/scenario.json").read_text())
if scenario["mode"] == "timeout":
    ticks = root / "ticks.txt"
    before = ticks.stat().st_size if ticks.exists() else 0
    time.sleep(0.5)
    after = ticks.stat().st_size if ticks.exists() else 0
    reward = int(before > 0 and before == after)
    (logs / "stopped.json").write_text(json.dumps({"before": before, "after": after}))
else:
    expected = sum(int(row["value"]) for row in csv.DictReader((root / "input.csv").open()))
    answer = root / "answer.json"
    actual = json.loads(answer.read_text())["total"] if answer.exists() else None
    reward = int(actual == expected)
    (logs / "observed.json").write_text(json.dumps({"expected": expected, "actual": actual}))
if scenario["mode"] != "missing_reward":
    (logs / "reward.txt").write_text(str(reward))
