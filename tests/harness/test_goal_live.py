"""显式开启的 DeepSeek Goal 两轮真实模型、文件工具与 SQLite 验证。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from iris.config import init_config, reset
from iris.harness import AgentRunner, SessionManager
from iris.lifecycle import AgentRunOptions, RunLimits
from iris.store import SQLiteStore

from .test_goal_integration import (
    SESSION,
    assert_file_goal_result,
    collect_completed_goal,
    file_goal_config,
)

pytestmark = [pytest.mark.live_deepseek, pytest.mark.asyncio]

_SYSTEM = """按目标与验收要求操作真实文件。本次验收严格要求两个独立 Run。
当前轮号只取 iris.goal 的“已启动自动 Run”或 get_goal 的 rounds_started；调用工具不会增加轮号。
report_goal 只提交报告，不会结束或切换 Run。任何 report_goal 调用返回后，你的下一次响应必须
只有简短普通文本，不含任何工具调用。宿主收到普通文本后才会结束本 Run 并自动启动下一轮。
第一轮只准备 numbers.json；第二轮才创建 result.json。严禁在同一 Run 内自行切到下一轮。
"""

_OBJECTIVE = """这是验证 Goal 自动续跑的实际文件任务，必须严格分成两个自动 Run 完成。
以当前 Goal 上下文中的 rounds_started 判断本轮编号。你不能在第一轮提前完成第二轮。

第一轮（rounds_started=1）：
1. 使用 write_file 创建 numbers.json，完整内容必须是 JSON 数组 [7, 11, 13]。
2. 使用 read_file 读取 numbers.json，依据工具实际返回核对数值。
3. 单独调用 report_goal，使用当前 goal_id/revision，decision=continue，说明数据已准备好。
4. 返回一条简短普通文本结束本 Run。这一轮禁止创建 result.json，禁止报告 complete。

第二轮（rounds_started=2）：
1. 使用 read_file 重新读取 numbers.json，计算三个数的和。
2. 使用 write_file 创建 result.json，完整内容必须是 JSON 对象 {"sum": 31}。
3. 使用 read_file 读取 result.json，依据实际返回核对结果。
4. 确认两份文件都满足要求后，单独调用 report_goal，使用当前 goal_id/revision，
   decision=complete，reason 说明读回文件已核对。最后返回简短普通文本结束本 Run。

工具实际名称是 write_file/read_file/get_goal/report_goal。文件工具的路径参数名是 file_path，
请使用 numbers.json/result.json 这两个相对路径；write_file 的 content 参数传完整 JSON 文本。
每次写入后先等待工具结果再读取，每次申报后不再调用工作工具。需要版本时可调用 get_goal。
两轮之间由宿主自动继续，不要请求用户再次输入，也不要伪造工具执行或只回答文件内容。
"""


@pytest.fixture
def live_goal_config(request: pytest.FixtureRequest) -> Iterator[None]:
    """显式开关之后才读取现有配置；只检查凭据存在，不输出凭据对象。"""
    if not request.config.getoption("--run-live-deepseek"):
        pytest.skip("使用 --run-live-deepseek 显式开启真实 Goal API 调用")
    reset()
    try:
        config = init_config(env_file=".env.local")
        if not (config.provider_api_keys.get("deepseek") or config.api_key):
            pytest.fail(".env.local 中缺少 DeepSeek API 凭据")
        yield
    finally:
        reset()


@pytest.mark.usefixtures("live_goal_config")
async def test_real_deepseek_two_round_goal_commits_files_and_report(tmp_path: Path) -> None:
    """不注入 provider/stub，不手动 submit 第二轮，验证真实自动推进闭环。"""
    config = file_goal_config(tmp_path)
    config = config.model_copy(
        update={
            "system": _SYSTEM,
            "model": config.model.model_copy(
                update={
                    "temperature": 0,
                    "max_tokens": 2048,
                    "timeout": 60,
                }
            ),
        }
    )
    store = SQLiteStore(tmp_path / "goal-live.db")
    runner = AgentRunner.from_config(config, store=store)
    manager = SessionManager(runner, SESSION)
    try:
        created = await manager.goal.create(
            _OBJECTIVE,
            max_rounds=2,
            run_options=AgentRunOptions(limits=RunLimits(max_model_steps=14)),
        )
        observed = await collect_completed_goal(manager, timeout=360)
        goal, run_ids = assert_file_goal_result(
            tmp_path, store, created.view.goal.goal_id, require_tokens=True
        )
        evidence = {
            "goal": goal.model_dump(mode="json"),
            "runs": [store.load_result(run_id).model_dump(mode="json") for run_id in run_ids],
            "tools": {
                run_id: [call.model_dump(mode="json") for call in store.list_tool_calls(run_id)]
                for run_id in run_ids
            },
            "host_event_count": len(observed),
        }
        (tmp_path / "goal-evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"Goal evidence={tmp_path}, rounds={goal.rounds_started}, "
            f"tokens={sum(store.load_result(run_id).run.usage.total_tokens for run_id in run_ids)}"
        )
    finally:
        await manager.close(cancel_run=True)
        await runner.aclose()
