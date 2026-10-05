"""Goal 动态上下文保留宿主贡献并绑定本轮目标。"""

from pathlib import Path

import pytest

from iris.context import ContextBuildScope, ContextContribution, ContextSnapshot
from iris.exceptions import IrisConfigError, IrisGoalError, IrisTemplateError
from iris.goal import GoalReason, GoalService
from iris.goal.context import GoalContextSource, render_continuation
from iris.goal.store import AdmitGoalRun
from iris.prompts import PromptSource
from iris.store import InMemoryLifecycleStore
from iris.utils.templating import TemplateRenderer
from tests.store.test_lifecycle_store_contract import _create_command


class _HostSource:
    """记录采集次数并返回原样宿主条目。"""

    def __init__(self, key: str = "host") -> None:
        self.calls = 0
        self.contribution = ContextContribution(key, "host text", required=False, priority=7)

    async def collect(self, scope: ContextBuildScope) -> ContextSnapshot:
        self.calls += 1
        return ContextSnapshot((self.contribution,))


def _scope() -> ContextBuildScope:
    return ContextBuildScope("session-1", "run-1", 1, Path.cwd(), "继续")


def _admitted() -> tuple[GoalService, str]:
    store = InMemoryLifecycleStore()
    service = GoalService(store)
    goal = service.create("session-1", "完整目标首行\n完整目标次行")
    admitted = store.admit_goal_run(AdmitGoalRun(expected=goal.ref, create_run=_create_command()))
    return service, admitted.goal.goal_id


@pytest.mark.asyncio
async def test_project_template_snapshot_preserves_dynamic_goal(tmp_path: Path) -> None:
    """旧快照保持正文，但每一步仍使用最新 Goal revision。"""
    prompts = PromptSource.initialize(tmp_path)
    template = prompts.root / "goal_context.j2"
    template.write_text("OLD {{ goal.revision }}", encoding="utf-8")
    service, goal_id = _admitted()
    source = GoalContextSource(service, prompt_snapshot=prompts.snapshot())
    assert (await source.collect(_scope())).contributions[0].text == "OLD 1"
    template.write_text("NEW {{ goal.revision }}", encoding="utf-8")
    service.pause(service.get(goal_id).ref, reason=GoalReason(code="user", text="等待"))
    assert (await source.collect(_scope())).contributions[0].text == "OLD 2"
    replacement = GoalContextSource(service, prompt_snapshot=prompts.snapshot())
    assert (await replacement.collect(_scope())).contributions[0].text == "NEW 2"


@pytest.mark.asyncio
async def test_context_collects_host_once_and_reads_latest_goal_each_step(tmp_path: Path) -> None:
    service, goal_id = _admitted()
    host = _HostSource()
    source = GoalContextSource(
        service, host_source=host, prompt_snapshot=PromptSource.initialize(tmp_path).snapshot()
    )
    first = await source.collect(_scope())
    assert host.calls == 1
    assert first.contributions[0] is host.contribution
    goal_item = first.contributions[1]
    assert goal_item.key == "iris.goal" and goal_item.required
    assert "完整目标首行\n完整目标次行" in goal_item.text
    assert "report_goal" in goal_item.text
    assert "revision=1" in goal_item.text
    goal = service.get(goal_id)
    service.pause(goal.ref, reason=GoalReason(code="user", text="先等等"))
    second = await source.collect(_scope())
    assert host.calls == 2
    assert "paused" in second.contributions[1].text
    assert "收尾" in second.contributions[1].text
    assert "先等等" in second.contributions[1].text


@pytest.mark.asyncio
async def test_old_goal_run_cannot_receive_replacement_objective(tmp_path: Path) -> None:
    service, goal_id = _admitted()
    source = GoalContextSource(
        service, prompt_snapshot=PromptSource.initialize(tmp_path).snapshot()
    )
    service.clear("session-1", expected=service.get(goal_id).ref)
    service.create("session-1", "新目标绝不能注入旧执行")
    context = await source.collect(_scope())
    text = context.contributions[0].text
    assert "原目标已停用" in text
    assert "新目标绝不能注入旧执行" not in text
    assert "完整目标首行" not in text


@pytest.mark.asyncio
async def test_ordinary_run_does_not_receive_goal_push_instructions(tmp_path: Path) -> None:
    service = GoalService(InMemoryLifecycleStore())
    service.create("session-1", "私有目标正文")
    source = GoalContextSource(
        service, prompt_snapshot=PromptSource.initialize(tmp_path).snapshot()
    )
    text = (await source.collect(_scope())).contributions[0].text
    assert "普通 Run" in text
    assert "私有目标正文" not in text


@pytest.mark.asyncio
async def test_host_goal_key_collision_is_configuration_error(tmp_path: Path) -> None:
    host = _HostSource("iris.goal")
    source = GoalContextSource(
        GoalService(InMemoryLifecycleStore()),
        host_source=host,
        prompt_snapshot=PromptSource.initialize(tmp_path).snapshot(),
    )
    with pytest.raises(IrisConfigError, match="iris.goal"):
        await source.collect(_scope())
    assert host.calls == 1


def test_goal_input_template_references_dynamic_goal_without_copying_objective(
    tmp_path: Path,
) -> None:
    service, goal_id = _admitted()
    text = render_continuation(
        service.get(goal_id), prompt_snapshot=PromptSource.initialize(tmp_path).snapshot()
    )
    assert goal_id in text
    assert "完整目标首行" not in text
    assert "report_goal" in text


@pytest.mark.asyncio
async def test_goal_template_failure_is_normalized_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: TemplateRenderer, template_path: Path, context: dict[str, object]) -> str:
        raise IrisTemplateError("broken")

    monkeypatch.setattr(TemplateRenderer, "render_file", broken)
    service, _ = _admitted()
    with pytest.raises(IrisGoalError) as caught:
        await GoalContextSource(
            service, prompt_snapshot=PromptSource.initialize(tmp_path).snapshot()
        ).collect(_scope())
    assert isinstance(caught.value.__cause__, IrisTemplateError)
