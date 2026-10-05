"""Decision 接点开关、SDK 注入与自有资源的装配边界。"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fakes import FakeProvider

from iris.agents import AgentConfig, ToolsConfig
from iris.config import Config
from iris.decision import ChoiceAnswer, DecisionRequest, DecisionResponse, DecisionUsage
from iris.decision import factory as decision_factory
from iris.exceptions import IrisCommandError, IrisConfigError, IrisMCPError
from iris.harness import AgentRunner, AgentRunRequest
from iris.harness._context_access import ContextAccess
from iris.message import ToolUseBlock
from iris.runtime import RuntimeFactory
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability

from ..harness.fakes import StaticProvider, text_response, tool_response


class BorrowedEvaluator:
    """借用对象只实现 evaluate，不要求可关闭能力。"""

    async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        raise AssertionError("构造/关闭不得调用 Decision")


class OwnedEvaluator(BorrowedEvaluator):
    """记录 factory 创建和关闭的确定性替身。"""

    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs
        self.requests: list[DecisionRequest] = []
        self.closed = 0

    async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        """为本测试的单候选发现选择唯一工具。"""
        self.requests.append(request)
        return DecisionResponse(
            provider="typesafe",
            model=self.options["model"],
            answers={
                key: ChoiceAnswer(choice="c0", probabilities={"c0": 1.0}, confidence=1.0)
                for key in request.questions
            },
            usage=DecisionUsage(input_tokens=10, output_tokens=1),
        )

    async def aclose(self) -> None:
        self.closed += 1


def _mock_owned_clients(monkeypatch: pytest.MonkeyPatch) -> list[OwnedEvaluator]:
    """仅替换外部客户端构造，保留配置、装配和 runner 的真实生命周期。"""
    clients: list[OwnedEvaluator] = []

    def create(**kwargs: Any) -> OwnedEvaluator:
        client = OwnedEvaluator(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(
        decision_factory,
        "get_config",
        lambda: Config(provider_api_keys={"typesafe": "decision-key"}),
    )
    monkeypatch.setattr(decision_factory, "JevClient", create)
    return clients


def _agent(tmp_path: Path, *, enabled: bool, deferred: bool = True) -> AgentConfig:
    (tmp_path / "decision.yaml").write_text(
        f"tools:\n  discovery: {str(enabled).lower()}\n", encoding="utf-8"
    )
    return AgentConfig.model_validate(
        {
            "name": "test",
            "model": "openai/test",
            "system": "instructions",
            "decision": {"path": "decision.yaml"},
            "context_policy": {"enabled": True, "deferred_tools": deferred},
            "permissions": {"workspace": str(tmp_path / "workspace")},
        }
    )


def _no_keys() -> None:
    raise AssertionError("此路径不得读取全局凭据")


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["runner", "runtime"])
@pytest.mark.parametrize("from_yaml", [False, True])
async def test_sdk_entrypoints_borrow_evaluate_only_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str, from_yaml: bool
) -> None:
    import yaml

    config = _agent(tmp_path, enabled=True)
    monkeypatch.setattr(decision_factory, "get_config", _no_keys)
    client = BorrowedEvaluator()
    entry = AgentRunner if owner == "runner" else RuntimeFactory
    kwargs: dict[str, Any] = {"provider": FakeProvider([]), "decision_client": client}
    if owner == "runtime":
        kwargs["context_access"] = ContextAccess(InMemoryLifecycleStore())
    path = tmp_path / "agent.yaml"
    if from_yaml:
        path.write_text(
            yaml.safe_dump(config.model_dump(mode="json", exclude_unset=True)), encoding="utf-8"
        )
        value = entry.from_config_path(path, **kwargs)
    else:
        value = entry.from_config(config, config_path=path, **kwargs)
    runtime = value.runtime if owner == "runner" else value
    assert runtime.environment.decision_client is client
    assert runtime.environment.owned_decision_client is None
    search = runtime.environment.tool_bridge.tool_view.registry.get("tool_search")
    assert search.definition.capabilities == {ToolCapability.READ, ToolCapability.NETWORK}
    if owner == "runner":
        await value.aclose()
    else:
        await runtime.environment.aclose()


@pytest.mark.asyncio
async def test_disabled_does_not_read_keys_or_use_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _agent(tmp_path, enabled=False)
    monkeypatch.setattr(decision_factory, "get_config", _no_keys)
    runner = AgentRunner.from_config(
        config,
        config_path=tmp_path / "agent.yaml",
        provider=FakeProvider([]),
        decision_client=BorrowedEvaluator(),
    )
    assert runner.runtime.environment.decision_client is None
    assert runner.runtime.environment.owned_decision_client is None
    search = runner.runtime.environment.tool_bridge.tool_view.registry.get("tool_search")
    assert search.definition.capabilities == {ToolCapability.READ}
    await runner.aclose()


def test_enabled_discovery_requires_deferred_before_reading_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _agent(tmp_path, enabled=True, deferred=False)
    monkeypatch.setattr(decision_factory, "get_config", _no_keys)
    with pytest.raises(IrisConfigError, match="deferred"):
        AgentRunner.from_config(
            config, config_path=tmp_path / "agent.yaml", provider=FakeProvider([])
        )


@pytest.mark.asyncio
async def test_sdk_relative_decision_path_uses_cwd_without_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _agent(tmp_path, enabled=True)
    monkeypatch.chdir(tmp_path)
    runner = AgentRunner.from_config(
        config, provider=FakeProvider([]), decision_client=BorrowedEvaluator()
    )
    assert runner.runtime.environment.decision_client is not None
    await runner.aclose()


@pytest.mark.asyncio
async def test_owned_client_is_reused_across_sessions_and_closed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """两次离线发现使用同一自建 evaluator，完整 runner 关闭才释放它。"""
    owned_clients = _mock_owned_clients(monkeypatch)
    discovery = tool_response(
        ToolUseBlock(id="search", name="tool_search", input={"queries": ["alpha"]})
    )
    runner = AgentRunner.from_config(
        _agent(tmp_path, enabled=True),
        config_path=tmp_path / "agent.yaml",
        provider=StaticProvider(discovery, text_response(), discovery, text_response()),
    )
    client = owned_clients[0]
    runner.runtime.environment.tool_bridge.tool_view.registry.register_function(
        lambda: "alpha", name="alpha", deferred=True
    )
    try:
        for session in ("one", "two"):
            result = await runner.start(AgentRunRequest(input="查找 alpha", session_id=session))
            assert result.run.stop_reason.value == "completed"
            assert runner.runtime.environment.decision_client is client
            assert client.closed == 0
        assert owned_clients == [client]
        assert runner.runtime.environment.owned_decision_client is client
        assert len(client.requests) == 2
        assert client.options["api_key"] == "decision-key"
    finally:
        await runner.aclose()
    await runner.aclose()
    assert client.closed == 1


@pytest.mark.asyncio
async def test_each_child_builds_and_closes_its_own_client_without_root_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """两个 child 读取自己的 Decision 文件，自建资源在返回 parent 前已经关闭。"""
    owned_clients = _mock_owned_clients(monkeypatch)
    root_config = _agent(tmp_path, enabled=True).model_copy(
        update={"tools": ToolsConfig(subagent=tmp_path / "subagents.yaml")}
    )
    (tmp_path / "subagents.yaml").write_text(
        "default: alpha\nagents:\n"
        "  alpha: {path: alpha.yaml, description: Alpha child}\n"
        "  beta: {path: beta.yaml, description: Beta child}\n",
        encoding="utf-8",
    )
    for name in ("alpha", "beta"):
        (tmp_path / f"{name}.yaml").write_text(
            f"name: {name}\nmodel: openai/test\nsystem: Child instructions\n"
            "permissions: {workspace: workspace}\n"
            "context_policy: {deferred_tools: true}\n"
            f"decision: {{path: decision-{name}.yaml}}\n",
            encoding="utf-8",
        )
        (tmp_path / f"decision-{name}.yaml").write_text(
            f"model: jev-child-{name}\ntools: {{discovery: true}}\n", encoding="utf-8"
        )
    root_client = OwnedEvaluator()
    runner = AgentRunner.from_config(
        root_config,
        config_path=tmp_path / "agent.yaml",
        provider=StaticProvider(
            *(
                tool_response(
                    ToolUseBlock(
                        id=f"delegate-{name}",
                        name="subagent",
                        input={"agent": name, "prompt": "Complete child task"},
                    )
                )
                for name in ("alpha", "beta")
            ),
            text_response("Parent complete"),
        ),
        child_provider_factory=lambda config, *, config_path: StaticProvider(text_response()),
        decision_client=root_client,
    )
    try:
        assert owned_clients == []
        result = await runner.start(AgentRunRequest(input="Delegate", run_id="parent"))
        assert result.run.stop_reason.value == "completed"
        assert [client.options["model"] for client in owned_clients] == [
            "jev-child-alpha",
            "jev-child-beta",
        ]
        assert [client.closed for client in owned_clients] == [1, 1]
        assert runner.runtime.environment.decision_client is root_client
        assert root_client.closed == 0
    finally:
        await runner.aclose()
    assert root_client.closed == 0
    assert [client.closed for client in owned_clients] == [1, 1]


def test_missing_typesafe_key_fails_during_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """主聊天 key 不能替代启用 Decision 所需的独立凭据。"""
    monkeypatch.setattr(
        decision_factory,
        "get_config",
        lambda: Config.model_construct(api_key="chat-key", provider_api_keys={}),
    )
    with pytest.raises(IrisConfigError, match="provider_api_keys.typesafe"):
        AgentRunner.from_config(
            _agent(tmp_path, enabled=True),
            config_path=tmp_path / "agent.yaml",
            provider=FakeProvider([]),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["mcp", "command"])
async def test_earlier_resource_close_failure_still_closes_owned_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource: str,
) -> None:
    """环境的前序资源关闭失败，仍释放自有 Decision 并保留原异常。"""
    owned_clients = _mock_owned_clients(monkeypatch)
    runner = AgentRunner.from_config(
        _agent(tmp_path, enabled=True),
        config_path=tmp_path / "agent.yaml",
        provider=FakeProvider([]),
    )
    environment = runner.runtime.environment
    if resource == "mcp":
        error = IrisMCPError("mcp close failed")
        monkeypatch.setattr(
            environment, "mcp_manager", SimpleNamespace(aclose=AsyncMock(side_effect=error))
        )
    else:
        error = IrisCommandError("command close failed")
        monkeypatch.setattr(
            environment.command_binding.service, "aclose", AsyncMock(side_effect=error)
        )
    with pytest.raises(type(error)) as raised:
        await runner.aclose()
    assert raised.value is error
    assert owned_clients[0].closed == 1
