"""项目模板在运行时构造和完整压缩边界采用。"""

from pathlib import Path

import pytest
import yaml
from fakes import FakeProvider, FakeRuntimeCommitPort, MutableCancellationSignal, start_activation

from iris.agents import AgentConfig
from iris.context import ContextBuilder, ContextSlot
from iris.exceptions import IrisContextError, IrisTemplateError
from iris.message import LLMRequest, LLMResponse, Msg, TextBlock
from iris.prompts import PromptSource
from iris.runtime import RuntimeActivationOutcome, RuntimeFactory


def test_runtime_freezes_context_sources_but_not_slot_data(tmp_path: Path) -> None:
    """动态依赖与入口正文冻结，槽值仍来自每次调用；独立 SDK 仍读新文件。"""
    templates = tmp_path / "templates"
    templates.mkdir()
    template = templates / "system.j2"
    template.write_text("{% include slots[0].content %}:{{ slots[1].content }}", encoding="utf-8")
    dependency = templates / "detail.j2"
    dependency.write_text("旧正文", encoding="utf-8")
    context = tmp_path / "context.yaml"
    context.write_text(
        yaml.safe_dump(
            {
                "system": {
                    "template": "templates/system.j2",
                    "slots": [
                        {"name": "dependency", "content": "detail.j2"},
                        {"name": "value", "content": "初值"},
                    ],
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config = AgentConfig(
        name="context",
        model="openai/test",
        context={"path": str(context)},
        permissions={"workspace": str(tmp_path)},
        context_policy={"enabled": False},
    )
    runtime = RuntimeFactory.from_config(config, provider=FakeProvider([]))
    dependency.write_text("新正文", encoding="utf-8")
    data = runtime.environment.context_input
    updated = data.model_copy(
        update={
            "system": data.system.model_copy(
                update={
                    "slots": [
                        data.system.slots[0],
                        ContextSlot(name="value", content="新值"),
                    ]
                }
            )
        }
    )
    assert runtime.environment.context_builder.build(updated).system.text == "旧正文:新值"
    assert ContextBuilder().build(updated).system.text == "新正文:新值"
    rebuilt = RuntimeFactory.from_config(config, provider=FakeProvider([]))
    assert rebuilt.environment.context_builder.build(updated).system.text == "新正文:新值"


def test_context_snapshot_read_failure_keeps_context_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """装配提前读取模板时继续使用 context 领域异常与原路径。"""
    template = tmp_path / "system.j2"
    template.write_text("{{ slots[0].content }}", encoding="utf-8")
    context = tmp_path / "context.yaml"
    context.write_text(
        "system:\n  template: system.j2\n  slots:\n    - name: instructions\n      content: base\n",
        encoding="utf-8",
    )
    read_bytes = Path.read_bytes

    def fail_template(path: Path) -> bytes:
        if path == template:
            raise OSError("template unavailable")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_template)
    with pytest.raises(IrisContextError) as caught:
        RuntimeFactory.from_config(
            AgentConfig(
                name="context",
                model="openai/test",
                context={"path": str(context)},
                permissions={"workspace": str(tmp_path)},
                context_policy={"enabled": False},
            ),
            provider=FakeProvider([]),
        )
    assert caught.value.context["path"] == str(template)
    assert isinstance(caught.value.__cause__, IrisTemplateError)


@pytest.mark.asyncio
async def test_all_compaction_batches_and_estimates_share_one_source(tmp_path: Path) -> None:
    """首批改写指令及动态 include 后，本次计量/请求仍旧版，下次压缩采用新版。"""
    source = PromptSource.initialize(tmp_path)
    (source.root / "compaction.j2").write_text("策略旧", encoding="utf-8")
    (source.root / "compaction_input.j2").write_text(
        "{% set dependency = 'input-part.j2' %}{% include dependency %}\n"
        "{{ previous_summary_or_none }}\n{{ serialized_history }}",
        encoding="utf-8",
    )
    dependency = source.root / "input-part.j2"
    dependency.write_text("输入旧", encoding="utf-8")

    class Provider(FakeProvider):
        """每批最多容纳约一条历史，并在第一批完成时编辑项目源。"""

        def __init__(self) -> None:
            super().__init__([])
            self.estimates: list[LLMRequest] = []
            self.changed = False

        def estimate_input_tokens(self, request: LLMRequest) -> int:
            if request.provider_options.get("num_retries") == 0:
                self.estimates.append(request)
                return sum(len(message.text) for message in request.messages)
            return sum(len(message.text) for message in request.messages) + 100

        async def complete(self, request: LLMRequest) -> LLMResponse:
            self._requests.append(request)
            summary = request.provider_options.get("num_retries") == 0
            if summary and not self.changed:
                self.changed = True
                (source.root / "compaction.j2").write_text("策略新", encoding="utf-8")
                dependency.write_text("输入新", encoding="utf-8")
                (source.root / "compaction_input.j2").write_text(
                    "{% include 'input-part.j2' %}\n新版入口\n"
                    "{{ previous_summary_or_none }}\n{{ serialized_history }}",
                    encoding="utf-8",
                )
            return LLMResponse(
                provider="fake",
                finish_reason="stop",
                content=[TextBlock(text="摘要" if summary else "完成")],
            )

    provider = Provider()
    runtime = RuntimeFactory.from_config(
        AgentConfig(
            name="batch",
            model="openai/test",
            system="业务指令",
            permissions={"workspace": str(tmp_path)},
            context_policy={"enabled": False},
            compaction={"input_budget_tokens": 500, "keep_recent_ratio": 0.1},
        ),
        provider=provider,
        prompt_source=source,
    )
    for operation, expected in enumerate(("旧", "新")):
        first_request = len(provider.requests)
        first_estimate = len(provider.estimates)
        activation = start_activation(input="任务", initial_session_message_count=4)
        result = await runtime.execute(
            activation,
            commits=FakeRuntimeCommitPort(
                activation, messages=[Msg.user(f"历史{index}" * 120) for index in range(4)]
            ),
            cancellation=MutableCancellationSignal(),
        )
        assert result.outcome is RuntimeActivationOutcome.COMPLETED, result.error
        summaries = [
            request
            for request in provider.requests[first_request:]
            if request.provider_options.get("num_retries") == 0
        ]
        assert len(summaries) > 1
        for request in [*summaries, *provider.estimates[first_estimate:]]:
            assert request.messages[0].text == f"策略{expected}"
            assert request.messages[1].text.startswith(f"输入{expected}")
            assert ("新版入口" in request.messages[1].text) is bool(operation)
