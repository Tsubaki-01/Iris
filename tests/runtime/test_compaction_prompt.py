"""摘要指令文件的配置与运行时使用。"""

from pathlib import Path

import pytest
from fakes import FakeProvider, FakeRuntimeCommitPort, MutableCancellationSignal, start_activation

from iris.agents import AgentConfig
from iris.message import LLMRequest, LLMResponse, Msg, TextBlock
from iris.prompts import PromptSource
from iris.runtime import RuntimeActivationOutcome, RuntimeFactory


class _PromptProvider(FakeProvider):
    """将首个旧历史请求触发压缩，同时保留实际摘要指令。"""

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        if request.provider_options.get("num_retries") == 0:
            return sum(len(message.text) for message in request.messages)
        if any(message.text.startswith("<summary>") for message in request.messages):
            return 1000
        return 95000


def _config(workspace: Path) -> AgentConfig:
    return AgentConfig(
        name="agent",
        model="openai/test",
        system="业务指令",
        context_policy={"enabled": False},
        permissions={"workspace": str(workspace)},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", [False, True])
async def test_summary_uses_prompt_file_and_framework_provides_history(
    tmp_path: Path,
    custom: bool,
) -> None:
    source = PromptSource.initialize(tmp_path)
    prompt = source.root / "compaction.j2"
    instructions = "只保留任务约束与待办，使用两个中文标题。"
    if custom:
        prompt.write_text(instructions, encoding="utf-8")
    provider = _PromptProvider(
        [
            LLMResponse(
                provider="fake", finish_reason="stop", content=[TextBlock(text="摘要结果")]
            ),
            LLMResponse(provider="fake", finish_reason="stop", content=[TextBlock(text="完成")]),
        ]
    )
    runtime = RuntimeFactory.from_config(
        _config(tmp_path),
        provider=provider,
    )
    activation = start_activation(input="新任务", initial_session_message_count=1)
    commits = FakeRuntimeCommitPort(activation, messages=[Msg.user("已经保存的历史")])

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    summary, main = provider.requests
    if custom:
        assert summary.messages[0].text == instructions
    else:
        assert "## Goal & Constraints" in summary.messages[0].text
        assert "## Critical Paths & Identifiers" in summary.messages[0].text
    assert "=== BEGIN PREVIOUS SUMMARY ===\n(none)" in summary.messages[1].text
    assert "已经保存的历史" in summary.messages[1].text
    assert "新任务" not in summary.messages[1].text
    assert main.messages[-1].text == "新任务"


def test_sdk_prompt_root_uses_workspace_and_preserves_injected_source(tmp_path: Path) -> None:
    source = PromptSource.initialize(tmp_path, "shared-prompts")
    missing = source.root / "compaction.j2"
    missing.unlink()
    runtime = RuntimeFactory.from_config(
        _config(tmp_path / "child"),
        config_path=tmp_path / "config" / "agent.yaml",
        provider=FakeProvider([]),
        prompt_source=source,
    )
    assert runtime.environment.prompt_source is source
    assert runtime.environment.prompt_snapshot.root == source.root
    assert not missing.exists()
    assert not (tmp_path / "child" / ".iris" / "prompts").exists()


@pytest.mark.asyncio
async def test_missing_prompt_fails_when_compaction_reads_it(tmp_path: Path) -> None:
    provider = _PromptProvider([])
    runtime = RuntimeFactory.from_config(_config(tmp_path), provider=provider)
    (runtime.environment.prompt_source.root / "compaction.j2").unlink()
    activation = start_activation(input="新任务", initial_session_message_count=1)
    commits = FakeRuntimeCommitPort(activation, messages=[Msg.user("已经保存的历史")])

    result = await runtime.execute(
        activation,
        commits=commits,
        cancellation=MutableCancellationSignal(),
    )

    assert result.outcome is RuntimeActivationOutcome.FAILED
    assert result.error is not None and result.error.source == "context"
    assert "模板来源" in result.error.message
    assert provider.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("xml", [False, True])
async def test_compaction_template_uses_plain_text_unless_explicitly_escaped(
    tmp_path: Path, xml: bool
) -> None:
    """摘要指令独立于 ContextBuilder，XML 输出由模板显式选择。"""
    source = PromptSource.initialize(tmp_path)
    prompt = source.root / "compaction.j2"
    expression = "{{ '<a>&' }}"
    if xml:
        expression = "{% autoescape true %}" + expression + "{% endautoescape %}"
    prompt.write_text("  " + expression + "  ", encoding="utf-8")
    provider = _PromptProvider(
        [
            LLMResponse(provider="fake", finish_reason="stop", content=[TextBlock(text=text)])
            for text in ("摘要结果", "完成")
        ]
    )
    runtime = RuntimeFactory.from_config(_config(tmp_path), provider=provider)
    activation = start_activation(input="新任务", initial_session_message_count=1)
    commits = FakeRuntimeCommitPort(activation, messages=[Msg.user('原文 <a>&"')])
    result = await runtime.execute(
        activation, commits=commits, cancellation=MutableCancellationSignal()
    )
    assert result.outcome is RuntimeActivationOutcome.COMPLETED
    summary = provider.requests[0]
    assert summary.messages[0].text == ("&lt;a&gt;&amp;" if xml else "<a>&")
    assert '原文 <a>&"' in summary.messages[1].text
