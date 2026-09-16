"""摘要指令文件的配置与运行时使用。"""

from pathlib import Path

import pytest
from fakes import FakeProvider, FakeRuntimeCommitPort, MutableCancellationSignal, start_activation

from iris.agents import AgentConfig, CompactionConfig
from iris.message import LLMRequest, LLMResponse, Msg, TextBlock
from iris.runtime import RuntimeActivationOutcome, RuntimeFactory


class _PromptProvider(FakeProvider):
    """将首个旧历史请求触发压缩，同时保留实际摘要指令。"""

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        if request.provider_options.get("num_retries") == 0:
            return sum(len(message.text) for message in request.messages)
        if any(message.text.startswith("<summary>") for message in request.messages):
            return 1000
        return 95000


def _config(prompt: Path | None = None) -> AgentConfig:
    return AgentConfig(
        name="agent",
        model="openai/test",
        system="业务指令",
        compaction=CompactionConfig(prompt=prompt),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("custom", [False, True])
async def test_summary_uses_prompt_file_and_framework_provides_history(
    tmp_path: Path,
    custom: bool,
) -> None:
    prompt = tmp_path / "summary.j2"
    instructions = "只保留任务约束与待办，使用两个中文标题。"
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
        _config(prompt if custom else None),
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


def test_sdk_prompt_path_uses_supplied_config_directory(tmp_path: Path) -> None:
    prompt = tmp_path / "summary.j2"
    prompt.write_text("本地摘要指令", encoding="utf-8")
    runtime = RuntimeFactory.from_config(
        _config(Path("summary.j2")),
        config_path=tmp_path / "agent.yaml",
        provider=FakeProvider([]),
    )
    assert runtime.environment.agent_config.compaction.prompt == prompt.resolve()


@pytest.mark.asyncio
async def test_missing_prompt_fails_when_compaction_reads_it(tmp_path: Path) -> None:
    provider = _PromptProvider([])
    runtime = RuntimeFactory.from_config(_config(tmp_path / "missing.j2"), provider=provider)
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
