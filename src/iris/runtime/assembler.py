"""Runtime 消息与请求装配器。

本模块只负责把已准备好的 context 输出、history 和当前输入装配成 `Conversation`，
或进一步结合 Agent 配置生成 provider-neutral `LLMRequest`。

Example:
    conversation = RuntimeMessageAssembler().build_conversation(
        context_output=context_output,
        history=[],
        current_input=Msg.user("你好"),
    )
    request = RuntimeMessageAssembler().build_request(
        agent_config=config,
        context_output=context_output,
        history=[],
        current_input=Msg.user("你好"),
    )
"""

# region imports
from __future__ import annotations

from ..agents import AgentConfig
from ..context import ContextBuildOutput
from ..message import Conversation, LLMRequest, Msg

# endregion


class RuntimeMessageAssembler:
    """装配单次 provider 请求前的 runtime messages。

    Assembler 只处理顺序和 provider-neutral 请求构造；context 构建、session 读取、
    memory recall、工具执行和 provider 调用由外层 runtime 阶段负责。
    """

    def build_conversation(
        self,
        *,
        context_output: ContextBuildOutput,
        history: list[Msg],
        current_input: Msg | None,
    ) -> Conversation:
        """按 runtime 固定顺序构造会话消息。

        Args:
            context_output (ContextBuildOutput): 已构建完成的 context 输出。
            history (list[Msg]): 会话历史消息，由调用方或 session 层提供。
            current_input (Msg | None): 当前用户输入；后续 loop 步骤可传 None。

        Returns:
            Conversation: 包含本次请求完整消息顺序的会话快照。
        """
        messages = [context_output.system]
        if context_output.memory is not None:
            messages.append(context_output.memory)
        messages.extend(history)
        if context_output.before_current_input is not None:
            messages.append(context_output.before_current_input)
        if current_input is not None:
            messages.append(current_input)

        return Conversation(messages=messages)

    def build_request(
        self,
        *,
        agent_config: AgentConfig,
        context_output: ContextBuildOutput,
        history: list[Msg],
        current_input: Msg | None,
    ) -> LLMRequest:
        """按 runtime 固定顺序构造一次 LLMRequest。

        Args:
            agent_config (AgentConfig): 已加载并校验的 Agent 配置。
            context_output (ContextBuildOutput): 已构建完成的 context 输出。
            history (list[Msg]): 会话历史消息，由调用方或 session 层提供。
            current_input (Msg | None): 当前用户输入；后续 loop 步骤可传 None。

        Returns:
            LLMRequest: provider-neutral 的一次模型调用请求。
        """
        conversation = self.build_conversation(
            context_output=context_output,
            history=history,
            current_input=current_input,
        )

        return conversation.to_llm_request(
            model=agent_config.model.name,
            **agent_config.model.to_llm_request_options(),
        )


__all__ = ["RuntimeMessageAssembler"]
