"""装配 lifecycle 示例所需的 runner。

Example:
    >>> from pathlib import Path
    >>> runner = build_runner(
    ...     Path("examples/lifecycle/agent.yaml"),
    ...     env_file=None,
    ...     requires_provider=False,
    ... )
"""

# region imports
from pathlib import Path

from iris.config import init_config, is_config_initialized
from iris.exceptions import IrisRunStateError
from iris.harness import AgentRunner
from iris.message import LLMRequest, LLMResponse

# endregion


class NonExecutingProvider:
    """保证只读 lifecycle 示例不会调用真实 provider。

    Example:
        >>> provider = NonExecutingProvider()
    """

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """拒绝只读 lifecycle 示例发起的 provider 请求。

        Args:
            request (LLMRequest): 原本将发送给 provider 的标准化请求。

        Raises:
            IrisRunStateError: 始终抛出，避免只读示例意外访问 provider。
        """
        del request
        raise IrisRunStateError("当前 lifecycle 示例不允许调用 provider")


def build_runner(
    config_path: Path,
    *,
    env_file: Path | None,
    requires_provider: bool,
) -> AgentRunner:
    """按示例类型装配真实或禁止执行的 provider。

    Args:
        config_path (Path): lifecycle 示例 ``agent.yaml`` 的路径。
        env_file (Path | None): 可选的 provider 环境变量文件。
        requires_provider (bool): 是否允许装配真实 provider。

    Returns:
        AgentRunner: 使用示例配置和相应 provider 装配的 runner。
    """
    if not is_config_initialized():
        init_config(env_file=str(env_file) if env_file is not None else None)
    provider = None if requires_provider else NonExecutingProvider()
    return AgentRunner.from_config_path(config_path, provider=provider)
