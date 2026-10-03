"""基于既有配置构建或借用 Decision evaluator。"""

from ..config import get_config
from ..exceptions import IrisConfigError
from .client import DecisionEvaluator
from .config import DecisionConfig
from .jev import JevClient


def build_decision_client(
    config: DecisionConfig,
    *,
    decision_client: DecisionEvaluator | None = None,
) -> tuple[DecisionEvaluator | None, JevClient | None]:
    """返回业务使用的 evaluator 与环境负责关闭的自有对象。

    Args:
        config: 已解析且跨领域依赖已在 assembly 确认的配置。
        decision_client: 可选宿主对象，只要求 evaluate，不转移关闭权。

    Returns:
        evaluator 和 owned client；关闭所有接点时两者均为空。

    Raises:
        IrisConfigError: 启用接点但未配置 TypeSafe 凭据。
    """
    if not config.enabled:
        return None, None
    if decision_client is not None:
        return decision_client, None
    key = get_config().provider_api_keys.get("typesafe")
    if key is None:
        raise IrisConfigError("启用 Decision 需要 provider_api_keys.typesafe")
    client = JevClient(api_key=key, model=config.model, timeout_seconds=config.timeout_seconds)
    return client, client


__all__ = ["build_decision_client"]
