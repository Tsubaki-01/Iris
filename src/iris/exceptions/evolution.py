"""项目经验维护与有限策略修订的领域失败。"""

from .base import IrisError


class IrisEvolutionError(IrisError):
    """自进化后台处理失败，不改变业务 Run 的完成结果。"""

    runtime_error_code = "EVOLUTION_ERROR"
