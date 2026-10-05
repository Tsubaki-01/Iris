"""将当前 Todo 文档投影为每个模型步骤的必要上下文。"""

from ..context import ContextContribution
from ..exceptions import IrisTemplateError, IrisTodoError
from ..prompts import PromptSnapshot
from .models import TodoSnapshot, TodoStatus

_MARKERS = {
    TodoStatus.PENDING: " ",
    TodoStatus.IN_PROGRESS: "-",
    TodoStatus.COMPLETED: "x",
}


def render_todo_context(
    snapshot: TodoSnapshot,
    *,
    prompt_snapshot: PromptSnapshot,
    remind: bool = False,
) -> ContextContribution:
    """用已解析的快照构造 required contribution，不读取或重新校验清单。

    Args:
        snapshot: 当前步骤读取所得的不可变文件快照。
        prompt_snapshot: 构造时固定的模板正文和依赖，业务数据仍由本次传入。
        remind: 当前步骤是否为已安排的唯一结束自查步骤。

    Returns:
        保留完整条目、实际路径与维护说明的 iris.todo contribution。

    Raises:
        IrisTodoError: Todo 模板无法读取、解析或渲染。
    """
    variables = {"snapshot": snapshot, "markers": _MARKERS}
    parts: list[str] = []
    for template in ("todo_context", "todo_reminder") if remind else ("todo_context",):
        try:
            parts.append(prompt_snapshot.render(template, variables))
        except IrisTemplateError as exc:
            raise IrisTodoError(
                "Todo 提示模板渲染失败",
                template=str(prompt_snapshot.root / f"{template}.j2"),
                error=str(exc),
            ) from exc
    return ContextContribution("iris.todo", "\n\n".join(parts), required=True)
