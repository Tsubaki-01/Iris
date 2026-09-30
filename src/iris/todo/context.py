"""将当前 Todo 文档投影为每个模型步骤的必要上下文。"""

from pathlib import Path

from ..context import ContextContribution
from ..exceptions import IrisTemplateError, IrisTodoError
from ..utils.templating import TemplateRenderer
from .models import TodoSnapshot, TodoStatus

_TEMPLATE = Path(__file__).resolve().parents[1] / "prompts" / "todo_context.j2"
_REMINDER_TEMPLATE = _TEMPLATE.with_name("todo_reminder.j2")
_RENDERER = TemplateRenderer()
_MARKERS = {
    TodoStatus.PENDING: " ",
    TodoStatus.IN_PROGRESS: "-",
    TodoStatus.COMPLETED: "x",
}


def render_todo_context(snapshot: TodoSnapshot, *, remind: bool = False) -> ContextContribution:
    """用已解析的快照构造 required contribution，不读取或重新校验清单。

    Args:
        snapshot: 当前步骤读取所得的不可变文件快照。
        remind: 当前步骤是否为已安排的唯一结束自查步骤。

    Returns:
        保留完整条目、实际路径与维护说明的 iris.todo contribution。

    Raises:
        IrisTodoError: Todo 模板无法读取、解析或渲染。
    """
    variables = {"snapshot": snapshot, "markers": _MARKERS}
    parts: list[str] = []
    for template in (_TEMPLATE, _REMINDER_TEMPLATE) if remind else (_TEMPLATE,):
        try:
            parts.append(_RENDERER.render_file(template, variables))
        except IrisTemplateError as exc:
            raise IrisTodoError(
                "Todo 提示模板渲染失败", template=str(template), error=str(exc)
            ) from exc
    return ContextContribution("iris.todo", "\n\n".join(parts), required=True)
