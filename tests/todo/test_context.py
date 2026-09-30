"""Todo 当前快照及文件维护规则的模型上下文。"""

from pathlib import Path

import pytest

from iris.exceptions import IrisTemplateError, IrisTodoError
from iris.todo import TodoItem, TodoSnapshot, TodoStatus
from iris.todo import context as todo_context
from iris.todo.context import render_todo_context


def test_context_preserves_current_path_order_status_and_plain_text(tmp_path: Path) -> None:
    """当前文件位置和完整任务投影为 required 的独立 contribution。"""
    path = tmp_path / ".iris/todos/776f726b.md"
    contribution = render_todo_context(
        TodoSnapshot(
            path,
            (
                TodoItem("第一项 <tag>&保持文本", TodoStatus.PENDING),
                TodoItem("进行中", TodoStatus.IN_PROGRESS),
                TodoItem("进行中", TodoStatus.IN_PROGRESS),
                TodoItem("已完成", TodoStatus.COMPLETED),
            ),
            None,
        )
    )
    assert contribution.key == "iris.todo"
    assert contribution.required
    assert str(path) in contribution.text
    assert (
        "- [ ] 第一项 <tag>&保持文本\n- [-] 进行中\n- [-] 进行中\n- [x] 已完成" in contribution.text
    )


def test_empty_context_keeps_location_and_editing_instructions(tmp_path: Path) -> None:
    """无清单时仍告知模型实际路径与普通文件读写规则。"""
    path = tmp_path / ".iris/todos/656d707479.md"
    text = render_todo_context(TodoSnapshot(path, (), None)).text
    assert str(path) in text
    assert "当前清单为空" in text
    assert "read_file" in text
    assert "write_file" in text
    assert "edit_file" in text
    assert "读记录" in text
    assert "父会话" in text
    assert "用户指令" in text
    assert "简单" in text
    assert "替换或清空" in text
    assert "移除" in text
    assert "自动验收" in text
    assert "假报完成" in text
    assert "权限" in text
    assert "另一个模型步骤单独" in text
    assert "report_goal" in text


def test_format_diagnostic_is_not_rendered_as_empty_or_completed(tmp_path: Path) -> None:
    """无法解析时先展示诊断，不能把空 items 当作任务已结束。"""
    path = tmp_path / ".iris/todos/626164.md"
    diagnostic = "Todo 第 4 行格式错误：未知标记。"
    text = render_todo_context(TodoSnapshot(path, (), diagnostic)).text
    assert str(path) in text
    assert diagnostic in text
    assert "修复" in text
    assert "当前清单为空" not in text
    assert "全部完成" not in text


def test_completed_state_does_not_remove_tasks(tmp_path: Path) -> None:
    """模型仍看到已完成条目和进度声明的证据边界。"""
    text = render_todo_context(
        TodoSnapshot(
            tmp_path / ".iris/todos/776f726b.md",
            (TodoItem("已完成事项", TodoStatus.COMPLETED),),
            None,
        )
    ).text
    assert "- [x] 已完成事项" in text
    assert "当前清单为空" not in text
    assert "自动验收" in text


@pytest.mark.parametrize("template_content", [None, "{% invalid %}"])
def test_template_failure_is_normalized_once_with_template_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, template_content: str | None
) -> None:
    """缺失与语法错误由共享 renderer 报告，再由 Todo 边界转换一次。"""
    template = tmp_path / "todo_context.j2"
    if template_content is not None:
        template.write_text(template_content, encoding="utf-8")
    monkeypatch.setattr(todo_context, "_TEMPLATE", template)
    with pytest.raises(IrisTodoError) as caught:
        render_todo_context(TodoSnapshot(tmp_path / "todo.md", (), None))
    error = caught.value
    assert error.runtime_source == "runtime"
    assert error.runtime_code == "TODO_ERROR"
    assert error.context["template"] == str(template)
    assert isinstance(error.__cause__, IrisTemplateError)
    assert error.__cause__.context["path"] == str(template)


@pytest.mark.parametrize("remind", [False, True])
@pytest.mark.parametrize("state", ["pending", "empty", "invalid"])
def test_reminder_only_projects_when_requested(tmp_path: Path, remind: bool, state: str) -> None:
    """目标步按已安排标记自查，最新文件变空或格式出错也使用同一指令。"""
    snapshot = TodoSnapshot(
        tmp_path / "todo.md",
        (TodoItem("需要检查", TodoStatus.PENDING),) if state == "pending" else (),
        "Todo 第 2 行格式错误" if state == "invalid" else None,
    )
    contribution = render_todo_context(snapshot, remind=remind)
    assert contribution.required
    assert ("Todo 结束自查" in contribution.text) == remind
    if remind:
        assert "最新" in contribution.text
        assert "允许结束" in contribution.text
        assert "假报完成" in contribution.text
        assert "修复" in contribution.text
        assert "移除" in contribution.text
        assert "状态" in contribution.text


def test_reminder_template_failure_reports_its_own_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """自查模板缺失不会误报主模板路径，且关闭提醒不读取该模板。"""
    missing = tmp_path / "todo_reminder.j2"
    monkeypatch.setattr(todo_context, "_REMINDER_TEMPLATE", missing)
    snapshot = TodoSnapshot(tmp_path / "todo.md", (), None)
    assert "Todo 结束自查" not in render_todo_context(snapshot).text
    with pytest.raises(IrisTodoError) as caught:
        render_todo_context(snapshot, remind=True)
    assert caught.value.context["template"] == str(missing)
    assert caught.value.runtime_code == "TODO_ERROR"
    assert isinstance(caught.value.__cause__, IrisTemplateError)
    assert caught.value.__cause__.context["path"] == str(missing)
