"""窗口采用只比较核心事实与知识范围或全部知识范围，并共享总预算。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from iris.context import ContextBuilder, ContextBuildInput, ContextSection, ContextSlot
from iris.exceptions import IrisContextError, IrisTemplateError
from iris.lifecycle import SessionContextWindow
from iris.memory import MemoryOverviewDocument, MemoryService, SQLiteMemoryStore
from iris.message import LLMRequest, LLMResponse, Msg
from iris.runtime.memory_context import load_context_windows, select_context_window
from iris.utils import TemplateRenderer


class TextTokenProvider:
    """用字符作为可解释的token测试单位。"""

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """计入全部消息和工具schema，以观察固定部分不会重复扣除。"""
        return sum(len(message.text) for message in request.messages) + len(str(request.tools))

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """采用概览不应调用模型。"""
        raise AssertionError("窗口采用不能生成摘要")


def _request(window: SessionContextWindow) -> LLMRequest:
    return LLMRequest(
        model="effective-main-model",
        messages=[
            Msg.system(
                "base" + ("\n\n" + window.memory_overview if window.memory_overview else "")
            ),
            Msg.user("question"),
        ],
        tools=[{"type": "function", "function": {"name": "memory_search"}}],
    )


@pytest.mark.parametrize("limit", [20, 21])
def test_combined_overview_budget_selects_full_or_all_navigation(limit: int) -> None:
    """两个 namespace 和包装共享限额，超限时保留全部知识范围。"""
    full = SessionContextWindow(memory_overview="facts-one|facts-two", mode="full")
    navigation = SessionContextWindow(memory_overview="topic1|topic2", mode="navigation")
    window, request = select_context_window(
        candidates=(full, navigation),
        build_request=_request,
        provider=TextTokenProvider(),
        memory_budget_tokens=limit,
        input_budget_tokens=1000,
    )
    assert window is (navigation if limit == 20 else full)
    assert request.messages[0].text.endswith(window.memory_overview)


def test_navigation_over_budget_is_an_explicit_capacity_error() -> None:
    """没有第三模式，也不截断知识范围。"""
    full = SessionContextWindow(memory_overview="long facts and navigation")
    navigation = SessionContextWindow(memory_overview="topic1|topic2", mode="navigation")
    with pytest.raises(IrisContextError, match="知识范围.*预算"):
        select_context_window(
            candidates=(full, navigation),
            build_request=_request,
            provider=TextTokenProvider(),
            memory_budget_tokens=12,
            input_budget_tokens=1000,
        )


def test_full_request_limit_can_select_navigation_without_charging_history_to_memory() -> None:
    """完整请求超限可触发导航，但可压缩历史不归入memory专用额度。"""
    full = SessionContextWindow(memory_overview="facts" * 10)
    navigation = SessionContextWindow(memory_overview="topics", mode="navigation")
    window, request = select_context_window(
        candidates=(full, navigation),
        build_request=_request,
        provider=TextTokenProvider(),
        memory_budget_tokens=100,
        input_budget_tokens=1,
    )
    assert window is navigation
    # 整体历史是否可压缩继续交给既有runtime，不把它误报成导航过大。
    assert TextTokenProvider().estimate_input_tokens(request) > 1


def test_system_character_limit_can_select_navigation(tmp_path: Path) -> None:
    """字符限制仍由ContextBuilder判断，采用逻辑只处理容量降级。"""
    template = tmp_path / "system.j2"
    template.write_text("base", encoding="utf-8")
    context = ContextBuildInput(
        system=ContextSection(
            slots=[ContextSlot(name="base", content="base")], template=template, max_chars=12
        )
    )

    def build(window: SessionContextWindow) -> LLMRequest:
        output = ContextBuilder().build(context, system_addendum=window.memory_overview)
        return LLMRequest(model="main", messages=[output.system, Msg.user("question")])

    full = SessionContextWindow(memory_overview="facts" * 10)
    navigation = SessionContextWindow(memory_overview="topics", mode="navigation")
    window, request = select_context_window(
        candidates=(full, navigation),
        build_request=build,
        provider=TextTokenProvider(),
        memory_budget_tokens=100,
        input_budget_tokens=1000,
    )
    assert window is navigation
    assert request.messages[0].text == "base\n\ntopics"


class DocumentsService(MemoryService):
    """提供一次完整文档读取，记录窗口读取顺序。"""

    def __init__(self, path: Path, documents: tuple[MemoryOverviewDocument, ...]) -> None:
        super().__init__(SQLiteMemoryStore(path))
        self.documents = documents
        self.reads: list[tuple[str, ...]] = []

    async def aload_overviews(
        self, namespaces: Sequence[str]
    ) -> tuple[MemoryOverviewDocument, ...]:
        """返回按配置顺序提供的可信文档。"""
        self.reads.append(tuple(namespaces))
        return self.documents


def _document(namespace: str) -> MemoryOverviewDocument:
    return MemoryOverviewDocument(
        namespace=namespace,
        path=Path(namespace) / "Memory.md",
        source_revision=2,
        text=f"事实 {namespace}：" + "确认的稳定事实。" * 10 + f"\n\n知识范围 {namespace}",
        navigation=f"知识范围 {namespace}",
        warning=f"{namespace} 概览可能旧于数据库当前条目。",
    )


@pytest.mark.asyncio
async def test_all_namespaces_instructions_and_warnings_share_the_actual_request_budget(
    tmp_path: Path,
) -> None:
    """按配置顺序读取一次，两个候选包含所有来源、警告、范围和工具指引。"""
    service = DocumentsService(
        tmp_path / "memory.db", (_document("research"), _document("project"))
    )
    full, navigation = await load_context_windows(
        prompt_renderer=TemplateRenderer(),
        memory_service=service,
        namespaces=["research", "project"],
        tool_names=["memory_search", "memory_fetch"],
    )
    assert service.reads == [("research", "project")]
    assert full.sources == navigation.sources
    assert [source.namespace for source in full.sources] == ["research", "project"]
    assert full.memory_overview.index("## research") < full.memory_overview.index("## project")
    for document in service.documents:
        assert document.text in full.memory_overview
        assert document.navigation in navigation.memory_overview
        assert document.warning in full.memory_overview
        assert document.warning in navigation.memory_overview
        assert f"事实 {document.namespace}" not in navigation.memory_overview
    provider = TextTokenProvider()
    full_cost = provider.estimate_input_tokens(_request(full)) - provider.estimate_input_tokens(
        _request(SessionContextWindow())
    )
    navigation_cost = provider.estimate_input_tokens(
        _request(navigation)
    ) - provider.estimate_input_tokens(_request(SessionContextWindow()))
    assert navigation_cost < full_cost
    window, _ = select_context_window(
        candidates=(full, navigation),
        build_request=_request,
        provider=provider,
        memory_budget_tokens=navigation_cost,
        input_budget_tokens=10000,
    )
    assert window is navigation
    with pytest.raises(IrisContextError, match="知识范围.*预算"):
        select_context_window(
            candidates=(full, navigation),
            build_request=_request,
            provider=provider,
            memory_budget_tokens=navigation_cost - 1,
            input_budget_tokens=10000,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_names", [[], ["memory_search"], ["memory_fetch"], ["memory_search", "memory_fetch"]]
)
async def test_instructions_follow_actual_tools_and_limit_queries_to_covered_topics(
    tmp_path: Path, tool_names: list[str]
) -> None:
    service = DocumentsService(tmp_path / "memory.db", (_document("project"),))
    candidates = await load_context_windows(
        prompt_renderer=TemplateRenderer(),
        memory_service=service,
        namespaces=["project"],
        tool_names=tool_names,
    )
    for window in candidates:
        text = window.memory_overview
        assert "没有提及的主题默认没有" in text
        assert "不搜索这些主题" in text
        assert "已覆盖" in text
        assert ("memory_search" in text) is ("memory_search" in tool_names)
        assert ("memory_fetch" in text) is ("memory_fetch" in tool_names)
        assert "read_file" not in text and "grep_search" not in text
        if tool_names == ["memory_fetch"]:
            assert "已知 item_id" in text
        if not tool_names:
            assert "当前没有专用数据库读取工具" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("with_mirror", [False, True])
async def test_missing_overview_keeps_chat_available_without_long_term_queries(
    tmp_path: Path, with_mirror: bool
) -> None:
    """缺产物只读取确定性说明；不扫描条目或生成知识范围。"""
    from iris.memory import FileMemoryMirror

    store = SQLiteMemoryStore(tmp_path / "memory.db")
    service = MemoryService(
        store, mirror=FileMemoryMirror(tmp_path / "mirror") if with_mirror else None
    )
    full, navigation = await load_context_windows(
        prompt_renderer=TemplateRenderer(),
        memory_service=service,
        namespaces=["project"],
        tool_names=["memory_search", "memory_fetch"],
    )
    assert full is navigation
    if with_mirror:
        assert full.sources[0].source_revision is None
    else:
        assert full.sources == ()
        assert full.memory_overview.endswith(
            "## project\n\n未配置概览发布物，本窗口不使用长期记忆。"
        )
    assert "## project" in full.memory_overview
    assert "没有概览则本窗口暂不使用长期记忆" in full.memory_overview
    assert "不查询长期记忆" in full.memory_overview
    assert not (tmp_path / "mirror").exists()


@pytest.mark.asyncio
async def test_no_service_or_empty_scope_initializes_an_empty_window(tmp_path: Path) -> None:
    service = DocumentsService(tmp_path / "memory.db", (_document("project"),))
    for memory_service, namespaces in ((None, ["project"]), (service, [])):
        full, navigation = await load_context_windows(
            prompt_renderer=TemplateRenderer(),
            memory_service=memory_service,
            namespaces=namespaces,
            tool_names=["memory_search"],
        )
        assert full == navigation == SessionContextWindow()
    assert service.reads == []


@pytest.mark.asyncio
async def test_window_template_preserves_exact_plain_text_and_document_whitespace(
    tmp_path: Path,
) -> None:
    """模板输出保持文案、段落和文档尾换行，正文不执行 XML 转义。"""
    document = MemoryOverviewDocument(
        namespace="project",
        path=tmp_path / "Memory.md",
        source_revision=2,
        text='  <fact> "A&B" {{ raw }}\n\n',
        navigation='范围 <topic> & "details"\n',
        warning="概览 <old> & current",
    )
    second_document = MemoryOverviewDocument(
        namespace="research",
        path=tmp_path / "Research.md",
        source_revision=2,
        text="第二份事实\n",
        navigation="第二份范围\n",
        warning="",
    )
    full, navigation = await load_context_windows(
        prompt_renderer=TemplateRenderer(),
        memory_service=DocumentsService(tmp_path / "memory.db", (document, second_document)),
        namespaces=["project", "research"],
        tool_names=[],
    )
    instructions = (
        "以当前概览为长期记忆范围；没有提及的主题默认没有，不搜索这些主题。"
        "仅对概览已覆盖且问题需要的主题按需读取，无关问题无需读取。"
        "没有概览则本窗口暂不使用长期记忆，正常聊天但不查询长期记忆。"
        "概览或文件未同步的提示不扩展主题范围，也不阻断已覆盖主题的数据库查询；"
        "概览可能旧于数据库当前记录，不能视为已核实的当前值。"
        " 当前没有专用数据库读取工具。"
    )
    heading = f"## project\n\n{document.warning}\n\n"
    assert full.memory_overview == (
        f"# Memory overview\n\n{instructions}\n\n{heading}{document.text}"
        f"\n\n## research\n\n{second_document.text}"
    )
    assert navigation.memory_overview == (
        "# Memory overview\n\n本窗口仅载入知识范围，未载入核心事实。\n\n"
        f"{instructions}\n\n{heading}{document.navigation}"
        f"\n\n## research\n\n{second_document.navigation}"
    )


@pytest.mark.asyncio
async def test_window_template_failure_is_a_context_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """读取器错误在窗口模板边界转换为 context 领域异常。"""
    missing = tmp_path / "missing.j2"
    monkeypatch.setattr("iris.runtime.memory_context._MEMORY_CONTEXT_PROMPT", missing)
    with pytest.raises(IrisContextError) as caught:
        await load_context_windows(
            prompt_renderer=TemplateRenderer(),
            memory_service=DocumentsService(tmp_path / "memory.db", (_document("project"),)),
            namespaces=["project"],
            tool_names=[],
        )
    assert caught.value.runtime_source == "context"
    assert caught.value.context["path"] == str(missing)
    assert isinstance(caught.value.__cause__, IrisTemplateError)
