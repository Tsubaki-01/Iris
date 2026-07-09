"""DeepSeek memory live 验证场景。"""

from __future__ import annotations

from pathlib import Path

from iris.memory import (
    MemoryCategory,
    MemoryItem,
    MemoryItemKind,
    MemoryLevel,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)
from iris.runtime import RuntimeFactory
from iris.runtime.models import RuntimeOptions

from .fixtures import prepare_text_agent
from .models import ScenarioReport
from .providers import recording_provider
from .reporting import scenario_report
from .utils import _runtime_error_code, _runtime_error_message


async def run_memory_results_live(work_dir: Path, retries: int) -> ScenarioReport:
    """验证显式 memory_results 会进入真实 API 请求并影响回答。"""
    del retries
    token = "MEMORY_RESULTS_TOKEN_0708"
    agent_path = prepare_text_agent(work_dir, "memory-results-live")
    provider = recording_provider()
    runtime = RuntimeFactory.from_config_path(agent_path, provider=provider)
    result = await runtime.run_loop(
        f"根据记忆回答，只输出 MEMORY_RESULTS_OK: {token}",
        options=RuntimeOptions(
            session_id="memory-results-live",
            memory_results=[_memory_result(token=token, item_id="memory-results-1")],
            memory_max_chars=300,
            metadata={"scenario": "memory_results_live"},
        ),
    )
    final_text = (
        result.assistant_message.text.strip() if result.assistant_message else ""
    )
    prompt_text = "\n".join(message.text for message in provider.requests[0].messages)
    ok = (
        result.status.value == "ok"
        and "MEMORY_RESULTS_OK" in final_text
        and token in final_text
        and "<memory_context>" in prompt_text
        and "0.98" not in prompt_text
        and "sqlite" not in prompt_text
    )
    return scenario_report(
        name="memory_results_live",
        ok=ok,
        status=result.status.value,
        api_calls=provider.api_call_count,
        steps=result.steps,
        expected=f"MEMORY_RESULTS_OK: {token}",
        actual=final_text,
        evidence={
            "memory_context_in_request": "<memory_context>" in prompt_text,
            "score_or_source_leaked": "0.98" in prompt_text or "sqlite" in prompt_text,
            "request_snapshots": provider.request_snapshots(),
        },
        error_code="" if ok else _runtime_error_code(result),
        error_message=(
            "" if ok else _runtime_error_message(result, "memory_results 验证失败")
        ),
    )


async def run_memory_query_live(work_dir: Path, retries: int) -> ScenarioReport:
    """验证真实 MemoryService + memory_query 召回会进入请求并影响回答。"""
    del retries
    token = "MEMORY_QUERY_TOKEN_0708"
    scope = MemoryScope(workspace_id="deepseek-flow", agent_id="memory-query-live")
    service = MemoryService(SQLiteMemoryStore(work_dir / "memory.db", use_fts=False))
    service.remember(
        MemoryWriteInput(
            scope=scope,
            text=f"用户需要在回答中包含 {token}",
            reason="DeepSeek live flow 验证 memory_query",
            category=MemoryCategory.USER,
            kind=MemoryItemKind.PREFERENCE,
        )
    )
    agent_path = prepare_text_agent(work_dir, "memory-query-live")
    provider = recording_provider()
    runtime = RuntimeFactory.from_config_path(
        agent_path,
        provider=provider,
        memory_service=service,
    )
    result = await runtime.run_loop(
        f"根据记忆回答，只输出 MEMORY_QUERY_OK: {token}",
        options=RuntimeOptions(
            session_id="memory-query-live",
            memory_query=MemoryQuery(scope=scope, text=token, limit=3),
            memory_max_chars=300,
            metadata={"scenario": "memory_query_live"},
        ),
    )
    final_text = (
        result.assistant_message.text.strip() if result.assistant_message else ""
    )
    prompt_text = "\n".join(message.text for message in provider.requests[0].messages)
    ok = (
        result.status.value == "ok"
        and "MEMORY_QUERY_OK" in final_text
        and token in final_text
        and "<memory_context>" in prompt_text
        and token in prompt_text
    )
    return scenario_report(
        name="memory_query_live",
        ok=ok,
        status=result.status.value,
        api_calls=provider.api_call_count,
        steps=result.steps,
        expected=f"MEMORY_QUERY_OK: {token}",
        actual=final_text,
        evidence={
            "memory_context_in_request": "<memory_context>" in prompt_text,
            "request_snapshots": provider.request_snapshots(),
        },
        error_code="" if ok else _runtime_error_code(result),
        error_message=(
            "" if ok else _runtime_error_message(result, "memory_query 验证失败")
        ),
    )


def _memory_result(*, token: str, item_id: str) -> MemorySearchResult:
    """构造显式 memory 结果。"""
    item = MemoryItem(
        id=item_id,
        scope=MemoryScope(workspace_id="deepseek-flow", agent_id="memory-results-live"),
        text=f"用户要求回答中包含 {token}",
        category=MemoryCategory.USER,
        kind=MemoryItemKind.PREFERENCE,
        level=MemoryLevel.SEMANTIC,
    )
    return MemorySearchResult(
        item=item,
        score=0.98,
        source="sqlite",
        matched_text=item.text,
    )
