"""一次模型调用整理项目经验，资格与项目锁由宿主提供。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..exceptions import IrisEvolutionError, IrisSkillError, IrisTemplateError
from ..message import LLMRequest, LLMResponse, Msg
from ..prompts import PromptSource
from ..providers.protocols import CompletionProvider
from ..skill.frontmatter import split_frontmatter
from ..utils.files import atomic_write_text
from ..utils.generation_worker import check_generation_cancelled, generation_worker
from .config import EvolutionConfig
from .materials import EvolutionMaterialStore
from .models import (
    EvolutionMaintenanceScope,
    EvolutionMaterial,
    EvolutionRange,
    EvolutionResult,
    EvolutionSource,
)

ResultT = TypeVar("ResultT")
_FRONTMATTER = """---
name: project-experience
description: 本项目实际经历提炼的工作约定、适用条件与可复用方法。
---

"""
_EFFECT = (
    "首次生成由新 runner 发现；已登记 Skill 在下一次 load_skill 时读取新正文，旧消息保持原样。"
)
_INSTRUCTIONS = """仅返回 body 和 reason。
body 为完整 Markdown 正文，不含 frontmatter；null 表示 no-change。
reason 简要说明更新或保持原因。模型不能选择文件路径、改写其它目标或决定材料消费位置。
正文不得超过输入的 skill_max_chars；完整文件须能由现有 Skill loader 读取。
仅返回符合以下 JSON Schema 的 JSON，不附加文字或代码围栏。"""


class _SkillResponse(BaseModel):
    """模型只决定正文或 no-change，程序拥有发布目标和固定 frontmatter。"""

    model_config = ConfigDict(extra="forbid")
    body: str | None = Field(pattern=r"\S")
    reason: str = Field(pattern=r"\S")


def _check_cancelled() -> None:
    """拒绝已经撤销的模型结果，保留共享 worker 的取消语义。"""
    check_generation_cancelled()
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


class EvolutionService:
    """宿主绑定的项目 A 阶段服务，不拥有 timer、项目锁或业务 Run。"""

    def __init__(
        self,
        *,
        workspace_root: Path,
        skill_path: Path,
        store: EvolutionMaterialStore,
        provider: CompletionProvider,
        model: str,
        config: EvolutionConfig,
        prompt_source: PromptSource,
    ) -> None:
        """绑定装配层已经选定的唯一项目与 Skill 目标。"""
        self.workspace_root = workspace_root
        self.skill_path = skill_path
        self.store = store
        self.provider = provider
        self.model = model
        self.config = config
        self.prompt_source = prompt_source
        self._io_tasks: set[asyncio.Task[object]] = set()

    async def run_async_io(
        self, operation: Callable[[], ResultT], *, complete_on_cancel: bool = False
    ) -> ResultT:
        """捕获使用普通线程，后台维护借用本类任务绑定的独立 worker。"""
        worker = generation_worker.get()
        task = asyncio.create_task(
            asyncio.to_thread(operation) if worker is None else worker.run(operation)
        )
        self._io_tasks.add(task)
        task.add_done_callback(self._finish_io)
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if not complete_on_cancel or task.cancelled():
                    raise

    def _finish_io(self, task: asyncio.Task[object]) -> None:
        self._io_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def wait_pending_io(self) -> None:
        """等待已经派发的同步工作真正结束。"""
        while self._io_tasks:
            await asyncio.gather(*tuple(self._io_tasks), return_exceptions=True)

    async def alist_pending_sources(self) -> tuple[EvolutionSource, ...]:
        """列出已捕获完整且未消费的来源，由宿主再判断生命周期资格。"""
        return await self.run_async_io(self.store.list_pending_sources)

    def _read_skill(self) -> str | None:
        try:
            return self.skill_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def _prepare(
        self, materials: tuple[EvolutionMaterial, ...]
    ) -> tuple[str | None, tuple[EvolutionMaterial, ...], LLMRequest]:
        """固定本轮策略与文件基线，按完整消息选择实际请求的预算内前缀。"""
        try:
            snapshot = self.prompt_source.snapshot()
            strategy = snapshot.render("project_skill_update", {})
            policy = (
                (self.workspace_root / self.config.policy_skill).read_text(encoding="utf-8")
                if self.config.policy_skill is not None
                else files("iris.evolution")
                .joinpath("self-evolution/SKILL.md")
                .read_text(encoding="utf-8")
            )
            policy_body = split_frontmatter(policy)[1]
            baseline = self._read_skill()
            current_body = split_frontmatter(baseline)[1] if baseline is not None else ""
        except (OSError, UnicodeError, IrisTemplateError, IrisSkillError) as exc:
            raise IrisEvolutionError("项目经验策略或当前 Skill 读取失败", error=str(exc)) from exc
        schema = json.dumps(_SkillResponse.model_json_schema(), ensure_ascii=False)
        prompt = (
            f"{strategy}\n\n策略 Skill：\n{policy_body}\n\n"
            f"固定输出契约：\n{_INSTRUCTIONS}\n{schema}"
        )
        selected: list[EvolutionMaterial] = []
        request = self._request(prompt, current_body, ())
        for material in materials:
            check_generation_cancelled()
            candidate = self._request(prompt, current_body, (*selected, material))
            if self.provider.estimate_input_tokens(candidate) > self.config.input_budget_tokens:
                if not selected:
                    raise IrisEvolutionError("项目经验输入预算不足以容纳完整消息与策略")
                break
            selected.append(material)
            request = candidate
        return baseline, tuple(selected), request

    def _request(
        self, prompt: str, current_body: str, selected: tuple[EvolutionMaterial, ...]
    ) -> LLMRequest:
        return LLMRequest(
            model=self.model,
            messages=[
                Msg.system(prompt),
                Msg.user(
                    json.dumps(
                        {
                            "current_skill": current_body,
                            "skill_max_chars": self.config.skill_max_chars,
                            "materials": [item.model_dump(mode="json") for item in selected],
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
            max_tokens=self.config.output_budget_tokens,
            temperature=0,
            response_format="json_object",
        )

    def _parse(self, response: LLMResponse) -> tuple[_SkillResponse, str | None]:
        """只在模型输出边界解析 JSON 与发布文件额度。"""
        if response.finish_reason != "stop":
            raise IrisEvolutionError("项目经验响应未完整结束", finish_reason=response.finish_reason)
        try:
            parsed = _SkillResponse.model_validate_json(response.to_msg().text)
        except ValidationError as exc:
            raise IrisEvolutionError("项目经验输出不符合正文与原因契约") from exc
        if parsed.body is None:
            return parsed, None
        body = parsed.body.strip()
        content = f"{_FRONTMATTER}{body}\n"
        if (
            len(body) > self.config.skill_max_chars
            or len(content) > 50000
            or len(content.splitlines()) > 1000
        ):
            raise IrisEvolutionError("项目经验输出超过 Skill 正文或完整加载额度")
        return parsed, content

    def _commit(
        self,
        selected: tuple[EvolutionMaterial, ...],
        baseline: str | None,
        content: str | None,
        result: EvolutionResult,
    ) -> EvolutionResult:
        """短同步发布与确认消费；已写文件不因后续进度失败回滚。"""
        check_generation_cancelled()
        try:
            if self._read_skill() != baseline:
                conflict = EvolutionResult(
                    status="conflict", reason="项目经验文件在生成期间已被修改", usage=result.usage
                )
                self.store.record_step(conflict)
                return conflict
            if content is not None:
                atomic_write_text(self.skill_path, content)
        except (OSError, UnicodeError) as exc:
            raise IrisEvolutionError(
                "项目经验文件发布失败", path=str(self.skill_path), error=str(exc)
            ) from exc
        self.store.consume(selected, result)
        return result

    async def maintain_cycle(self, *, scope: EvolutionMaintenanceScope) -> EvolutionResult:
        """在宿主项目锁内整理一批新材料，最多调用模型一次。"""
        usage: dict[str, int] = {}
        recorded = False
        try:
            pending = await self.run_async_io(
                lambda: self.store.read_pending(allowed_sources=scope.allowed_sources)
            )
            if not pending.items:
                return EvolutionResult(status="empty")
            baseline, selected, request = await self.run_async_io(
                lambda: self._prepare(pending.items)
            )
            sources = tuple(dict.fromkeys(item.source for item in selected))
            _check_cancelled()
            if not await scope.check(sources):
                raise asyncio.CancelledError
            if any(item.records for item in selected):
                response = await self.provider.complete(request)
                usage = {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response.total_tokens,
                }
                _check_cancelled()
                parsed, content = await self.run_async_io(lambda: self._parse(response))
            else:
                parsed, content = _SkillResponse(body=None, reason="本批仅含过滤后的空区间"), None
            _check_cancelled()
            if not await scope.check(sources):
                raise asyncio.CancelledError
            result = EvolutionResult(
                status="updated" if content is not None else "no_change",
                reason=parsed.reason,
                consumed_ranges=tuple(
                    EvolutionRange.model_construct(
                        source=item.source,
                        start_message_count=item.start_message_count,
                        end_message_count=item.end_message_count,
                    )
                    for item in selected
                ),
                usage=usage,
                has_more=pending.has_more or len(selected) < len(pending.items),
                effect=_EFFECT if content is not None else "项目 Skill 保持原样。",
            )
            result = await self.run_async_io(
                lambda: self._commit(selected, baseline, content, result), complete_on_cancel=True
            )
            recorded = True
            _check_cancelled()
            return result
        except (Exception, asyncio.CancelledError) as exc:
            if not recorded:
                failure = EvolutionResult(
                    status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                    reason=str(exc) or type(exc).__name__,
                    usage=usage,
                )
                await self.run_async_io(
                    lambda: self.store.record_step(failure), complete_on_cancel=True
                )
            raise
