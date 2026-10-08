"""一次模型调用整理项目经验，资格与项目锁由宿主提供。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..exceptions import IrisEvolutionError, IrisSkillError, IrisTemplateError
from ..message import LLMRequest, LLMResponse, Msg
from ..observability.facts import record_source_adoption
from ..observability.provider import observe_provider
from ..observability.service import Observability
from ..prompts import PromptSnapshot, PromptSource
from ..providers.protocols import CompletionProvider
from ..skill.frontmatter import split_frontmatter
from ..utils.background_io import BackgroundIO
from ..utils.files import atomic_write_text
from ..utils.generation_worker import check_generation_cancelled
from ..utils.sources import SourceDocument
from ._publication import PublicationJournal
from .config import EvolutionConfig
from .history import (
    EvolutionHistoryCursor,
    PublicationDocument,
    PublicationHistoryEntry,
    PublicationPage,
    PublicationRecord,
    RevisionRequestPage,
)
from .materials import EvolutionMaterialStore
from .models import (
    EvolutionLearningReadiness,
    EvolutionMaintenanceScope,
    EvolutionMaterial,
    EvolutionRange,
    EvolutionResult,
    EvolutionSession,
    EvolutionSource,
    ExperienceOrigin,
    HostOrigin,
    RevisionEvidence,
    RevisionItem,
    RevisionRequest,
    RevisionTarget,
)
from .revision import (
    ConfigTarget,
    PreparedRevision,
    PromptTarget,
    RevisionContext,
    check_targets,
    prepare_candidate,
    prepare_revision,
    publish_revision,
    revision_response_schema,
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
_INSTRUCTIONS = """返回 body、reason 和可选 issue。
body 为完整 Markdown 正文，不含 frontmatter；null 表示 no-change。
reason 简要说明更新或保持原因。模型不能选择文件路径、改写其它目标或决定材料消费位置。
正文不得超过输入的 skill_max_chars；完整文件须能由现有 Skill loader 读取。
仅在本批材料揭示具体机制问题时提出 issue；普通事实缺失或单次失败不要求修订。
issue.targets 只能选输入中的开放目标；evidence 的 ref 必须来自本批材料，quote 必须逐字引用原文。
引用全部支持该问题的来源，必要片段随问题保留；没有具体问题或没有开放目标时 issue 为 null。
仅返回符合以下 JSON Schema 的 JSON，不附加文字或代码围栏。"""


class _IssueProposal(BaseModel):
    """A 只提出有本批原文依据的开放目标问题。"""

    model_config = ConfigDict(extra="forbid")
    description: str = Field(pattern=r"\S")
    targets: tuple[RevisionTarget, ...] = Field(min_length=1)
    evidence: tuple[RevisionEvidence, ...] = Field(min_length=1)


class _SkillResponse(BaseModel):
    """模型只决定正文或 no-change，程序拥有发布目标和固定 frontmatter。"""

    model_config = ConfigDict(extra="forbid")
    body: str | None = Field(pattern=r"\S")
    reason: str = Field(pattern=r"\S")
    issue: _IssueProposal | None = None


def _skill_contract() -> str:
    """真实 A 请求与候选说明共享一份输出协议。"""
    return f"{_INSTRUCTIONS}\n{json.dumps(_SkillResponse.model_json_schema(), ensure_ascii=False)}"


def project_skill_prompt_description() -> tuple[str, dict[str, Any]]:
    """提供与实际 A 请求同源的变量说明、固定契约和代表变量。"""
    return (
        "无模板变量；当前 Skill、真实材料、开放目标与正文额度由独立 user 消息提供。\n"
        + _skill_contract(),
        {},
    )


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
        prompt_targets: tuple[PromptTarget, ...] = (),
        config_target: ConfigTarget | None = None,
        observability: Observability | None = None,
    ) -> None:
        """绑定装配层已经选定的唯一项目与 Skill 目标。"""
        self.workspace_root = workspace_root
        self.skill_path = skill_path
        self.store = store
        self.observability = observability if observability is not None else Observability()
        self.provider = observe_provider(provider, self.observability)
        self.model = model
        self.config = config
        self.prompt_source = prompt_source
        self.prompt_targets = prompt_targets
        self.config_target = config_target
        self._allowed_targets = frozenset(
            [("prompt", name) for name in config.prompt_targets]
            + [("config", name) for name in config.config_targets]
        )
        self._background_io = BackgroundIO()
        self._publications = PublicationJournal(store)

    def list_publications(
        self, *, after: EvolutionHistoryCursor | None = None, limit: int = 50
    ) -> PublicationPage:
        """读取发布历史摘要，完整正文通过 get_publication 按 ID 查询。"""
        return self.store.list_publications(after=after, limit=limit)

    async def alist_publications(
        self, *, after: EvolutionHistoryCursor | None = None, limit: int = 50
    ) -> PublicationPage:
        """异步读取项目发布档案页。"""
        return await self.run_async_io(lambda: self.list_publications(after=after, limit=limit))

    def get_publication(self, publication_id: str) -> PublicationHistoryEntry | None:
        """读取完整或过期的发布历史，不重读当前目标来替代原正文。"""
        return self.store.get_publication(publication_id)

    async def aget_publication(self, publication_id: str) -> PublicationHistoryEntry | None:
        """异步读取原始 ID 对应的发布记录。"""
        return await self.run_async_io(lambda: self.get_publication(publication_id))

    def list_revision_requests(
        self, *, after: EvolutionHistoryCursor | None = None, limit: int = 50
    ) -> RevisionRequestPage:
        """读取请求历史摘要，包含已结算请求；完整证据由详情接口返回。"""
        return self.store.list_revision_requests(after=after, limit=limit)

    async def alist_revision_requests(
        self, *, after: EvolutionHistoryCursor | None = None, limit: int = 50
    ) -> RevisionRequestPage:
        """异步读取请求历史页。"""
        return await self.run_async_io(
            lambda: self.list_revision_requests(after=after, limit=limit)
        )

    def get_revision_request(self, revision_id: str) -> RevisionItem | None:
        """按 ID 读取完整修订请求及证据，不改变处理状态。"""
        return self.store.get_revision_request(revision_id)

    async def aget_revision_request(self, revision_id: str) -> RevisionItem | None:
        """在线程中读取完整修订请求及证据。"""
        return await self.run_async_io(lambda: self.get_revision_request(revision_id))

    async def run_async_io(
        self, operation: Callable[[], ResultT], *, complete_on_cancel: bool = False
    ) -> ResultT:
        """捕获使用普通线程，后台维护借用本类任务绑定的独立 worker。"""
        return await self._background_io.run(operation, complete_on_cancel=complete_on_cancel)

    async def wait_pending_io(self) -> None:
        """等待已经派发的同步工作真正结束。"""
        await self._background_io.wait_pending()

    async def alist_pending_sources(self) -> tuple[EvolutionSource, ...]:
        """列出已捕获完整且未消费的来源，由宿主再判断生命周期资格。"""
        return await self.run_async_io(self.store.list_pending_sources)

    async def alist_pending_sessions(self) -> tuple[EvolutionSession, ...]:
        """提供宿主请求的会话归属，由 harness 判定 WAITING 与 reader 资格。"""
        return await self.run_async_io(self.store.list_pending_sessions)

    async def aread_learning_readiness(self) -> EvolutionLearningReadiness:
        """异步读取原文就绪短状态，不触发准入或生成。"""
        return await self.run_async_io(self.store.read_learning_readiness)

    async def ahas_pending_recovery(self) -> bool:
        """同时保留原 owner 的进程内收据与数据库未收尾事实。"""
        return await self.run_async_io(self._publications.has_pending)

    async def ahas_pending_revisions(self, *, scope: EvolutionMaintenanceScope) -> bool:
        """按当前目标和宿主资格查询 B 候选，不读取请求证据。"""
        return await self.run_async_io(
            lambda: self.store.has_pending_revisions(
                allowed_sources=scope.allowed_sources,
                allowed_sessions=scope.allowed_sessions,
                allowed_targets=self._allowed_targets,
            )
        )

    async def aadmit_learning_sources(
        self, *, allowed_sources: frozenset[tuple[str, str]], threshold: int
    ) -> EvolutionLearningReadiness:
        """异步提交宿主当前合格范围的原文准入。"""
        return await self.run_async_io(
            lambda: self.store.admit_learning_sources(
                allowed_sources=allowed_sources, threshold=threshold
            ),
            complete_on_cancel=True,
        )

    async def enqueue_revision(self, request: RevisionRequest) -> RevisionItem:
        """在宿主入口校验开放目标，完整保存显式请求而不伪造经历。"""
        check_targets(request.targets, self.config)
        item = RevisionItem.model_construct(
            description=request.description,
            targets=request.targets,
            origin=HostOrigin.model_construct(session=request.session),
        )
        await self.run_async_io(lambda: self.store.enqueue_revision(item), complete_on_cancel=True)
        return item

    def _prompt(self, prompt_id: str, contract: str) -> tuple[PromptSnapshot, str]:
        """每次 A/B 获锁后固定项目模板与必加载的策略 Skill。"""
        try:
            snapshot = self.prompt_source.snapshot()
            strategy = snapshot.render(prompt_id, {})
            policy = (
                (self.workspace_root / self.config.policy_skill).read_text(encoding="utf-8")
                if self.config.policy_skill is not None
                else files("iris.evolution")
                .joinpath("self-evolution/SKILL.md")
                .read_text(encoding="utf-8")
            )
            policy_body = split_frontmatter(policy)[1]
        except (OSError, UnicodeError, IrisTemplateError, IrisSkillError) as exc:
            raise IrisEvolutionError("项目修订策略读取失败", error=str(exc)) from exc
        record_source_adoption(
            owner_kind="evolution",
            source_kind="prompt_snapshot",
            boundary=prompt_id,
            documents=(
                *snapshot.source_documents(),
                SourceDocument(
                    "evolution_policy",
                    str(self.workspace_root / self.config.policy_skill)
                    if self.config.policy_skill is not None
                    else "iris.evolution/self-evolution/SKILL.md",
                    policy,
                ),
            ),
        )
        return snapshot, f"{strategy}\n\n策略 Skill：\n{policy_body}\n\n固定输出契约：\n{contract}"

    def _read_skill(self) -> str | None:
        try:
            return self.skill_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def _prepare(
        self, materials: tuple[EvolutionMaterial, ...]
    ) -> tuple[str | None, tuple[EvolutionMaterial, ...], LLMRequest]:
        """固定本轮策略与文件基线，按完整消息选择实际请求的预算内前缀。"""
        _, prompt = self._prompt("project_skill_update", _skill_contract())
        try:
            baseline = self._read_skill()
            current_body = split_frontmatter(baseline)[1] if baseline is not None else ""
        except (OSError, UnicodeError, IrisSkillError) as exc:
            raise IrisEvolutionError("项目经验策略或当前 Skill 读取失败", error=str(exc)) from exc
        prefix = (
            json.dumps(
                {"current_skill": current_body, "skill_max_chars": self.config.skill_max_chars},
                ensure_ascii=False,
            )[:-1]
            + ', "materials": ['
        )
        suffix = (
            '], "open_targets": '
            + json.dumps(
                {"prompt": self.config.prompt_targets, "config": self.config.config_targets},
                ensure_ascii=False,
            )
            + "}"
        )
        selected: list[EvolutionMaterial] = []
        serialized: list[str] = []
        request = self._request(prompt, prefix + suffix)
        for material in materials:
            check_generation_cancelled()
            serialized.append(json.dumps(material.model_dump(mode="json"), ensure_ascii=False))
            candidate = self._request(prompt, prefix + ", ".join(serialized) + suffix)
            if self.provider.estimate_input_tokens(candidate) > self.config.input_budget_tokens:
                if not selected:
                    raise IrisEvolutionError("项目经验输入预算不足以容纳完整消息与策略")
                break
            selected.append(material)
            request = candidate
        return baseline, tuple(selected), request

    def _request(self, prompt: str, content: str) -> LLMRequest:
        return LLMRequest(
            model=self.model,
            messages=[Msg.system(prompt), Msg.user(content)],
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

    def _bind_issue(
        self, proposal: _IssueProposal | None, selected: tuple[EvolutionMaterial, ...]
    ) -> RevisionItem | None:
        """输出边界一次核对原文引用并绑定实际来源，不保留整批材料。"""
        if proposal is None:
            return None
        check_targets(proposal.targets, self.config)
        records = {
            record.ref: (item.source, record) for item in selected for record in item.records
        }
        sources: list[EvolutionSource] = []
        for evidence in proposal.evidence:
            current = records.get(evidence.ref)
            if current is None or evidence.quote not in current[1].text:
                raise IrisEvolutionError(
                    "项目修订问题引用了本批以外或不匹配的原文", ref=evidence.ref
                )
            if current[0] not in sources:
                sources.append(current[0])
        return RevisionItem.model_construct(
            description=proposal.description,
            targets=proposal.targets,
            evidence=proposal.evidence,
            origin=ExperienceOrigin.model_construct(sources=tuple(sources)),
        )

    def _commit(
        self,
        baseline: str | None,
        content: str | None,
        result: EvolutionResult,
        publication: PublicationRecord,
    ) -> EvolutionResult:
        """短同步发布与确认消费；已写文件不因后续进度失败回滚。"""
        check_generation_cancelled()
        self._publications.begin(publication)
        try:
            if self._read_skill() != baseline:
                conflict = EvolutionResult(
                    status="conflict",
                    reason="项目经验文件在生成期间已被修改",
                    usage=result.usage,
                    publication_id=publication.publication_id,
                )
                return self._publications.complete(publication, conflict)
            if content is not None:
                atomic_write_text(self.skill_path, content)
        except (OSError, UnicodeError) as exc:
            raise IrisEvolutionError(
                "项目经验文件发布失败", path=str(self.skill_path), error=str(exc)
            ) from exc
        return self._publications.complete(publication, result)

    async def maintain_cycle(self, *, scope: EvolutionMaintenanceScope) -> EvolutionResult:
        """每次项目锁只执行一项 A 或 B；B 成功或失败均不重新消费 A。"""
        resumed = await self.run_async_io(self._publications.resume, complete_on_cancel=True)
        if resumed is not None:
            has_more = (
                await self.run_async_io(lambda: self._remaining(scope))
                if resumed.status in {"updated", "no_change"}
                else False
            )
            self.observability.maintenance_result(
                "evolution", resumed.stage, resumed.status, revision_id=resumed.revision_id
            )
            return resumed.model_copy(update={"has_more": has_more})
        if not scope.experience_only:
            revisions = await self.run_async_io(
                lambda: self.store.read_pending_revisions(
                    allowed_sources=scope.allowed_sources,
                    allowed_sessions=scope.allowed_sessions,
                    allowed_targets=self._allowed_targets,
                    requested_revision_id=scope.requested_revision_id,
                )
            )
            if revisions:
                result = await self._maintain_revision(revisions[0], scope)
                self.observability.maintenance_result(
                    "evolution", result.stage, result.status, revision_id=result.revision_id
                )
                return result
        return await self._maintain_experience(scope)

    async def _maintain_experience(self, scope: EvolutionMaintenanceScope) -> EvolutionResult:
        """整理一批 A，问题与本次消费同一进度提交后结束本轮。"""
        usage: dict[str, int] = {}
        recorded = False
        publication = PublicationRecord(
            stage="experience", origin="experience", description="从本批真实材料整理项目经验。"
        )
        try:
            readiness = await self.aread_learning_readiness()
            empty_sources = frozenset(
                (item.source.lifecycle_source_id, item.source.run_id)
                for item in readiness.sources
                if item.complete
                and not item.has_content
                and (item.source.lifecycle_source_id, item.source.run_id)
                in scope.experience_sources
            )
            pending = await self.run_async_io(
                lambda: self.store.read_pending(
                    allowed_sources=empty_sources or scope.experience_sources
                )
            )
            if not pending.items:
                self.observability.maintenance_result("evolution", "experience", "empty")
                return EvolutionResult(status="empty")
            if not any(record.text.strip() for item in pending.items for record in item.records):
                publication = publication.model_copy(update={"materials": pending.items})
                sources = tuple(dict.fromkeys(item.source for item in pending.items))
                _check_cancelled()
                if not await scope.check(sources):
                    raise asyncio.CancelledError
                result = EvolutionResult(
                    publication_id=publication.publication_id,
                    status="no_change",
                    reason="本批仅含过滤后的空区间",
                    effect="项目 Skill 保持原样。",
                )
                result = await self.run_async_io(
                    lambda: self._publications.complete(publication, result),
                    complete_on_cancel=True,
                )
                recorded = True
                self.observability.maintenance_result("evolution", result.stage, result.status)
                _check_cancelled()
                has_more = await self.run_async_io(lambda: self._remaining(scope))
                return result.model_copy(update={"has_more": has_more})
            baseline, selected, request = await self.run_async_io(
                lambda: self._prepare(pending.items)
            )
            publication = publication.model_copy(
                update={
                    "before_documents": (
                        PublicationDocument(path=str(self.skill_path), text=baseline),
                    ),
                    "materials": selected,
                }
            )
            sources = tuple(dict.fromkeys(item.source for item in selected))
            _check_cancelled()
            if not await scope.check(sources):
                raise asyncio.CancelledError
            if any(record.text.strip() for item in selected for record in item.records):
                with self.observability.bind({"iris.model.purpose": "evolution_experience"}):
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
            issue = self._bind_issue(parsed.issue, selected)
            _check_cancelled()
            if not await scope.check(sources):
                raise asyncio.CancelledError
            result = EvolutionResult(
                publication_id=publication.publication_id,
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
                has_more=issue is not None
                or pending.has_more
                or len(selected) < len(pending.items),
                effect=_EFFECT if content is not None else "项目 Skill 保持原样。",
            )
            publication = publication.model_copy(
                update={
                    "candidate_documents": (
                        PublicationDocument(path=str(self.skill_path), text=content),
                    )
                    if content is not None
                    else (),
                    "proposed_issue": issue,
                    "usage": usage,
                    "reason": result.reason,
                    "effect": result.effect,
                }
            )
            result = await self.run_async_io(
                lambda: self._commit(baseline, content, result, publication),
                complete_on_cancel=True,
            )
            recorded = True
            self.observability.maintenance_result("evolution", result.stage, result.status)
            _check_cancelled()
            return result
        except (Exception, asyncio.CancelledError) as exc:
            if not recorded:
                failure = EvolutionResult(
                    publication_id=publication.publication_id,
                    status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                    reason=str(exc) or type(exc).__name__,
                    usage=usage,
                )
                await self.run_async_io(
                    lambda: self._publications.failure(publication, failure),
                    complete_on_cancel=True,
                )
                self.observability.maintenance_result("evolution", failure.stage, failure.status)
            raise

    def _prepare_review(self, item: RevisionItem) -> tuple[RevisionContext, LLMRequest]:
        """B 重读当前文件与领域说明，不把当前磁盘值当作历史采用值。"""
        contract = (
            "只返回本次目标的一项 prompt 修订、一组 config 赋值或 no-change，并说明原因。\n"
            "模型不返回文件路径，不扩大开放范围；只返回符合以下 JSON Schema 的 JSON。\n"
            + json.dumps(revision_response_schema(), ensure_ascii=False)
        )
        snapshot, prompt = self._prompt("evolution_review", contract)
        context = prepare_revision(
            prompt_snapshot=snapshot,
            targets=item.targets,
            prompt_targets=self.prompt_targets,
            config_target=self.config_target,
        )
        payload = {
            "issue": item.model_dump(mode="json"),
            "targets": context.model_input,
        }
        request = LLMRequest(
            model=self.model,
            messages=[Msg.system(prompt), Msg.user(json.dumps(payload, ensure_ascii=False))],
            max_tokens=self.config.output_budget_tokens,
            temperature=0,
            response_format="json_object",
        )
        if self.provider.estimate_input_tokens(request) > self.config.input_budget_tokens:
            raise IrisEvolutionError("策略修订完整输入超过预算")
        return context, request

    async def _revision_eligible(
        self, item: RevisionItem, scope: EvolutionMaintenanceScope
    ) -> bool:
        """两类身份分别交给 harness 的唯一资格检查。"""
        if item.origin.kind == "experience":
            return await scope.check(item.origin.sources)
        return await scope.check_session(item.origin.session)

    def _commit_revision(
        self,
        item: RevisionItem,
        candidate: PreparedRevision,
        usage: dict[str, int],
        publication: PublicationRecord,
    ) -> EvolutionResult:
        """短 IO 发布候选并独立结算 B，不重新提交 A 进度。"""
        self._publications.begin(publication)
        published = publish_revision(candidate)
        result = EvolutionResult(
            stage="revision",
            revision_id=item.id,
            publication_id=publication.publication_id,
            targets=candidate.targets,
            status=("no_change" if candidate.action == "no_change" else "updated")
            if published
            else "conflict",
            reason=candidate.reason if published else "目标文件在生成期间已被修改",
            usage=usage,
            effect=candidate.effect if published else "原文件与待处理问题保持原样。",
        )
        return self._publications.complete(publication, result)

    def _remaining(self, scope: EvolutionMaintenanceScope) -> bool:
        """仅把本轮合格范围内的剩余 A/B 交回调度器。"""
        return self.store.has_pending_revisions(
            allowed_sources=scope.allowed_sources,
            allowed_sessions=scope.allowed_sessions,
            allowed_targets=self._allowed_targets,
        ) or self.store.has_pending_materials(allowed_sources=scope.experience_sources)

    async def _maintain_revision(
        self, item: RevisionItem, scope: EvolutionMaintenanceScope
    ) -> EvolutionResult:
        """处理单个 B；失败或取消带本请求身份返回，保留 pending。"""
        usage: dict[str, int] = {}
        committed: EvolutionResult | None = None
        publication = PublicationRecord(
            stage="revision",
            revision_id=item.id,
            origin="experience" if item.origin.kind == "experience" else "host_request",
            description=item.description,
            targets=item.targets,
        )
        try:
            context, request = await self.run_async_io(lambda: self._prepare_review(item))
            publication = publication.model_copy(
                update={
                    "before_documents": tuple(
                        PublicationDocument(path=str(path), text=text)
                        for path, text in context.baselines.items()
                    ),
                }
            )
            _check_cancelled()
            if not await self._revision_eligible(item, scope):
                raise asyncio.CancelledError
            with self.observability.bind({"iris.model.purpose": "evolution_revision"}):
                response = await self.provider.complete(request)
            usage = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
            }
            _check_cancelled()
            candidate = await self.run_async_io(lambda: prepare_candidate(response, context))
            publication = publication.model_copy(
                update={
                    "candidate_documents": (
                        (PublicationDocument(path=str(candidate.path), text=candidate.content),)
                        if candidate.path is not None
                        else ()
                    ),
                    "usage": usage,
                    "reason": candidate.reason,
                    "effect": candidate.effect,
                }
            )
            _check_cancelled()
            if not await self._revision_eligible(item, scope):
                raise asyncio.CancelledError
            committed = await self.run_async_io(
                lambda: self._commit_revision(item, candidate, usage, publication),
                complete_on_cancel=True,
            )
            if committed.status == "conflict":
                return committed
            _check_cancelled()
            has_more = await self.run_async_io(lambda: self._remaining(scope))
            return committed.model_copy(update={"has_more": has_more})
        except (Exception, asyncio.CancelledError) as exc:
            if committed is not None:
                return committed
            failure = EvolutionResult(
                stage="revision",
                revision_id=item.id,
                publication_id=publication.publication_id,
                targets=item.targets,
                status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                reason=str(exc) or type(exc).__name__,
                usage=usage,
            )
            await self.run_async_io(
                lambda: self._publications.failure(publication, failure), complete_on_cancel=True
            )
            return failure
