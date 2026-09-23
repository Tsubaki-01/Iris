"""将不可变经历提炼为观察，再把观察整理为正式知识的无工具生成流程。"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..exceptions import IrisMemoryError
from ..message import LLMRequest, LLMResponse, Msg
from ..providers.protocols import CompletionProvider
from .generation_models import (
    DreamOperation,
    DreamPlan,
    DreamSnapshot,
    EpisodeProgress,
    EpisodeSlice,
    FlushCommit,
    GenerationResult,
    MemoryConsumedRange,
    MemoryGenerationConfig,
    ObservationResolution,
)
from .models import (
    MemoryCategory,
    MemoryEvidenceRef,
    MemoryItemKind,
    MemoryObservation,
    _new_id,
)

if TYPE_CHECKING:
    from .service import MemoryService


class _ExtractedObservation(BaseModel):
    """模型只能引用本批提供的证据标签，持久身份由程序绑定。"""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(pattern=r"\S")
    applicability: str = ""
    category: MemoryCategory = MemoryCategory.USER
    kind: MemoryItemKind = MemoryItemKind.NOTE
    reason: str = Field(pattern=r"\S")
    evidence: tuple[str, ...] = Field(min_length=1)
    target_item_ids: tuple[str, ...] = ()


class _FlushResponse(BaseModel):
    """一次 flush 的完整模型输出，空观察列表也表示已读完本批。"""

    model_config = ConfigDict(extra="forbid")
    observations: tuple[_ExtractedObservation, ...]


class _ProposedOperation(BaseModel):
    """模型修改计划的外部边界；new_key 仅是本响应内的引用标签。"""

    model_config = ConfigDict(extra="forbid")
    action: Literal["add", "update", "merge", "delete", "support"]
    target_id: str | None = None
    new_key: str | None = None
    text: str | None = Field(default=None, pattern=r"\S")
    category: MemoryCategory | None = None
    kind: MemoryItemKind | None = None
    evidence: tuple[str, ...] = ()
    merge_ids: tuple[str, ...] = ()
    reason: str = Field(pattern=r"\S")


class _Resolution(BaseModel):
    """观察归入已有 ID、本响应新增标签，或以原因解释忽略。"""

    model_config = ConfigDict(extra="forbid")
    observation_id: str
    target_id: str | None = None
    reason: str = Field(pattern=r"\S")


class _DreamResponse(BaseModel):
    """一次完整 dreaming 响应。"""

    model_config = ConfigDict(extra="forbid")
    operations: tuple[_ProposedOperation, ...]
    resolutions: tuple[_Resolution, ...]


_FLUSH_PROMPT = """从本批经历中选择对未来任务有用的信息，忠实压缩为少量观察。
只返回符合 schema 的 JSON。
优先保留明确偏好与纠正、项目约定、值得参考的具体经历，以及能跨会话接续的具体任务线索。
任务线索如外部任务编号、具体阻塞和待验证改动；常规读取、正在整理等执行进度由会话历史保存，不进入记忆。
普通闲聊、常规操作流水、重复转述和无后续价值的临时要求可省略；不要把“用户要求记录”本身另记一条。
同一主题的有用信息集中表达，不按每句话或发言角色拆条；没有值得保留的内容时返回 observations: []。
text 直接写值得记住的要点及防止误用的必要条件，不复述完整过程。日期、版本、来源和备选方案按需保留。
允许省略次要细节，不允许补造数字、原因或执行状态；工具错误标记本身不能推出退出码。
保持原意与确定性：建议、已执行和已验证有区别；未验证不等于无效；用户归因不能变成工具证明的原因。
一次具体经历可以保留为本次经验，不自动提升为通用规律；明确声明的默认偏好即使只说一次也应保留。
技术经历可简写为“本次在 X 条件下，采取 Y 后得到 Z”，作为参考实例，不自行追加下次应如何处理的指令。
用户明确约定的未来做法按其范围保留；单次未采用某方案，不意味着该方案无需使用或无效。
未采用的助手建议通常省略。原文“下载超时；用户说关闭代理后好了，不是重装解决”，
可记为“本次下载超时，关闭代理后成功；用户归因为代理”，不补成“并非安装问题”。
applicability 简述必要适用范围，可为空；影响使用的限制也应在 text 中。
reason 只填保存用途（如“保留排查线索”），不另作因果推断。
category/kind 按 schema 选择；项目工具约定可归 reference，任务进展归 task，namespace 不是 category。
episodes 描述经历来源，records.episode_ref 指向对应 episodes.ref。
records 的 ref 才是本批证据；context 仅供理解，程序注入、既有记忆读回与 reasoning 不作为新事实。
每条观察至少引用一个支持其正文的 ref。已有写入结果可用于 target_item_ids 对照，不据此跳过其他原文。
不调用工具，不使用模型外部知识。
"""

_DREAM_PROMPT = """整理 observations 和 explicit_changes，结合给出的正式 items 与修订 events。
目标是少量有用、忠实、易于复用的记忆。筛掉无后续价值的观察，合并重复和同一主题的相关要点。
以 original_evidence 的原始记录及显式 events 核对模型提炼稿，纠正无依据的细节，保留有效知识。
observations 只是草稿，不是独立证据；text、applicability 中的主张均以原始记录为准。
原文只报告失败就只保留失败，草稿补出的退出码等数字或诊断结论必须删去，不能因草稿写得确定就沿用。
text 直接表达值得记住的结论、方法或任务状态，并保留防止误用所必需的对象和条件。
允许有损压缩：日期、版本、来源经过和未执行方案仅在影响理解、使用或用户明确要求时写入正文。
保留原意、必要归属和确定性；用户判断不变成已证明事实，未验证不变成无效，局部测试不变成全面成功。
一次处置成功不代表已排除其他原因；未采用的建议可省略，若保留则仍写未尝试，不替它得出无效结论。
单次经历不自动成为通用规则；明确的默认偏好按声明范围保留；重复转述不增加独立证据。
同一主题的零散观察整理成简洁条目；已准确简洁的内容保持原句，仅为去重、必要条件或纠错而改写。
明确约定已覆盖同一经历的可用信息时，保留约定即可，不因分类不同再写一条等价经验。
技术经历按“本次条件、实际行动、结果”保留，不自行增添后续处置规则，也不把未采用的方案写成无需或无效。
例如“本次下载超时，换网络后成功”可以原样记住；改成“下载超时换网络即可解决”改变了含义。
普通“已读文件、正在处理”的进度不值得独立记忆；未完成状态只有对后续行动有具体帮助才保留。
reason 只解释保存或整理用途，不再次生成事实结论；没有新增价值时可保持现状。
只返回符合 schema 的 JSON；所有证据引用使用 evidence 中的 ref 标签。
explicit_changes 列出待检查的事件 ref；events.changes 给出字段修改前后的值。
evidence.record 是本次请求内的原文分组，同一分组的区间可能重叠；真实定位由程序保存。
每个观察必须有且仅有一条 resolution：归入当前条目或本响应新增标签，
或 target_id=null 并说明忽略原因。
对于仅有常规读取或正在处理等进度的观察，直接给 target_id=null、reason="普通执行进度"，不生成操作。
add 使用唯一 new_key，本程序会生成数据库 ID；其它操作 target_id 只能来自本批 items。
update 保持 ID；merge 选择已有 keeper 为 target_id，merge_ids 指向要合并的其它条目。
同一项目的同一约定、偏好或状态被纠正时，必须优先 update 原条目，保持其身份。
不要把同一事实的纠正做成 delete 旧条目再 add 新条目；delete 用于整体已无保留价值的知识。
同一已有条目只能有一种最终操作，不能同时更新和被合并。
evidence 给出支持当前最终正文的完整证据集合；旧说法被纠正后不再是新说法的当前支持。
support 只补充证据不改正文；重复观察可仅归入已有条目，无需膨胀其证据集合。
删除与失效条目以及显式较新纠正必须参与比较；不能把被忘记的旧事实重新写回。
明确的新用户要求可以构成新的证据，但不要仅凭旧观察、模型知识或未验证提议反转事实。
不同适用范围的冲突应保留条件。delete 只用于明确失效或否定，不因长期没检索而删除。
输入已足够完整且无需改变时 operations 可以为空；所有 explicit_changes 仍表示本批已检查。
"""


def _request(model: str, prompt: str, source: dict[str, object], max_tokens: int) -> LLMRequest:
    """构造一次独立预算的无工具请求。"""
    return LLMRequest(
        model=model,
        messages=[Msg.system(prompt), Msg.user(json.dumps(source, ensure_ascii=False))],
        max_tokens=max_tokens,
        temperature=0,
        response_format={"type": "json_object"},
    )


def _dependencies(service: MemoryService) -> tuple[CompletionProvider, str]:
    """在显式 SDK 阶段入口检查实际所需的生成依赖。"""
    if service.generation_provider is None or service.generation_model is None:
        raise IrisMemoryError("memory 生成 provider/model 未配置")
    return service.generation_provider, service.generation_model


def _parse[T: BaseModel](response: LLMResponse, schema: type[T]) -> T:
    """在外部响应边界接受完整 JSON，一次解析为可信模型。"""
    if response.finish_reason != "stop":
        raise IrisMemoryError("memory 生成响应未完整结束", finish_reason=response.finish_reason)
    try:
        return schema.model_validate_json(response.to_msg().text)
    except ValidationError as exc:
        raise IrisMemoryError("memory 生成响应不符合阶段 JSON 契约") from exc


def _usage(response: LLMResponse) -> dict[str, int]:
    """记录真实维护成本，与主运行 usage 分开。"""
    return {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
    }


async def before_generation_commit() -> None:
    """给前台入场机会，并拒绝吞掉取消后返回的迟到模型结果。"""
    await asyncio.sleep(0)
    raise_if_generation_cancelled()


def raise_if_generation_cancelled() -> None:
    """短提交收口后继续传播已请求的取消，不再启动下一阶段或派生工作。"""
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


def _flush_input(
    namespace: str, slices: list[EpisodeSlice], progresses: tuple[EpisodeProgress, ...]
) -> dict[str, object]:
    """序列化固定原文片段并附带不参与消费的有限前情。"""
    progresses_by_episode = {progress.episode.id: progress for progress in progresses}
    episode_refs: dict[str, str] = {}
    episode_inputs: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    context: list[str] = []
    for index, piece in enumerate(slices):
        progress = progresses_by_episode[piece.episode_id]
        episode = progress.episode
        captured = "lifecycle_source_id" in episode.metadata
        if episode.id not in episode_refs:
            episode_ref = f"p{len(episode_refs)}"
            episode_refs[episode.id] = episode_ref
            episode_input: dict[str, object] = {
                "ref": episode_ref,
                "source_type": episode.source_type.value,
            }
            if episode.source_id:
                episode_input["source_id"] = episode.source_id
            metadata = {
                key: value
                for key, value in episode.metadata.items()
                if not captured
                or key
                not in {
                    "lifecycle_source_id",
                    "run_id",
                    "start_message_count",
                    "end_message_count",
                    "outcome",
                }
            }
            if metadata:
                episode_input["metadata"] = metadata
            outcome = progress.source_outcome
            if outcome is None and captured:
                outcome = episode.metadata.get("outcome")
            if outcome is not None:
                episode_input["run_outcome"] = outcome
            episode_inputs.append(episode_input)
        record_index, record = next(
            (i, record) for i, record in enumerate(episode.records) if record.id == piece.record_id
        )
        projected: dict[str, object] = {
            "ref": f"e{index}"
            if record.text and record.metadata.get("evidence_allowed", True)
            else None,
            "episode_ref": episode_refs[episode.id],
            "role": record.role,
            "start": piece.start,
            "end": piece.end,
            "text": piece.text,
            "occurred_at": record.occurred_at,
            "source_type": record.source_type.value,
        }
        metadata = {
            key: value
            for key, value in record.metadata.items()
            if key != "evidence_allowed"
            and (not captured or key not in {"message_ordinal", "block_index"})
        }
        if metadata:
            projected["metadata"] = metadata
        if record.artifacts:
            projected["artifacts"] = [
                artifact.model_dump(mode="json") for artifact in record.artifacts
            ]
        records.append(projected)
        if index == 0:
            if piece.start:
                context.append(record.text[max(0, piece.start - 256) : piece.start])
            elif record_index:
                context.append(episode.records[record_index - 1].text[-256:])
    return {
        "namespace": namespace,
        "episodes": episode_inputs,
        "records": records,
        "context": context,
    }


def _select_flush(
    namespace: str,
    progresses: tuple[EpisodeProgress, ...],
    provider: CompletionProvider,
    model: str,
    config: MemoryGenerationConfig,
) -> tuple[tuple[EpisodeSlice, ...], LLMRequest]:
    """先按记录选片，单条超长才二分截取固定字符区间。"""
    slices: list[EpisodeSlice] = []
    prompt = _FLUSH_PROMPT + json.dumps(_FlushResponse.model_json_schema(), ensure_ascii=False)

    def build(pieces: list[EpisodeSlice]) -> LLMRequest:
        return _request(
            model,
            prompt,
            _flush_input(namespace, pieces, progresses),
            config.flush_output_budget_tokens,
        )

    for progress in progresses:
        for index in range(progress.cursor.record_index, len(progress.episode.records)):
            record = progress.episode.records[index]
            start = progress.cursor.text_offset if index == progress.cursor.record_index else 0
            end = len(record.text)
            piece = EpisodeSlice(progress.episode.id, record.id, start, end, record.text[start:end])
            if (
                provider.estimate_input_tokens(build([*slices, piece]))
                <= config.flush_input_budget_tokens
            ):
                slices.append(piece)
                continue
            if slices:
                return tuple(slices), build(slices)
            low, high = start, end
            while low < high:
                middle = (low + high + 1) // 2
                candidate = EpisodeSlice(
                    piece.episode_id, piece.record_id, start, middle, record.text[start:middle]
                )
                if (
                    provider.estimate_input_tokens(build([candidate]))
                    <= config.flush_input_budget_tokens
                ):
                    low = middle
                else:
                    high = middle - 1
            if low == start:
                raise IrisMemoryError("memory flush 输入预算不足以容纳一个证据片段")
            paragraph_end = record.text.rfind("\n", start, low)
            if paragraph_end > start:
                low = paragraph_end + 1
            slices.append(
                EpisodeSlice(piece.episode_id, piece.record_id, start, low, record.text[start:low])
            )
            return tuple(slices), build(slices)
    return tuple(slices), build(slices)


async def flush(service: MemoryService, namespace: str) -> GenerationResult:
    """消费一批固定原文片段，原子保存观察和处理位置。"""
    provider, model = _dependencies(service)
    started = perf_counter()
    usage: dict[str, int] = {}
    input_ids: tuple[str, ...] = ()
    result_recorded = False
    try:
        progresses = tuple(
            await service.run_async_io(lambda: service.store.list_pending_episodes(namespace))
        )
        if not progresses:
            return GenerationResult(namespace=namespace, stage="flush", status="empty")
        slices, request = _select_flush(
            namespace, progresses, provider, model, service.generation_config
        )
        input_ids = tuple(dict.fromkeys(piece.episode_id for piece in slices))
        records = {
            (progress.episode.id, record.id): record
            for progress in progresses
            for record in progress.episode.records
        }
        refs = {
            f"e{i}": MemoryEvidenceRef(
                kind="episode",
                source_id=piece.episode_id,
                record_id=piece.record_id,
                start=piece.start,
                end=piece.end,
            )
            for i, piece in enumerate(slices)
            if piece.text
            and records[piece.episode_id, piece.record_id].metadata.get("evidence_allowed", True)
        }
        if refs:
            response = await provider.complete(request)
            usage = _usage(response)
            extracted = _parse(response, _FlushResponse)
        else:
            extracted = _FlushResponse(observations=())
        known_items = {
            item_id
            for piece in slices
            for item_id in records[piece.episode_id, piece.record_id].metadata.get(
                "memory_item_ids", []
            )
        }
        observations: list[MemoryObservation] = []
        for observation in extracted.observations:
            if not set(observation.evidence) <= refs.keys():
                raise IrisMemoryError("flush 引用了本批之外的证据")
            if not set(observation.target_item_ids) <= known_items:
                raise IrisMemoryError("flush 引用了材料之外的记忆条目")
            observations.append(
                MemoryObservation.model_construct(
                    namespace=namespace,
                    text=observation.text,
                    applicability=observation.applicability,
                    category=observation.category,
                    kind=observation.kind,
                    reason=observation.reason,
                    evidence=tuple(refs[key] for key in observation.evidence),
                    target_item_ids=observation.target_item_ids,
                    generation_model=model,
                )
            )
        await before_generation_commit()
        result = GenerationResult(
            namespace=namespace,
            stage="flush",
            status="completed",
            usage=usage,
            elapsed_seconds=perf_counter() - started,
            input_ids=input_ids,
            counts={"observations": len(observations), "slices": len(slices)},
            consumed_ranges=tuple(
                MemoryConsumedRange(
                    episode_id=piece.episode_id,
                    record_id=piece.record_id,
                    start=piece.start,
                    end=piece.end,
                )
                for piece in slices
            ),
        )
        committed = await service.run_async_io(
            lambda: service.store.commit_flush(
                FlushCommit(namespace, slices, tuple(observations), result)
            ),
            complete_on_cancel=True,
        )
        if not committed:
            result = result.model_copy(update={"status": "conflict", "consumed_ranges": ()})
            await service.run_async_io(
                lambda: service.store.record_generation_result(result), complete_on_cancel=True
            )
        result_recorded = True
        raise_if_generation_cancelled()
        state = await service.ageneration_state(namespace)
        return result.model_copy(update={"has_more": bool(state.pending_episodes)})
    except (Exception, asyncio.CancelledError) as exc:
        if not result_recorded:
            await _failed(service, namespace, "flush", exc, started, usage, input_ids)
        raise


def _dream_input(snapshot: DreamSnapshot) -> tuple[dict[str, object], dict[str, MemoryEvidenceRef]]:
    """只投影整理所需内容，完整证据定位留在程序内。"""
    refs: dict[str, MemoryEvidenceRef] = {}
    by_value: dict[MemoryEvidenceRef, str] = {}

    def evidence(values: tuple[MemoryEvidenceRef, ...]) -> list[str]:
        keys: list[str] = []
        for value in values:
            if value not in by_value:
                key = f"e{len(refs)}"
                refs[key] = value
                by_value[value] = key
            keys.append(by_value[value])
        return keys

    observations = [
        {
            "id": observation.id,
            "text": observation.text,
            "applicability": observation.applicability,
            "category": observation.category.value,
            "kind": observation.kind.value,
            "evidence": evidence(observation.evidence),
            "target_item_ids": observation.target_item_ids,
        }
        for observation in snapshot.observations
    ]
    items: list[dict[str, object]] = []
    for item in snapshot.items:
        projected: dict[str, object] = {
            "id": item.id,
            "text": item.text,
            "category": item.category.value,
            "kind": item.kind.value,
            "status": item.status.value,
            "evidence": evidence(item.evidence),
        }
        if item.superseded_by is not None:
            projected["superseded_by"] = item.superseded_by
        if item.artifacts:
            projected["artifacts"] = [
                artifact.model_dump(mode="json") for artifact in item.artifacts
            ]
        if item.metadata:
            projected["metadata"] = item.metadata
        items.append(projected)
    events: list[dict[str, object]] = []
    event_refs: dict[str, str] = {}
    for event in snapshot.events:
        ref = evidence((MemoryEvidenceRef(kind="event", source_id=event.id),))[0]
        event_refs[event.id] = ref
        before, after = event.before or {}, event.after or {}
        changes = {
            field: {"before": before.get(field), "after": after.get(field)}
            for field in (
                "text",
                "category",
                "kind",
                "status",
                "superseded_by",
                "artifacts",
                "metadata",
            )
            if before.get(field) != after.get(field)
        }
        events.append(
            {
                "ref": ref,
                "event_type": event.event_type.value,
                "actor": event.actor.value,
                "source_type": event.source_type.value,
                "item_id": event.item_id,
                "reason": event.reason,
                "created_at": event.created_at,
                "changes": changes,
            }
        )
    record_refs: dict[tuple[str, str | None], str] = {}
    locators: list[dict[str, object]] = []
    for ref, value in refs.items():
        if value.kind == "event":
            locators.append({"ref": ref, "kind": "event"})
            continue
        record = (value.source_id, value.record_id)
        if record not in record_refs:
            record_refs[record] = f"r{len(record_refs)}"
        locators.append(
            {
                "ref": ref,
                "kind": "episode",
                "record": record_refs[record],
                "start": value.start,
                "end": value.end,
            }
        )
    return {
        "namespace": snapshot.namespace,
        "observations": observations,
        "items": items,
        "explicit_changes": [event_refs[change.event_id] for change in snapshot.changes],
        "events": events,
        "evidence": locators,
    }, refs


def _with_original_evidence(
    service: MemoryService,
    snapshot: DreamSnapshot,
) -> tuple[dict[str, object], dict[str, MemoryEvidenceRef]]:
    """按引用补充不可变原文；可变知识已经固定在同一 DreamSnapshot 内。"""
    source, refs = _dream_input(snapshot)
    episodes = {
        ref.source_id: service.store.get_episode(ref.source_id, snapshot.namespace)
        for ref in refs.values()
        if ref.kind == "episode"
    }
    excerpts: dict[str, dict[str, object]] = {}
    for key, ref in refs.items():
        if ref.kind == "episode":
            episode = episodes[ref.source_id]
            record = next(record for record in episode.records if record.id == ref.record_id)
            excerpts[key] = {
                "role": record.role,
                "text": record.text[ref.start : ref.end],
                "occurred_at": record.occurred_at,
                "metadata": record.metadata,
            }
    source["original_evidence"] = excerpts
    return source, refs


def _bind_plan(
    response: _DreamResponse,
    snapshot: DreamSnapshot,
    refs: dict[str, MemoryEvidenceRef],
) -> DreamPlan:
    """一次校验外部引用与整批去向，然后生成唯一内部计划。"""
    items = {item.id: item for item in snapshot.items}
    new_ids: dict[str, str] = {}
    touched: set[str] = set()
    operations: list[DreamOperation] = []
    for operation in response.operations:
        if not set(operation.evidence) <= refs.keys():
            raise IrisMemoryError("dream 引用了快照之外的证据")
        if operation.action == "add":
            key = operation.new_key
            if not key or key in new_ids or key in items or operation.target_id is not None:
                raise IrisMemoryError("dream 新增标签必须唯一且不冒充已有条目")
            new_ids[key] = _new_id()
        elif operation.target_id not in items or operation.new_key is not None:
            raise IrisMemoryError("dream 修改目标必须来自本次快照")
        targets = ({operation.target_id} if operation.target_id else set()) | set(
            operation.merge_ids
        )
        if not targets <= items.keys() or targets & touched:
            raise IrisMemoryError("dream 对同一条目给出了重复或范围外操作")
        if operation.action == "merge" and (
            not operation.merge_ids or operation.target_id in operation.merge_ids
        ):
            raise IrisMemoryError("dream 合并必须指定 keeper 之外的条目")
        if operation.action != "merge" and operation.merge_ids:
            raise IrisMemoryError("仅合并操作可以指定 merge_ids")
        if operation.action in {"add", "update", "merge"} and (
            operation.text is None or not operation.evidence
        ):
            raise IrisMemoryError("dream 新增或改写必须提供完整正文及当前支持证据")
        touched.update(targets)
        current = items.get(operation.target_id)
        operations.append(
            DreamOperation.model_construct(
                action=operation.action,
                target_id=operation.target_id,
                new_id=new_ids.get(operation.new_key),
                text=operation.text,
                category=operation.category
                or (current.category if current else MemoryCategory.USER),
                kind=operation.kind or (current.kind if current else MemoryItemKind.NOTE),
                evidence=tuple(refs[key] for key in operation.evidence),
                merge_ids=operation.merge_ids,
                reason=operation.reason,
            )
        )
    ids = [resolution.observation_id for resolution in response.resolutions]
    if len(set(ids)) != len(ids) or set(ids) != {item.id for item in snapshot.observations}:
        raise IrisMemoryError("dream 必须为本批每个观察提供唯一去向")
    retired = {target for operation in operations for target in operation.merge_ids}
    retired.update(
        cast(str, operation.target_id) for operation in operations if operation.action == "delete"
    )
    resolutions: list[ObservationResolution] = []
    for resolution in response.resolutions:
        target = resolution.target_id
        if target is not None and (
            target not in items and target not in new_ids or target in retired
        ):
            raise IrisMemoryError("dream 观察不能归入未知或本次退役条目")
        resolutions.append(
            ObservationResolution.model_construct(
                observation_id=resolution.observation_id,
                item_id=new_ids.get(target, target),
                reason=resolution.reason,
            )
        )
    return DreamPlan.model_construct(operations=tuple(operations), resolutions=tuple(resolutions))


async def dream(
    service: MemoryService,
    namespace: str,
    *,
    retry_blocked: bool = False,
) -> GenerationResult:
    """按完整比较包预算选材，原子整理一批观察及显式变更。"""
    provider, model = _dependencies(service)
    config = service.generation_config
    started = perf_counter()
    usage: dict[str, int] = {}
    input_ids: tuple[str, ...] = ()
    blocked = 0
    result_recorded = False
    try:
        await service.run_async_io(
            lambda: service.store.retry_blocked(
                namespace, budget=None if retry_blocked else config.dream_input_budget_tokens
            )
        )
        while True:
            snapshot = await service.run_async_io(
                lambda: service.store.read_dream_snapshot(namespace)
            )
            while True:
                if not snapshot.observations and not snapshot.changes:
                    result = GenerationResult(
                        namespace=namespace,
                        stage="dream",
                        status="blocked" if blocked else "empty",
                        counts={"blocked": blocked},
                    )
                    if blocked:
                        await service.run_async_io(
                            partial(service.store.record_generation_result, result)
                        )
                    return result
                source, refs = await service.run_async_io(
                    partial(_with_original_evidence, service, snapshot)
                )
                request = _request(
                    model,
                    _DREAM_PROMPT
                    + json.dumps(_DreamResponse.model_json_schema(), ensure_ascii=False),
                    source,
                    config.dream_output_budget_tokens,
                )
                if provider.estimate_input_tokens(request) <= config.dream_input_budget_tokens:
                    break
                inputs = [
                    *(("observation", item.id) for item in snapshot.observations),
                    *(("change", item.event_id) for item in snapshot.changes),
                ]
                if len(inputs) == 1:
                    marked = await service.run_async_io(
                        partial(
                            service.store.block_dream,
                            snapshot,
                            reason="完整比较材料超过 dreaming 输入预算",
                            budget=config.dream_input_budget_tokens,
                            dependency_item_ids=tuple(item.id for item in snapshot.items),
                        )
                    )
                    if not marked:
                        return GenerationResult(
                            namespace=namespace, stage="dream", status="conflict"
                        )
                    blocked += 1
                    break
                selected = inputs[: max(1, len(inputs) // 2)]
                snapshot = await service.run_async_io(
                    partial(
                        service.store.read_dream_snapshot,
                        namespace,
                        observation_ids=tuple(
                            key for kind, key in selected if kind == "observation"
                        ),
                        change_ids=tuple(key for kind, key in selected if kind == "change"),
                    )
                )
            if provider.estimate_input_tokens(request) <= config.dream_input_budget_tokens:
                break
        input_ids = tuple(
            [item.id for item in snapshot.observations]
            + [item.event_id for item in snapshot.changes]
        )
        response = await provider.complete(request)
        usage = _usage(response)
        plan = _bind_plan(_parse(response, _DreamResponse), snapshot, refs)
        await before_generation_commit()
        counts: dict[str, int] = dict(Counter(operation.action for operation in plan.operations))
        counts.update(
            blocked=blocked, ignored=sum(item.item_id is None for item in plan.resolutions)
        )
        changed_targets = {
            target
            for operation in plan.operations
            if operation.action != "support"
            for target in (operation.target_id, operation.new_id, *operation.merge_ids)
            if target is not None
        }
        counts.update(
            processed_observations=len(snapshot.observations),
            processed_changes=len(snapshot.changes),
            unchanged=sum(
                resolution.item_id is not None and resolution.item_id not in changed_targets
                for resolution in plan.resolutions
            )
            + sum(change.item_id not in changed_targets for change in snapshot.changes),
        )
        result = GenerationResult(
            namespace=namespace,
            stage="dream",
            status="completed",
            usage=usage,
            elapsed_seconds=perf_counter() - started,
            input_ids=input_ids,
            counts=counts,
            item_revision=snapshot.item_revision,
        )
        committed = await service.run_async_io(
            lambda: service.store.commit_dream(snapshot, plan, result=result),
            complete_on_cancel=True,
        )
        if not committed:
            result = result.model_copy(update={"status": "conflict"})
            await service.run_async_io(
                lambda: service.store.record_generation_result(result), complete_on_cancel=True
            )
        result_recorded = True
        raise_if_generation_cancelled()
        if committed:
            state = await service.ageneration_state(namespace)
            if state.item_revision != snapshot.item_revision:
                await service.run_async_io(lambda: service._rebuild_committed(namespace))
        state = await service.ageneration_state(namespace)
        return result.model_copy(
            update={
                "item_revision": state.item_revision,
                "has_more": bool(state.pending_observations or state.pending_changes),
            }
        )
    except (Exception, asyncio.CancelledError) as exc:
        if not result_recorded:
            await _failed(service, namespace, "dream", exc, started, usage, input_ids)
        raise


async def _failed(
    service: MemoryService,
    namespace: str,
    stage: Literal["flush", "dream"],
    exc: BaseException,
    started: float,
    usage: dict[str, int],
    input_ids: tuple[str, ...],
) -> None:
    """失败与取消保留已知成本和输入定位，不消费材料或伪装为空结果。"""
    result = GenerationResult(
        namespace=namespace,
        stage=stage,
        status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
        error=str(exc) or type(exc).__name__,
        usage=usage,
        elapsed_seconds=perf_counter() - started,
        input_ids=input_ids,
    )
    await service.run_async_io(
        lambda: service.store.record_generation_result(result), complete_on_cancel=True
    )
