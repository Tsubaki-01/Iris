"""有限 B 候选的唯一响应边界、离线检查与单文件发布。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ..exceptions import IrisError, IrisEvolutionError, IrisTemplateError
from ..message import LLMResponse
from ..prompts import PromptSnapshot
from ..utils.files import atomic_write_text
from ..utils.generation_worker import check_generation_cancelled
from .config import EvolutionConfig
from .models import RevisionTarget


@dataclass(frozen=True, slots=True)
class PromptTarget:
    """消费领域提供的当前变量与固定契约说明。"""

    name: str
    description: str
    sample_variables: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ConfigTarget:
    """装配绑定的唯一主 YAML 与原路径闭包解析器。"""

    path: Path
    validate: Callable[[dict[str, Any]], None]
    descriptions: dict[str, str]


@dataclass(frozen=True, slots=True)
class RevisionContext:
    """本次 B 读取的来源、原声明、目标基线及模型可见说明。"""

    prompt_snapshot: PromptSnapshot
    targets: tuple[RevisionTarget, ...]
    prompts: dict[str, PromptTarget]
    config: ConfigTarget | None
    raw_config: dict[str, Any] | None
    baselines: dict[Path, str]
    model_input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PreparedRevision:
    """已经通过本次目标和候选检查、等待资格检查后发布的结果。"""

    action: Literal["no_change", "prompt", "config"]
    reason: str
    targets: tuple[RevisionTarget, ...]
    effect: str
    baselines: dict[Path, str]
    path: Path | None = None
    content: str | None = None


class _NoChange(BaseModel):
    """本次不修改目标。"""

    model_config = ConfigDict(extra="forbid")
    action: Literal["no_change"]
    reason: str = Field(pattern=r"\S")


class _PromptChange(BaseModel):
    """只替换一个命名 prompt 的策略正文。"""

    model_config = ConfigDict(extra="forbid")
    action: Literal["prompt"]
    target: str
    body: str
    reason: str = Field(pattern=r"\S")


class _ConfigChange(BaseModel):
    """只给主配置中的有限叶字段赋值，不接受整份 YAML。"""

    model_config = ConfigDict(extra="forbid")
    action: Literal["config"]
    assignments: dict[str, Any] = Field(min_length=1)
    reason: str = Field(pattern=r"\S")


_RESPONSE = TypeAdapter(
    Annotated[_NoChange | _PromptChange | _ConfigChange, Field(discriminator="action")]
)


def revision_response_schema() -> dict[str, Any]:
    """返回与实际 B 解析器同源的固定响应结构。"""
    return _RESPONSE.json_schema()


def check_targets(targets: tuple[RevisionTarget, ...], config: EvolutionConfig) -> None:
    """在 A 或 host 的原始目标入口确认当前项目已开放这些目标。"""
    for target in targets:
        opened = config.prompt_targets if target.kind == "prompt" else config.config_targets
        if target.name not in opened:
            raise IrisEvolutionError("修订目标未开放", kind=target.kind, target=target.name)


def _read_text(path: Path) -> str:
    """读取本轮目标正文，初始化后的缺失文件不静默补种。"""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise IrisEvolutionError("修订目标读取失败", path=str(path), error=str(exc)) from exc


def _declared_value(raw: dict[str, Any], name: str) -> tuple[bool, Any]:
    """保留未声明与显式 null 的区别，不猜测历史运行采用值。"""
    node: Any = raw
    for part in name.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def prepare_revision(
    *,
    prompt_snapshot: PromptSnapshot,
    targets: tuple[RevisionTarget, ...],
    prompt_targets: tuple[PromptTarget, ...],
    config_target: ConfigTarget | None,
) -> RevisionContext:
    """在项目锁内重读本轮目标及原 YAML，构造 B 的真实当前输入。"""
    prompts = {target.name: target for target in prompt_targets}
    baselines: dict[Path, str] = {}
    raw_config: dict[str, Any] | None = None
    descriptions: list[dict[str, Any]] = []
    for target in targets:
        if target.kind == "prompt":
            binding = prompts[target.name]
            path = prompt_snapshot.root / f"{target.name}.j2"
            baselines[path] = _read_text(path)
            descriptions.append(
                {
                    "kind": "prompt",
                    "name": target.name,
                    "description": binding.description,
                    "sample_variables": binding.sample_variables,
                    "current_content": baselines[path],
                }
            )
        else:
            if config_target is None:
                raise IrisEvolutionError("配置修订缺少主 YAML 解析绑定")
            if raw_config is None:
                baselines[config_target.path] = _read_text(config_target.path)
                try:
                    raw = yaml.safe_load(baselines[config_target.path])
                except yaml.YAMLError as exc:
                    raise IrisEvolutionError(
                        "主 YAML 解析失败", path=str(config_target.path)
                    ) from exc
                if not isinstance(raw, dict):
                    raise IrisEvolutionError("主 YAML 必须是对象", path=str(config_target.path))
                raw_config = raw
            declared, value = _declared_value(raw_config, target.name)
            descriptions.append(
                {
                    "kind": "config",
                    "name": target.name,
                    "description": config_target.descriptions.get(target.name, ""),
                    "declared": declared,
                    "current_value": value,
                }
            )
    return RevisionContext(
        prompt_snapshot,
        targets,
        prompts,
        config_target,
        raw_config,
        baselines,
        {
            "targets": descriptions,
            "historical_context": (
                "历史实际采用的 prompt/config 和内部请求轨迹未知；"
                "以下只反映本轮读取的磁盘声明，不能证明贡献 Run 曾采用这些值。"
            ),
        },
    )


def prepare_candidate(response: LLMResponse, context: RevisionContext) -> PreparedRevision:
    """解析一次 B 响应，在当前开放目标内检查完整候选，不写文件。"""
    if response.finish_reason != "stop":
        raise IrisEvolutionError("修订响应未完整结束", finish_reason=response.finish_reason)
    try:
        candidate = _RESPONSE.validate_json(response.to_msg().text)
    except ValidationError as exc:
        raise IrisEvolutionError("修订响应不符合有限候选契约") from exc
    if candidate.action == "no_change":
        return PreparedRevision(
            "no_change", candidate.reason, context.targets, "目标保持不变。", context.baselines
        )
    allowed = {(target.kind, target.name) for target in context.targets}
    if candidate.action == "prompt":
        if ("prompt", candidate.target) not in allowed:
            raise IrisEvolutionError("修订目标不在本轮范围", target=candidate.target)
        binding = context.prompts[candidate.target]
        try:
            context.prompt_snapshot.with_template(candidate.target, candidate.body).render(
                candidate.target, binding.sample_variables
            )
        except (IrisTemplateError, UnicodeError) as exc:
            raise IrisEvolutionError(
                "Prompt 候选无法使用领域代表输入渲染", target=candidate.target, error=str(exc)
            ) from exc
        path = context.prompt_snapshot.root / f"{candidate.target}.j2"
        effect = (
            "下一次完整压缩采用；已开始的压缩保持原快照。"
            if candidate.target in {"compaction", "compaction_input"}
            else "下一次项目经验整理采用；本轮保持原快照。"
            if candidate.target == "project_skill_update"
            else "下一次 Memory 生成操作或自动维护周期采用；已开始周期保持原快照。"
        )
        return PreparedRevision(
            "prompt",
            candidate.reason,
            (RevisionTarget.model_construct(kind="prompt", name=candidate.target),),
            effect,
            {path: context.baselines[path]},
            path,
            candidate.body,
        )
    if any(("config", name) not in allowed for name in candidate.assignments):
        raise IrisEvolutionError("配置赋值包含未开放的目标")
    config = cast(ConfigTarget, context.config)
    original = cast(dict[str, Any], context.raw_config)
    if "system" in candidate.assignments and original.get("system") is None:
        raise IrisEvolutionError("system 只允许更新当前简单模式的文本")
    raw = original.copy()
    try:
        for name, value in candidate.assignments.items():
            parts = name.split(".")
            node = raw
            for part in parts[:-1]:
                # 沿赋值路径复制映射，避免 YAML 别名连带修改其它声明。
                node[part] = dict(node.get(part, {}))
                node = node[part]
            node[parts[-1]] = value
        config.validate(raw)
        content = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
    except (IrisError, ValidationError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise IrisEvolutionError("配置候选未通过原声明解析", error=str(exc)) from exc
    path = config.path
    return PreparedRevision(
        "config",
        candidate.reason,
        tuple(
            RevisionTarget.model_construct(kind="config", name=name)
            for name in candidate.assignments
        ),
        "仅新 runner 采用；已有 runner 的新 session、follow-up 和 Goal 连续运行保持原配置。",
        {path: context.baselines[path]},
        path,
        content,
    )


def publish_revision(candidate: PreparedRevision) -> bool:
    """资格通过后的短 IO：比较原文件基线，再原子发布；冲突返回 False。"""
    check_generation_cancelled()
    for path, baseline in candidate.baselines.items():
        try:
            current = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return False
        except (OSError, UnicodeError) as exc:
            raise IrisEvolutionError("修订发布前读取失败", path=str(path), error=str(exc)) from exc
        if current != baseline:
            return False
    if candidate.path is not None:
        try:
            atomic_write_text(
                candidate.path,
                candidate.content,
                temporary_directory=candidate.path.parent.parent
                if candidate.action == "prompt"
                else None,
            )
        except (OSError, UnicodeError) as exc:
            raise IrisEvolutionError(
                "修订发布失败", path=str(candidate.path), error=str(exc)
            ) from exc
    return True
