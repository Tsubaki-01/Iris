"""Workspace 内 Skill 目录的确定性发现与优先级归并。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..exceptions import (
    IrisSkillError,
    IrisSkillFormatError,
    IrisSkillPathError,
)
from ..tools.permissions import WorkspacePolicy
from .frontmatter import parse_frontmatter, split_frontmatter
from .models import (
    _NAME_RE,
    SKILL_FILE_NAME,
    SkillDiagnostic,
    SkillDiscoveryOptions,
    SkillDiscoveryResult,
    SkillMetadata,
    SkillScope,
    _normalize_description,
)


def resolve_skills_root(
    raw_root: str,
    *,
    workspace_root: Path,
    policy: WorkspacePolicy | None = None,
) -> Path:
    """把配置中的 skills root 解析为 workspace 内的绝对 realpath。"""
    resolved_workspace = workspace_root.resolve(strict=False)
    if not raw_root.strip():
        raise IrisSkillPathError(
            "skills root 不能为空",
            path=raw_root,
            workspace_root=str(resolved_workspace),
        )

    raw_path = Path(raw_root)
    candidate = raw_path if raw_path.is_absolute() else resolved_workspace / raw_path
    resolved = candidate.resolve(strict=False)
    workspace_policy = policy or WorkspacePolicy()
    if not workspace_policy.is_within_workspace(resolved, resolved_workspace):
        raise IrisSkillPathError(
            "skills root 不在 workspace 内",
            path=str(resolved),
            workspace_root=str(resolved_workspace),
        )
    return resolved


def discover_skills(options: SkillDiscoveryOptions) -> SkillDiscoveryResult:
    """按 roots 声明顺序发现、归并并稳定排序 Skill。"""
    workspace_root = options.workspace_root.resolve(strict=False)
    scanned: list[tuple[int, list[SkillMetadata]]] = []
    diagnostics: list[SkillDiagnostic] = []

    for root_index, (scope, root) in enumerate(options.roots):
        resolved_root = resolve_skills_root(
            str(root),
            workspace_root=workspace_root,
        )
        skills, root_diagnostics = _scan_root(
            resolved_root,
            workspace_root=workspace_root,
            scope=scope,
            root_index=root_index,
            max_description_chars=options.max_description_chars,
        )
        scanned.append((root_index, skills))
        diagnostics.extend(root_diagnostics)

    winners, collision_diagnostics = _merge_by_priority(scanned)
    diagnostics.extend(collision_diagnostics)
    winners.sort(key=lambda skill: (skill.root_index, skill.name))
    return SkillDiscoveryResult(
        skills=tuple(winners),
        diagnostics=tuple(diagnostics),
    )


def _scan_root(
    root: Path,
    *,
    workspace_root: Path,
    scope: SkillScope,
    root_index: int,
    max_description_chars: int,
) -> tuple[list[SkillMetadata], list[SkillDiagnostic]]:
    """扫描单个 root，并把单条 Skill 错误降级为诊断。"""
    resolved_root = root.resolve(strict=False)
    try:
        if not resolved_root.is_dir():
            return [], [_root_missing_diagnostic(resolved_root, scope, root_index)]
        candidates = sorted(resolved_root.iterdir(), key=lambda path: path.name)
    except OSError:
        return [], [_root_missing_diagnostic(resolved_root, scope, root_index)]

    skills: list[SkillMetadata] = []
    diagnostics: list[SkillDiagnostic] = []
    workspace_policy = WorkspacePolicy()
    resolved_workspace = workspace_root.resolve(strict=False)

    for candidate in candidates:
        if candidate.name.startswith("."):
            continue
        try:
            if not candidate.is_dir():
                continue
        except OSError as exc:
            diagnostics.append(
                SkillDiagnostic(
                    code="INVALID_SKILL",
                    message="无法访问 Skill 目录",
                    path=candidate.resolve(strict=False),
                    detail={"reason": str(exc)},
                )
            )
            continue

        if _NAME_RE.fullmatch(candidate.name) is None:
            diagnostics.append(
                SkillDiagnostic(
                    code="INVALID_NAME",
                    message="Skill 目录名必须是小写 kebab-case",
                    path=candidate.resolve(strict=False),
                    detail={"name": candidate.name},
                )
            )
            continue

        resolved_candidate = candidate.resolve(strict=False)
        candidate_is_contained = workspace_policy.is_within_workspace(
            resolved_candidate,
            resolved_root,
        ) and workspace_policy.is_within_workspace(
            resolved_candidate,
            resolved_workspace,
        )
        if candidate_is_contained:
            try:
                has_skill_file = any(
                    child.name == SKILL_FILE_NAME for child in candidate.iterdir()
                )
            except OSError as exc:
                diagnostics.append(
                    SkillDiagnostic(
                        code="INVALID_SKILL",
                        message="无法枚举 Skill 目录",
                        path=resolved_candidate,
                        detail={"reason": str(exc)},
                    )
                )
                continue
            if not has_skill_file:
                diagnostics.append(
                    SkillDiagnostic(
                        code="MISSING_SKILL_FILE",
                        message=f"Skill 目录缺少 {SKILL_FILE_NAME}",
                        path=resolved_candidate,
                    )
                )
                continue

        try:
            metadata = _load_skill(
                candidate,
                workspace_root=resolved_workspace,
                scope=scope,
                root=resolved_root,
                root_index=root_index,
                max_description_chars=max_description_chars,
            )
        except IrisSkillError as exc:
            diagnostics.append(_invalid_skill_diagnostic(exc, candidate))
            continue

        skills.append(metadata)
        if metadata.declared_name is not None and metadata.declared_name != metadata.name:
            diagnostics.append(
                SkillDiagnostic(
                    code="NAME_MISMATCH",
                    message="frontmatter name 与 Skill 目录名不一致",
                    path=metadata.skill_file,
                    detail={
                        "declared_name": metadata.declared_name,
                        "directory_name": metadata.name,
                    },
                )
            )
        if metadata.description_truncated:
            diagnostics.append(
                SkillDiagnostic(
                    code="DESCRIPTION_TRUNCATED",
                    message="Skill description 超过字符上限并已截断",
                    path=metadata.skill_file,
                    detail={"limit": str(max_description_chars)},
                )
            )

    return skills, diagnostics


def _load_skill(
    skill_dir: Path,
    *,
    workspace_root: Path,
    scope: SkillScope,
    root: Path,
    root_index: int,
    max_description_chars: int,
) -> SkillMetadata:
    """严格加载单个 Skill 目录并构造不含正文的元数据。"""
    if _NAME_RE.fullmatch(skill_dir.name) is None:
        raise IrisSkillFormatError(
            "Skill 目录名必须是小写 kebab-case",
            path=str(skill_dir.resolve(strict=False)),
            name=skill_dir.name,
        )

    resolved_workspace = workspace_root.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    resolved_skill_dir = skill_dir.resolve(strict=False)
    _require_containment(
        resolved_root,
        boundary=resolved_workspace,
        workspace_root=resolved_workspace,
    )
    _require_containment(
        resolved_skill_dir,
        boundary=resolved_root,
        workspace_root=resolved_workspace,
    )
    _require_containment(
        resolved_skill_dir,
        boundary=resolved_workspace,
        workspace_root=resolved_workspace,
    )

    try:
        skill_file = next(
            (
                child
                for child in skill_dir.iterdir()
                if child.name == SKILL_FILE_NAME
            ),
            None,
        )
    except OSError as exc:
        raise IrisSkillFormatError(
            "无法枚举 Skill 目录",
            path=str(resolved_skill_dir),
            reason=str(exc),
        ) from exc
    if skill_file is None:
        raise IrisSkillFormatError(
            f"Skill 目录缺少 {SKILL_FILE_NAME}",
            path=str(resolved_skill_dir),
        )

    resolved_skill_file = skill_file.resolve(strict=False)
    _require_containment(
        resolved_skill_file,
        boundary=resolved_root,
        workspace_root=resolved_workspace,
    )
    _require_containment(
        resolved_skill_file,
        boundary=resolved_workspace,
        workspace_root=resolved_workspace,
    )
    try:
        is_file = resolved_skill_file.is_file()
    except OSError as exc:
        raise IrisSkillFormatError(
            f"无法访问 {SKILL_FILE_NAME}",
            path=str(resolved_skill_file),
            reason=str(exc),
        ) from exc
    if not is_file:
        raise IrisSkillFormatError(
            f"{SKILL_FILE_NAME} 必须是普通文件",
            path=str(resolved_skill_file),
        )

    try:
        text = resolved_skill_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IrisSkillFormatError(
            f"无法以 UTF-8 读取 {SKILL_FILE_NAME}",
            path=str(resolved_skill_file),
            reason=str(exc),
        ) from exc

    try:
        frontmatter_text, _body = split_frontmatter(text)
        frontmatter = parse_frontmatter(frontmatter_text)
    except IrisSkillFormatError as exc:
        context = {**exc.context, "path": str(resolved_skill_file)}
        raise IrisSkillFormatError(exc.message, **context) from exc

    raw_description = frontmatter.get("description")
    if not isinstance(raw_description, str):
        raise IrisSkillFormatError(
            "Skill description 必须是非空字符串",
            path=str(resolved_skill_file),
            field="description",
        )
    try:
        description, description_truncated = _normalize_description(
            raw_description,
            max_description_chars,
        )
    except ValueError as exc:
        raise IrisSkillFormatError(
            "Skill description 必须是非空字符串",
            path=str(resolved_skill_file),
            field="description",
        ) from exc

    raw_name = frontmatter.get("name")
    declared_name = None if raw_name is None else str(raw_name)
    extra_frontmatter = {
        key: value
        for key, value in frontmatter.items()
        if key not in {"name", "description"}
    }
    del _body

    return SkillMetadata(
        name=skill_dir.name,
        description=description,
        scope=scope,
        skill_file=resolved_skill_file,
        root_dir=resolved_skill_dir,
        relative_skill_file=resolved_skill_file.relative_to(
            resolved_workspace
        ).as_posix(),
        root_index=root_index,
        description_truncated=description_truncated,
        declared_name=declared_name,
        extra_frontmatter=extra_frontmatter,
    )


def _merge_by_priority(
    scanned: Sequence[tuple[int, list[SkillMetadata]]],
) -> tuple[list[SkillMetadata], list[SkillDiagnostic]]:
    """按 root_index first-wins 归并并诊断真实同名冲突。"""
    winners: dict[str, SkillMetadata] = {}
    diagnostics: list[SkillDiagnostic] = []

    for _, skills in sorted(scanned, key=lambda item: item[0]):
        for skill in sorted(skills, key=lambda item: item.name):
            winner = winners.get(skill.name)
            if winner is None:
                winners[skill.name] = skill
                continue
            if winner.root_dir.resolve(strict=False) == skill.root_dir.resolve(
                strict=False
            ):
                continue
            diagnostics.append(
                SkillDiagnostic(
                    code="DUPLICATE_NAME",
                    message="较低优先级的同名 Skill 已被忽略",
                    path=skill.root_dir,
                    detail={
                        "winner_path": str(winner.root_dir),
                        "winner_scope": winner.scope.value,
                        "loser_path": str(skill.root_dir),
                        "loser_scope": skill.scope.value,
                    },
                )
            )

    return list(winners.values()), diagnostics


def _require_containment(
    path: Path,
    *,
    boundary: Path,
    workspace_root: Path,
) -> None:
    policy = WorkspacePolicy()
    if not policy.is_within_workspace(path, boundary):
        raise IrisSkillPathError(
            "Skill 路径不在允许的目录内",
            path=str(path),
            boundary=str(boundary),
            workspace_root=str(workspace_root),
        )


def _root_missing_diagnostic(
    root: Path,
    scope: SkillScope,
    root_index: int,
) -> SkillDiagnostic:
    return SkillDiagnostic(
        code="ROOT_MISSING",
        message="Skill root 不存在、不是目录或无法访问",
        path=root,
        detail={"scope": scope.value, "root_index": str(root_index)},
    )


def _invalid_skill_diagnostic(
    error: IrisSkillError,
    candidate: Path,
) -> SkillDiagnostic:
    raw_path = error.context.get("path")
    path = candidate.resolve(strict=False) if raw_path is None else Path(str(raw_path))
    return SkillDiagnostic(
        code="INVALID_SKILL",
        message="Skill 无效并已跳过",
        path=path,
        detail={"reason": error.message},
    )


__all__ = ["discover_skills", "resolve_skills_root"]
