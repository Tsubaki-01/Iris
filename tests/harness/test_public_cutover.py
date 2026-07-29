"""冻结 Phase 5 必须一次性删除的旧公共面与持久化真相。"""

from __future__ import annotations

from dataclasses import fields
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from shutil import copyfile

import pytest

_OLD_RUNTIME_METHODS = (
    "run_turn",
    "run_loop",
    "resume",
    "load_resumable_interaction",
)

_OLD_RUNTIME_MODELS = (
    "BoundedLoopOptions",
    "ProviderResponseSnapshot",
    "Runstate",
    "RuntimeContinuationClaim",
    "RuntimeErrorInfo",
    "RuntimeHITLCheckpoint",
    "RuntimeOptions",
    "RuntimeOptionsSnapshot",
    "RuntimeStatus",
    "RuntimeTurnInput",
    "RuntimeTurnResult",
    "ToolErrorPolicy",
    "ToolResultCommit",
)

_OLD_MODULES = (
    "iris.runtime.checkpoint",
    "iris.runtime.metadata",
    "iris.runtime.resume",
    "iris.runtime.tool_result_committer",
    "iris.session",
    "iris.hitl.store",
    "iris.hitl.in_memory",
    "iris.hitl._legacy_service",
    "iris.store._legacy_sqlite",
)

_OLD_HELPERS = (
    ("iris.runtime.checkpoint", "build_hitl_checkpoint"),
    ("iris.runtime.checkpoint", "validate_hitl_checkpoint"),
    ("iris.runtime.metadata", "build_run_metadata"),
    ("iris.runtime.metadata", "build_resume_run_metadata"),
    ("iris.runtime.metadata", "synchronize_resume_metadata"),
    ("iris.runtime.resume", "append_resumed_result"),
    ("iris.runtime.resume", "commit_ready_interaction"),
    ("iris.runtime.resume", "load_resumable_interaction"),
    ("iris.runtime.resume", "resolve_interaction_result"),
    ("iris.runtime.tool_result_committer", "commit_tool_results"),
    ("iris.runtime.tool_result_committer", "project_tool_result_messages"),
)

_OLD_WRITERS = (
    ("iris.session.store", "SessionStore", "save_messages"),
    ("iris.session.store", "SessionStore", "save_run_metadata"),
    ("iris.session.store", "SessionStore", "append_tool_event"),
    ("iris.hitl.store", "InteractionStore", "create_interaction"),
    ("iris.hitl.store", "InteractionStore", "resolve_interaction"),
    ("iris.hitl.store", "InteractionStore", "claim_interaction"),
    ("iris.hitl.store", "InteractionStore", "update_consumed_interaction"),
    ("iris.store", "SQLiteStore", "save_messages"),
    ("iris.store", "SQLiteStore", "save_run_metadata"),
    ("iris.store", "SQLiteStore", "append_tool_event"),
    ("iris.store", "SQLiteStore", "create_interaction"),
    ("iris.store", "SQLiteStore", "resolve_interaction"),
    ("iris.store", "SQLiteStore", "claim_interaction"),
    ("iris.store", "SQLiteStore", "update_consumed_interaction"),
)

_OLD_SCHEMA_FIXTURE = Path(__file__).parents[1] / "fixtures" / "lifecycle" / "old_session.db"


def _module_exists(module_name: str) -> bool:
    """在父 package 已删除时也安全判断旧 module 是否仍存在。"""
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize("method_name", _OLD_RUNTIME_METHODS)
def test_removed_runtime_method_target_contract(method_name: str) -> None:
    """定义 AgentRuntime 不再拥有 complete-run 编排方法。"""
    runtime_module = import_module("iris.runtime.runtime")

    assert not hasattr(runtime_module.AgentRuntime, method_name)


@pytest.mark.parametrize("model_name", _OLD_RUNTIME_MODELS)
def test_removed_runtime_model_target_contract(model_name: str) -> None:
    """定义旧 options/status/result/checkpoint/continuation 类型不可访问。"""
    models_module = import_module("iris.runtime.models")

    assert not hasattr(models_module, model_name)


@pytest.mark.parametrize("module_name", _OLD_MODULES)
def test_removed_persistence_module_target_contract(module_name: str) -> None:
    """定义旧 helper、session 与 interaction persistence module 必须删除。"""
    assert not _module_exists(module_name)


@pytest.mark.parametrize(("module_name", "helper_name"), _OLD_HELPERS)
def test_removed_checkpoint_metadata_helper_target_contract(
    module_name: str,
    helper_name: str,
) -> None:
    """逐项冻结旧 checkpoint、metadata、resume 与 committer helper 清单。"""
    if not _module_exists(module_name):
        return

    module = import_module(module_name)
    assert not hasattr(module, helper_name)


@pytest.mark.parametrize(("module_name", "owner_name", "writer_name"), _OLD_WRITERS)
def test_removed_split_writer_target_contract(
    module_name: str,
    owner_name: str,
    writer_name: str,
) -> None:
    """逐项冻结旧 session/interaction 分散写入入口。"""
    if not _module_exists(module_name):
        return

    module = import_module(module_name)
    owner = getattr(module, owner_name)
    assert not hasattr(owner, writer_name)


@pytest.mark.parametrize("field_name", ["session_store", "interaction_service"])
def test_runtime_environment_drops_persistence_owner_target_contract(field_name: str) -> None:
    """定义 low-level runtime environment 不再拥有 durable store/service。"""
    environment_module = import_module("iris.runtime.environment")
    field_names = {field.name for field in fields(environment_module.RuntimeEnvironment)}

    assert field_name not in field_names


def test_harness_exports_the_complete_run_surface_target_contract() -> None:
    harness = import_module("iris.harness")
    expected = {
        "AgentRunner",
        "AgentRunRequest",
        "AgentRunOptions",
        "RuntimeExecutionOptions",
        "RunLimits",
        "RunPhase",
        "RunStopReason",
        "RunUsage",
        "RunErrorInfo",
        "RunSnapshot",
        "RunResult",
        "RunEvent",
        "RunEventKind",
        "RunEventObserver",
    }

    assert expected <= set(harness.__all__)


def test_hitl_drops_standalone_resume_phase_target_contract() -> None:
    hitl = import_module("iris.hitl")

    assert not hasattr(hitl, "InteractionResumePhase")
    assert not hasattr(hitl.HumanInteraction, "resume_phase")
    assert not hasattr(hitl.HumanInteraction, "checkpoint")


def test_old_sqlite_schema_is_rejected_without_mutation_target_contract(
    tmp_path: Path,
) -> None:
    """定义旧 session/HITL schema 只能 fail closed，不能迁移或改写。"""
    assert _OLD_SCHEMA_FIXTURE.is_file()
    target = tmp_path / "old_session.db"
    copyfile(_OLD_SCHEMA_FIXTURE, target)
    before = target.read_bytes()
    store_module = import_module("iris.store")
    errors_module = import_module("iris.exceptions")

    with pytest.raises(errors_module.IrisLifecycleSchemaError):
        store_module.SQLiteStore(target)

    assert target.read_bytes() == before
