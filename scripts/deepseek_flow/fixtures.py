"""DeepSeek live 验证场景 fixture 写入。"""

from __future__ import annotations

import sys
from pathlib import Path

from .constants import DEFAULT_MODEL


def prepare_read_agent(base_dir: Path, *, session_backend: str) -> tuple[Path, str]:
    """准备只读文件工具验证用 agent 配置。"""
    workspace = base_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    token = f"deepseek-flow-token-{base_dir.name}"
    (workspace / "verification.txt").write_text(f"验证码: {token}\n", encoding="utf-8")
    session_yaml = (
        "session:\n  backend: sqlite\n  path: session.db"
        if session_backend == "sqlite"
        else "session:\n  backend: none"
    )
    agent_path = _write_agent_yaml(
        base_dir,
        f"""
name: read-agent-live
model:
  provider: deepseek
  name: {DEFAULT_MODEL}
  temperature: 0
  max_tokens: 200
  timeout: 60
system: |
  你是 Iris DeepSeek 文件读取验证助手。用户要求读取文件时，必须调用 read_file。
tools:
  builtin:
    - file.read
permissions:
  workspace: workspace
  writes: deny
{session_yaml}
""",
    )
    return agent_path, token


def prepare_text_agent(base_dir: Path, name: str) -> Path:
    """准备不带工具的文本 agent 配置。"""
    return _write_agent_yaml(
        base_dir,
        f"""
name: {name}
model:
  provider: deepseek
  name: {DEFAULT_MODEL}
  temperature: 0
  max_tokens: 160
system: |
  你是 Iris live 验证助手。严格遵循用户要求输出。
session:
  backend: none
""",
    )


def prepare_all_file_tools_agent(base_dir: Path) -> Path:
    """准备包含全部内置文件工具的 agent 配置和 workspace。"""
    workspace = base_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "generated.txt").unlink(missing_ok=True)
    (workspace / "notes.txt").write_text(
        "ALPHA_PATTERN_0708\nold-value\n",
        encoding="utf-8",
    )
    return _write_agent_yaml(
        base_dir,
        f"""
name: builtin-file-tools-live
model:
  provider: deepseek
  name: {DEFAULT_MODEL}
  temperature: 0
  max_tokens: 180
  timeout: 60
system: |
  你是 Iris 文件工具 live 验证助手。用户指定工具和参数时，必须调用对应工具。
tools:
  builtin:
    - file.list
    - file.read
    - file.grep
    - file.write
    - file.edit
permissions:
  workspace: workspace
  writes: allow
session:
  backend: none
""",
    )


def prepare_file_not_read_recovery_agent(base_dir: Path) -> Path:
    """准备 FILE_NOT_READ 后模型自恢复验证用 agent 配置。"""
    workspace = base_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "target.txt").write_text(
        "ORIGINAL_WRITE_TOKEN_0708\n", encoding="utf-8"
    )
    return _write_agent_yaml(
        base_dir,
        f"""
name: file-not-read-recovery-live
model:
  provider: deepseek
  name: {DEFAULT_MODEL}
  temperature: 0
  max_tokens: 180
  timeout: 60
system: |
  你是 Iris 文件工具错误恢复验证助手。必须按用户要求调用文件工具，不要直接回答。
tools:
  builtin:
    - file.read
    - file.write
permissions:
  workspace: workspace
  writes: allow
session:
  backend: none
""",
    )


def _write_python_tool_module(base_dir: Path) -> None:
    """写入 live 验证用 Python 工具模块。"""
    base_dir.mkdir(parents=True, exist_ok=True)
    module_path = base_dir / "deepseek_live_tools.py"
    module_path.write_text(
        '''
def search_notes(query: str) -> str:
    """搜索 live 验证笔记。"""
    return f"PYTHON_TOOL_TOKEN_0708: {query}"
'''.strip(),
        encoding="utf-8",
    )
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))


def _write_agent_yaml(base_dir: Path, content: str) -> Path:
    """写入 agent.yaml。"""
    base_dir.mkdir(parents=True, exist_ok=True)
    agent_path = base_dir / "agent.yaml"
    agent_path.write_text(content.strip() + "\n", encoding="utf-8")
    return agent_path
