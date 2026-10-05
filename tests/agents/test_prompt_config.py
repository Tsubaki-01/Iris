"""项目 prompt 的唯一声明入口只解析配置，不初始化来源。"""

from pathlib import Path

from iris.agents import AgentConfig, load_agent_config


def test_prompt_root_defaults_without_file_io(tmp_path: Path) -> None:
    """默认相对 root workspace，配置加载不打开任何模板。"""
    config = AgentConfig.model_validate({"name": "a", "model": "openai/test", "system": "a"})
    assert config.prompts.root == ".iris/prompts"
    path = tmp_path / "agent.yaml"
    path.write_text(
        "name: a\nmodel: openai/test\nsystem: a\nprompts:\n  root: custom-prompts\n",
        encoding="utf-8",
    )
    assert load_agent_config(path).prompts.root == "custom-prompts"
    assert list(tmp_path.iterdir()) == [path]
