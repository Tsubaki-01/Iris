"""图片示例使用真实导入、YAML 装配和 SQLite，仅替换模型响应。"""

from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from examples.image import basic
from iris.exceptions import IrisImageError
from iris.harness import AgentRunner
from iris.lifecycle import RunStopReason
from iris.message import ImageBlock, TextBlock
from iris.store import SQLiteStore
from iris.tools._paths import safe_path_segment
from tests.harness.fakes import StaticProvider, text_response


def _config(tmp_path: Path) -> Path:
    """复制真实示例配置，使所有持久化数据留在本次测试目录。"""
    folder = tmp_path / "agent"
    folder.mkdir()
    target = folder / "agent.yaml"
    target.write_text(basic.CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _provider_runner(
    monkeypatch: pytest.MonkeyPatch, provider: StaticProvider
) -> list[AgentRunner]:
    """保留示例的真实 runner 装配与关闭，仅在模型边界注入离线替身。"""
    construct = AgentRunner.from_config_path
    runners: list[AgentRunner] = []

    def from_config(path: Path) -> AgentRunner:
        runner = construct(path, provider=provider)
        runner.aclose = AsyncMock(wraps=runner.aclose)
        runners.append(runner)
        return runner

    monkeypatch.setattr(AgentRunner, "from_config_path", staticmethod(from_config))
    return runners


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["描述图片", ""])
async def test_example_imports_before_run_and_keeps_sqlite_image_after_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str
) -> None:
    """命令行来源路径相对当前目录解析，混排和纯图均持久化同一 cache 引用。"""
    config = _config(tmp_path)
    output = BytesIO()
    with Image.new("RGB", (12, 8), "red") as pixels:
        pixels.save(output, format="PNG")
    source = tmp_path / "source.png"
    source.write_bytes(output.getvalue())
    monkeypatch.chdir(tmp_path)
    provider = StaticProvider(text_response("图片包含红色区域"))
    runners = _provider_runner(monkeypatch, provider)

    result = await basic.run_agent(
        config_path=config, image_path=Path("source.png"), prompt=prompt, session_id="图/一"
    )

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert result.assistant_message.text == "图片包含红色区域"
    stored = SQLiteStore(config.parent / ".iris" / "image.db").load_run(result.run.run_id)
    content = stored.request.input
    assert isinstance(content, list)
    assert [block.type for block in content] == (["text", "image"] if prompt else ["image"])
    if prompt:
        assert content[0] == TextBlock(text=prompt)
    image = content[-1]
    assert isinstance(image, ImageBlock) and image.name == "source.png"
    assert image.model.path.parent == (
        config.parent / ".iris" / "image-cache" / safe_path_segment("图/一")
    )
    assert image.original.path.read_bytes() == output.getvalue()
    assert any(message.content == content for message in provider.requests[0].messages)
    assert runners[0].runtime.environment.tool_bridge.tool_view.get("read_file") is not None
    runners[0].aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_example_closes_runner_when_import_fails_before_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    source = tmp_path / "invalid.png"
    source.write_bytes(b"not an image")
    provider = StaticProvider()
    runners = _provider_runner(monkeypatch, provider)
    with pytest.raises(IrisImageError):
        await basic.run_agent(
            config_path=config, image_path=source, prompt="", session_id="invalid"
        )
    assert provider.requests == []
    assert runners[0].store.load_session("invalid").messages == []
    runners[0].aclose.assert_awaited_once()
