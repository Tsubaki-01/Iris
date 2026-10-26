"""Evolution 备份保留材料及消费位置。"""

from pathlib import Path

import pytest

from iris.evolution.materials import EvolutionMaterialStore
from iris.exceptions import IrisEvolutionError

from .test_materials import _block, _source


def test_backup_keeps_material_watermarks(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path / "live")
    source = _source()
    store.register_source(source, 2)
    store.commit_capture(_block(source, 2, 4))
    restored_root = tmp_path / "copy"
    store.backup(restored_root / ".iris/evolution/evolution.db")
    saved = EvolutionMaterialStore(restored_root)
    assert saved.list_capture_sources("lifecycle") == store.list_capture_sources("lifecycle")
    store.commit_capture(_block(source, 4, 5))
    assert saved.list_capture_sources("lifecycle")[0].captured_until == 4
    with pytest.raises(IrisEvolutionError):
        store.backup(store.path)
