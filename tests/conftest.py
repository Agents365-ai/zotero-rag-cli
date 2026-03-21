from pathlib import Path

import pytest

from rak.config import RakConfig


@pytest.fixture
def tmp_config(tmp_path: Path) -> RakConfig:
    return RakConfig(data_dir=tmp_path)
