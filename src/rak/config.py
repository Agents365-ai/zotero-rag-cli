from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_data_dir

DEFAULT_DATA_DIR = Path(user_data_dir("rak"))
DEFAULT_MODEL = "all-MiniLM-L6-v2"
NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def detect_zotero_storage() -> Path | None:
    storage_dir = Path.home() / "Zotero" / "storage"
    return storage_dir if storage_dir.is_dir() else None


@dataclass
class RakConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    model_name: str = DEFAULT_MODEL
    zot_command: str = "zot"
    zotero_storage_dir: Path | None = field(default_factory=detect_zotero_storage)

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def fts_db_path(self) -> Path:
        return self.data_dir / "fts.sqlite"
