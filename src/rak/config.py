from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_data_dir

DEFAULT_DATA_DIR = Path(user_data_dir("rak"))
DEFAULT_MODEL = "all-MiniLM-L6-v2"
NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"


CONFIG_FILENAME = "config.json"
CONFIGURABLE_KEYS = {"llm_base_url", "llm_model", "llm_api_key", "model_name", "zot_command", "chunk_size", "chunk_overlap"}


def load_config(data_dir: Path) -> dict:
    import json
    path = data_dir / CONFIG_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_config(data_dir: Path, key: str, value: str) -> None:
    import json
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(data_dir)
    cfg[key] = value
    (data_dir / CONFIG_FILENAME).write_text(json.dumps(cfg, indent=2))


def detect_zotero_storage() -> Path | None:
    storage_dir = Path.home() / "Zotero" / "storage"
    return storage_dir if storage_dir.is_dir() else None


@dataclass
class RakConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    model_name: str = DEFAULT_MODEL
    zot_command: str = "zot"
    zotero_storage_dir: Path | None = field(default_factory=detect_zotero_storage)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3"
    llm_api_key: str = "not-needed"
    chunk_size: int = 512
    chunk_overlap: int = 64

    def __post_init__(self) -> None:
        saved = load_config(self.data_dir)
        for key, value in saved.items():
            if key in CONFIGURABLE_KEYS and hasattr(self, key):
                setattr(self, key, value)

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def fts_db_path(self) -> Path:
        return self.data_dir / "fts.sqlite"
