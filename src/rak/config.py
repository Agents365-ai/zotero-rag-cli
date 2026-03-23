from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / "Zotero" / "rak"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


CONFIG_FILENAME = "config.json"
CONFIGURABLE_KEYS = {"llm_base_url", "llm_model", "llm_api_key", "model_name", "zot_command", "chunk_size", "chunk_overlap", "embedding_provider", "embedding_base_url", "embedding_api_key", "pdf_provider"}
CONFIG_TYPES: dict[str, type] = {"chunk_size": int, "chunk_overlap": int}


def _coerce_config_value(key: str, value: str | int | float) -> str | int:
    """Coerce config value to the correct type."""
    target_type = CONFIG_TYPES.get(key)
    if target_type is not None:
        try:
            return target_type(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value for {key}: expected {target_type.__name__}, got {value!r}")
    return str(value)


def _validate_zot_command(value: str) -> None:
    """Reject zot_command values containing path separators or shell metacharacters."""
    import re
    if re.search(r'[/\\;|&$`<>(){}\[\]!#~]', value):
        raise ValueError(f"Invalid zot_command: {value!r} — must be a simple executable name")


def load_config(data_dir: Path) -> dict:
    import json
    path = data_dir / CONFIG_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _validate_chunk_params(cfg: dict, key: str, value: int) -> None:
    """Validate that chunk_overlap < chunk_size after applying the new value."""
    chunk_size = cfg.get("chunk_size", 512)
    chunk_overlap = cfg.get("chunk_overlap", 64)
    if key == "chunk_size":
        chunk_size = value
    elif key == "chunk_overlap":
        chunk_overlap = value
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")


def save_config(data_dir: Path, key: str, value: str) -> None:
    import json
    if key == "zot_command":
        _validate_zot_command(value)
    coerced = _coerce_config_value(key, value)
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(data_dir)
    if key in ("chunk_size", "chunk_overlap"):
        _validate_chunk_params(cfg, key, coerced)
    cfg[key] = coerced
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
    embedding_provider: str = "local"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_api_key: str = "not-needed"
    pdf_provider: str = "pymupdf"

    def __post_init__(self) -> None:
        saved = load_config(self.data_dir)
        for key, value in saved.items():
            if key in CONFIGURABLE_KEYS and hasattr(self, key):
                setattr(self, key, _coerce_config_value(key, value))

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def fts_db_path(self) -> Path:
        return self.data_dir / "fts.sqlite"
