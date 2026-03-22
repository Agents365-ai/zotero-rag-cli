from __future__ import annotations

import hashlib
import json
from pathlib import Path

REGISTRY_FILENAME = "registry.json"


def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def save_registry(data_dir: Path, registry: dict[str, str]) -> None:
    (data_dir / REGISTRY_FILENAME).write_text(json.dumps(registry))


def load_registry(data_dir: Path) -> dict[str, str]:
    path = data_dir / REGISTRY_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())
