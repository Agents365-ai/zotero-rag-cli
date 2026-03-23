from __future__ import annotations

import hashlib
import json
from pathlib import Path

REGISTRY_FILENAME = "registry.json"


def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def save_registry(data_dir: Path, registry: dict[str, str]) -> None:
    import os
    import tempfile
    target = data_dir / REGISTRY_FILENAME
    fd, tmp_path = tempfile.mkstemp(dir=data_dir, suffix=".tmp")
    try:
        os.write(fd, json.dumps(registry).encode())
        os.close(fd)
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_registry(data_dir: Path) -> dict[str, str]:
    path = data_dir / REGISTRY_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Ensure all keys and values are strings
    return {str(k): str(v) for k, v in data.items()}
