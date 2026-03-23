from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

META_FILENAME = "meta.json"


@dataclass
class IndexMetadata:
    model_name: str
    last_indexed: str
    item_count: int


def save_metadata(data_dir: Path, model_name: str, item_count: int) -> None:
    import os
    import tempfile
    meta = {
        "model_name": model_name,
        "last_indexed": datetime.now(timezone.utc).isoformat(),
        "item_count": item_count,
    }
    target = data_dir / META_FILENAME
    fd, tmp_path = tempfile.mkstemp(dir=data_dir, suffix=".tmp")
    try:
        os.write(fd, json.dumps(meta, indent=2).encode())
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


def load_metadata(data_dir: Path) -> IndexMetadata | None:
    path = data_dir / META_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return IndexMetadata(
            model_name=data["model_name"],
            last_indexed=data["last_indexed"],
            item_count=data["item_count"],
        )
    except (json.JSONDecodeError, KeyError, OSError):
        return None
