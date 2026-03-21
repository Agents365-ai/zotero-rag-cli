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
    meta = {
        "model_name": model_name,
        "last_indexed": datetime.now(timezone.utc).isoformat(),
        "item_count": item_count,
    }
    (data_dir / META_FILENAME).write_text(json.dumps(meta, indent=2))


def load_metadata(data_dir: Path) -> IndexMetadata | None:
    path = data_dir / META_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return IndexMetadata(
        model_name=data["model_name"],
        last_indexed=data["last_indexed"],
        item_count=data["item_count"],
    )
