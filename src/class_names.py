from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .config import DEFAULT_CLASS_NAMES_PATH


def save_class_names(class_names: List[str], path: Path = DEFAULT_CLASS_NAMES_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_names, indent=2))
    return path


def load_class_names(path: Path = DEFAULT_CLASS_NAMES_PATH) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Class names file not found at {path}. "
            "Train first or pass a valid --class-names path."
        )

    data = json.loads(path.read_text())
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"Invalid class names format in {path}")

    return data
