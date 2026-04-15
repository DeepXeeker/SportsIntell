from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def setup_logger(log_dir: str | Path) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_dir))
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
        file_handler = logging.FileHandler(Path(log_dir) / "run.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def dump_json(obj: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
