"""模型存取与版本。model_version 会随预测结果一路带到 API 响应里（§9.2）。"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..config import path


def make_version(prefix: str = "gbdt-q") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y.%m.%d.%H%M')}"


def save(bundle: Dict[str, Any], version: str) -> Path:
    p = path("artifacts_dir") / f"model_{version}.pkl"
    with open(p, "wb") as f:
        pickle.dump(bundle, f)
    meta = {k: v for k, v in bundle.get("meta", {}).items()}
    (path("artifacts_dir") / f"model_{version}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (path("artifacts_dir") / "LATEST").write_text(version, encoding="utf-8")
    return p


def load(version: str | None = None) -> Dict[str, Any]:
    d = path("artifacts_dir")
    if version is None:
        latest = d / "LATEST"
        if not latest.exists():
            raise FileNotFoundError("尚未训练模型，请先执行：python -m src.cli train")
        version = latest.read_text(encoding="utf-8").strip()
    with open(d / f"model_{version}.pkl", "rb") as f:
        return pickle.load(f)


def latest_version() -> str | None:
    p = path("artifacts_dir") / "LATEST"
    return p.read_text(encoding="utf-8").strip() if p.exists() else None
