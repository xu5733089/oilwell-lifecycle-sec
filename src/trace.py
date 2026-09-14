"""追溯字段：方案 §9.2 要求每个响应都带 model_version / label_def_version /
data_source / trace_id。这里是它们的唯一生成处。"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from . import db
from .config import label_def_version

MODEL_VERSION_FILE = "model_version"


def new_trace_id(prefix: str = "tr") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def input_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def envelope(
    data_source: str = "SYNTHETIC",
    model_version: str = "n/a",
    trace_id: Optional[str] = None,
) -> Dict[str, str]:
    """所有对外返回体都要 merge 这个信封。"""
    return {
        "model_version": model_version,
        "label_def_version": label_def_version(),
        "data_source": data_source,
        "trace_id": trace_id or new_trace_id(),
    }


def audit(trace_id: str, actor: str, action: str, payload: Any, status: str = "ok") -> None:
    import pandas as pd

    row = pd.DataFrame([{
        "trace_id": trace_id,
        "actor": actor,
        "action": action,
        "payload": json.dumps(payload, ensure_ascii=False, default=str)[:4000],
        "status": status,
        "created_at": now_iso(),
    }])
    try:
        db.write_df(row, "audit_log")
    except Exception:
        pass  # 留痕失败不能阻断主流程
