"""合成数据适配器。

三条数据轨道（合成 / 公开 / 院内真实）共用同一套内部表结构，
每条轨道只写一个适配器。真实数据到位那天，新增 dqmds_adapter.py
把源字段映射到同样的 DataFrame 即可，上层零改动 —— 这是方案 §3.1 的抗风险设计。
"""
from __future__ import annotations

import pandas as pd

from .. import db
from ..config import config, path
from ..synth.well_generator import generate

TABLES = ("well_master", "geo_static", "prod_daily", "well_event")


def load(n_wells: int | None = None, seed: int | None = None) -> dict:
    cfg = config()["synth"]
    return generate(
        n_wells=n_wells or cfg["n_wells"],
        seed=seed if seed is not None else cfg["seed"],
        blocks=cfg["blocks"],
        layers=cfg["layers"],
    )


def ingest(n_wells: int | None = None, seed: int | None = None, reset: bool = True) -> dict:
    data = load(n_wells, seed)
    db.init_schema()
    if reset:
        with db.connect() as conn:
            for t in TABLES + ("lifecycle_label", "prediction", "model_run"):
                conn.execute(f"DELETE FROM {t}")
    counts = {t: db.write_df(data[t], t) for t in TABLES}
    # 真值只落文件、不进业务表 —— 真实数据没有真值，业务代码不许依赖它
    truth_path = path("data_dir") / "synth_truth.csv"
    data["truth"].to_csv(truth_path, index=False)
    counts["truth_file"] = str(truth_path)
    return counts


def read_truth() -> pd.DataFrame:
    p = path("data_dir") / "synth_truth.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
