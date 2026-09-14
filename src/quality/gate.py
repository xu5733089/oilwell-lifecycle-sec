"""数据质量门禁（方案 §3.4）。不过门禁的井进隔离区并出报告，不进训练集。"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

MAX_MISSING_RATE = 0.05      # 关键字段缺失率上限
MIN_CONTINUITY = 0.95        # 时间序列连续率下限
PHYS = {                     # 物理边界
    "oil_t": (0.0, 500.0),
    "water_m3": (0.0, 2000.0),
    "whp_mpa": (0.0, 60.0),
    "bhp_mpa": (0.0, 90.0),
    "water_cut": (0.0, 100.0),
    "gor": (0.0, 3000.0),
}
KEY_FIELDS = ["oil_t", "whp_mpa"]


def check_wells(prod: pd.DataFrame, master: pd.DataFrame,
                static: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    stat_wells = set(static["well_id"])
    for wid, g in prod.groupby("well_id"):
        g = g.sort_values("day_index")
        n = len(g)
        span = int(g["day_index"].max() - g["day_index"].min() + 1) if n else 0
        continuity = n / span if span else 0.0
        missing = {f: float(g[f].isna().mean()) for f in KEY_FIELDS if f in g}
        viol = 0
        for col, (lo, hi) in PHYS.items():
            if col in g:
                v = g[col].dropna()
                viol += int(((v < lo) | (v > hi)).sum())
        # 异常值：IQR + 3σ 双判，只标记不删除
        oil = g["oil_t"].dropna()
        if len(oil) > 20:
            q1, q3 = np.percentile(oil, [25, 75])
            iqr = q3 - q1
            out_iqr = int(((oil < q1 - 3 * iqr) | (oil > q3 + 3 * iqr)).sum())
        else:
            out_iqr = 0

        reasons = []
        if n < 60:
            reasons.append("历史不足60天")
        if continuity < MIN_CONTINUITY:
            reasons.append(f"连续率{continuity:.2f}<{MIN_CONTINUITY}")
        for f, m in missing.items():
            if m > MAX_MISSING_RATE:
                reasons.append(f"{f}缺失率{m:.2f}")
        if viol > 0:
            reasons.append(f"物理越界{viol}点")
        if wid not in stat_wells:
            reasons.append("缺静态参数")

        rows.append(dict(well_id=wid, n_days=n, continuity=round(continuity, 4),
                         missing_oil=round(missing.get("oil_t", 0.0), 4),
                         missing_whp=round(missing.get("whp_mpa", 0.0), 4),
                         phys_violations=viol, outliers_iqr=out_iqr,
                         passed=len(reasons) == 0,
                         reject_reason="; ".join(reasons)))
    return pd.DataFrame(rows)


def summary(report: pd.DataFrame) -> Dict:
    if report.empty:
        return dict(n_wells=0, n_passed=0, pass_rate=0.0, top_reasons={})
    reasons = (report.loc[~report.passed, "reject_reason"]
               .str.split("; ").explode().value_counts().head(5).to_dict())
    return dict(n_wells=int(len(report)), n_passed=int(report.passed.sum()),
                pass_rate=round(float(report.passed.mean()), 4), top_reasons=reasons)
