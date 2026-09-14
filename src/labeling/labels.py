"""标签提取（方案 §3.3）。

口径全部来自 conf/label_def.yaml，改口径不改代码；
产出带 label_def_version，历史标签并存可回溯。

达峰识别的三道防线（措施井伪峰是这里最大的坑）：
  1) 投产 min_days_after_start 天内的峰值不采信
  2) 措施作业后 exclude_days_after_workover 天内的候选峰剔除
  3) 峰后需连续 confirm_decline_months 个月下降才确认，否则顺延到下一候选
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import label_def, label_def_version
from ..reserves import dca


def _oil_break(g: pd.DataFrame, cfg: Dict) -> Optional[float]:
    thr, need = cfg["threshold"], int(cfg["consecutive_days"])
    v = g[g["hours_on"] > 0]
    ok = (v[cfg["metric"]] >= thr).to_numpy()
    days = v["day_index"].to_numpy()
    run = 0
    for i, flag in enumerate(ok):
        run = run + 1 if flag else 0
        if run >= need:
            return float(days[i - need + 1])
    return None


def _smoothed(g: pd.DataFrame, window: int) -> pd.Series:
    s = g.set_index("day_index")["oil_t"].astype(float)
    s = s.where(g.set_index("day_index")["hours_on"] > 0)      # 停井日不参与
    full = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
    return full.rolling(window, min_periods=max(2, window // 2), center=True).mean()


def _peak(g: pd.DataFrame, events: pd.DataFrame, cfg: Dict):
    sm = _smoothed(g, int(cfg["smooth_window_days"]))
    if sm.dropna().empty:
        return None, None, "平滑序列为空"
    cand = sm[sm.index >= int(cfg["min_days_after_start"])].dropna()
    if cand.empty:
        return None, None, "无有效候选峰"

    excl = int(cfg["exclude_days_after_workover"])
    wo_days = events["day_index"].dropna().astype(int).tolist() if not events.empty else []
    blocked = np.zeros(len(cand), dtype=bool)
    for d0 in wo_days:
        blocked |= (cand.index >= d0) & (cand.index <= d0 + excl)
    usable = cand[~blocked]
    if usable.empty:
        usable = cand           # 全被措施窗口覆盖时退回全体，并在质量位标记

    need_m = int(cfg["confirm_decline_months"])
    for day in usable.sort_values(ascending=False).index[:12]:
        qp = float(sm.loc[day])
        ok = True
        for k in range(1, need_m + 1):
            lo, hi = day + 30 * (k - 1), day + 30 * k
            seg = sm[(sm.index > lo) & (sm.index <= hi)].dropna()
            if len(seg) < 8 or float(seg.mean()) >= qp * (1.0 - 0.02 * k):
                ok = False
                break
        if ok:
            return float(day), qp, ""
    day = float(usable.idxmax())
    return day, float(sm.loc[day]), "峰后下降未确认"


def _peak_pressure(g: pd.DataFrame, peak_day: float, cfg: Dict) -> Optional[float]:
    col = cfg["source"]
    if col not in g:
        return None
    w = g[(g["day_index"] >= peak_day - 1) & (g["day_index"] <= peak_day + 1)][col].dropna()
    if w.empty:
        w = g[(g["day_index"] >= peak_day - 3) & (g["day_index"] <= peak_day + 3)][col].dropna()
    return float(w.mean()) if not w.empty else None


def extract_one(wid: str, g: pd.DataFrame, events: pd.DataFrame,
                q_econ: float = 2.0) -> Dict:
    ld = label_def()
    g = g.sort_values("day_index")
    out: Dict = dict(well_id=wid, label_def_version=label_def_version(),
                     label_quality="ok", reject_reason="")

    out["t_oil_break"] = _oil_break(g, ld["oil_break"])
    peak_day, q_peak, warn = _peak(g, events, ld["peak"])
    out["t_peak"], out["q_peak"] = peak_day, q_peak
    if warn:
        out["label_quality"], out["reject_reason"] = "suspect", warn
    out["p_peak"] = _peak_pressure(g, peak_day, ld["peak_pressure"]) if peak_day else None
    out["cum_360"] = float(g[g["day_index"] <= 360]["oil_t"].sum())

    # EUR：峰后拟合递减 + 峰前累产，历史不足则不出标签。
    # v2 起为"自然递减"口径：有措施的井只用首次措施之前的数据（累产也截到措施前）。
    # 早期数据预测不了几年后才做的措施；措施增油由 SEC 单元构成评估单列（src/sec/composition.py）。
    out.update(dca_model=None, di=None, b=None, d_min=None, eur=None)
    months = (g["day_index"].max() - g["day_index"].min()) / 30.4
    gg = g
    if ld["eur"].get("exclude_after_workover") and not events.empty:
        days = events["day_index"].dropna()
        if len(days):
            gg = g[g["day_index"] < int(days.min())]
    if peak_day and months >= float(ld["eur"]["min_history_months"]):
        post = gg[(gg["day_index"] >= peak_day) & (gg["hours_on"] > 0) & (gg["oil_t"] > 0)]
        if len(post) >= 60:
            try:
                tm = post["day_index"].to_numpy() / 30.4
                qq = post["oil_t"].to_numpy()
                f = dca.fit_best(tm, qq)
                cum = float(gg["oil_t"].sum())
                # EUR = 已累产 + 剩余可采，与 services.fit_dca 同一口径
                e = dca.eur(f, q_econ, t_start_month=float(tm.max() - tm.min()))
                out.update(dca_model=f.model, di=f.di, b=f.b, d_min=f.d_min,
                           eur=float(cum + e["remaining"]))
            except Exception as exc:
                out["label_quality"] = "suspect"
                out["reject_reason"] = (out["reject_reason"] + "; DCA失败:" + str(exc)[:60]).strip("; ")
    if out["t_peak"] is None or out["q_peak"] is None:
        out["label_quality"], out["reject_reason"] = "rejected", "无法确定达峰"
    return out


def extract_all(prod: pd.DataFrame, events: pd.DataFrame, q_econ: float = 2.0) -> pd.DataFrame:
    ev_by_well = dict(tuple(events.groupby("well_id"))) if not events.empty else {}
    empty = events.iloc[0:0] if not events.empty else pd.DataFrame(columns=["well_id", "day_index"])
    rows: List[Dict] = []
    for wid, g in prod.groupby("well_id"):
        rows.append(extract_one(wid, g, ev_by_well.get(wid, empty), q_econ))
    return pd.DataFrame(rows)
