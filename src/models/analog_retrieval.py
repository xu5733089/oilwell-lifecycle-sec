"""类比井检索（方案 §4.5）。

把工程师的"邻井类比"从经验做成可度量的检索：
用早期产量曲线形态 + 静态/完井参数找 Top-K 相似老井，
既作特征、又作解释、又作可核查的证据。

输出的话术是业务语言："该井与 A-12 相似度 0.91，A-12 在第 118 天达峰"。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CURVE_POINTS = 30
FEATURE_WEIGHT = 0.5          # 混合相似度里静态/完井特征的权重


def _resample_curve(g: pd.DataFrame, obs_days: int, n: int = CURVE_POINTS) -> np.ndarray:
    w = g[(g["day_index"] <= obs_days) & (g["hours_on"] > 0)].sort_values("day_index")
    if len(w) < 5:
        return np.full(n, np.nan)
    q = np.interp(np.linspace(1, obs_days, n), w["day_index"].to_numpy(float),
                  w["oil_t"].to_numpy(float))
    peak = np.nanmax(q)
    return q / peak if peak > 0 else q          # 归一化：比形态，不比绝对量级


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    cost = np.abs(a[:, None] - b[None, :])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


class AnalogIndex:
    """在训练井集合上建索引，对任意目标井返回 Top-K 类比井。"""

    def __init__(self, feature_cols: List[str], obs_days: int = 90):
        self.feature_cols = feature_cols
        self.obs_days = obs_days
        self.curves: Dict[str, np.ndarray] = {}
        self.F: Optional[np.ndarray] = None
        self.ids: List[str] = []
        self.mu = self.sd = None
        self.labels: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, prod: pd.DataFrame, labels: pd.DataFrame) -> "AnalogIndex":
        self.ids = X["well_id"].tolist()
        F = X[self.feature_cols].astype(float).to_numpy()
        self.mu, self.sd = np.nanmean(F, 0), np.nanstd(F, 0) + 1e-9
        self.F = (F - self.mu) / self.sd
        for wid, g in prod.groupby("well_id"):
            self.curves[wid] = _resample_curve(g, self.obs_days)
        self.labels = labels.set_index("well_id")
        return self

    def query(self, x_row: pd.Series, curve: np.ndarray, top_k: int = 5,
              exclude: Optional[str] = None) -> List[Dict]:
        f = (x_row[self.feature_cols].astype(float).to_numpy() - self.mu) / self.sd
        f = np.nan_to_num(f)
        Fd = np.linalg.norm(np.nan_to_num(self.F) - f, axis=1) / np.sqrt(len(self.feature_cols))

        out = []
        for i, wid in enumerate(self.ids):
            if exclude and wid == exclude:
                continue
            c = self.curves.get(wid)
            cd = _dtw(curve, c) if (c is not None and np.isfinite(c).all()
                                    and np.isfinite(curve).all()) else 1.0
            dist = FEATURE_WEIGHT * Fd[i] + (1 - FEATURE_WEIGHT) * cd * 3.0
            out.append((wid, dist))
        out.sort(key=lambda t: t[1])

        res = []
        for wid, dist in out[:top_k]:
            lab = self.labels.loc[wid] if wid in self.labels.index else None
            res.append(dict(
                well_id=wid,
                similarity=round(float(np.exp(-dist)), 3),
                t_peak_actual=None if lab is None else _f(lab.get("t_peak")),
                q_peak_actual=None if lab is None else _f(lab.get("q_peak")),
                p_peak_actual=None if lab is None else _f(lab.get("p_peak")),
                eur_actual=None if lab is None else _f(lab.get("eur"))))
        return res


def _f(v):
    return None if v is None or pd.isna(v) else round(float(v), 2)
