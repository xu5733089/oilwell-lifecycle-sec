"""空间建模：回归克里金（方案 §5.1）。

趋势项用梯度提升树吃协变量（埋深、层位、完井参数等），
残差用普通克里金按变差函数插值 —— 既有机器学习的拟合力，
又保留地质人员认可的变差函数分析，且天然给出预测方差。

纯 numpy/scipy 实现，不依赖 pykrige/gstools（内网常装不上）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from sklearn.ensemble import GradientBoostingRegressor

N_NEIGHBORS = 12


@dataclass
class Variogram:
    nugget: float
    sill: float
    rng: float          # 变程 range

    def gamma(self, h: np.ndarray) -> np.ndarray:
        """指数模型：γ(h) = c0 + c·(1 - exp(-3h/a))"""
        return self.nugget + (self.sill - self.nugget) * (1.0 - np.exp(-3.0 * h / max(self.rng, 1e-6)))


def experimental_variogram(coords: np.ndarray, values: np.ndarray, n_bins: int = 14):
    n = len(coords)
    i, j = np.triu_indices(n, k=1)
    h = np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
    g = 0.5 * (values[i] - values[j]) ** 2
    hmax = np.percentile(h, 60) or h.max()
    edges = np.linspace(0, hmax, n_bins + 1)
    hs, gs, ns = [], [], []
    for k in range(n_bins):
        m = (h >= edges[k]) & (h < edges[k + 1])
        if m.sum() >= 8:
            hs.append(float(h[m].mean()))
            gs.append(float(g[m].mean()))
            ns.append(int(m.sum()))
    return np.array(hs), np.array(gs), np.array(ns)


def fit_variogram(coords: np.ndarray, values: np.ndarray) -> Variogram:
    h, g, _ = experimental_variogram(coords, values)
    var = float(np.var(values)) or 1.0
    if len(h) < 4:
        return Variogram(nugget=var * 0.3, sill=var, rng=float(np.ptp(coords[:, 0]) / 3 + 1))
    x0 = [var * 0.2, var, float(h.max() / 2 + 1)]
    lo = [0.0, var * 1e-3, float(h.min() + 1e-6)]
    hi = [var * 0.95, var * 5.0, float(h.max() * 4)]
    res = least_squares(lambda p: Variogram(*p).gamma(h) - g, x0=x0, bounds=(lo, hi), max_nfev=2000)
    return Variogram(*[float(v) for v in res.x])


def _ordinary_kriging(coords, values, vg: Variogram, target: np.ndarray,
                      k: int = N_NEIGHBORS) -> Tuple[float, float]:
    d = np.hypot(coords[:, 0] - target[0], coords[:, 1] - target[1])
    idx = np.argsort(d)[:min(k, len(d))]
    c, v, dt = coords[idx], values[idx], d[idx]
    n = len(idx)
    H = np.hypot(c[:, 0][:, None] - c[:, 0][None, :], c[:, 1][:, None] - c[:, 1][None, :])
    G = vg.gamma(H)
    A = np.ones((n + 1, n + 1))
    A[:n, :n] = G
    A[n, n] = 0.0
    b = np.ones(n + 1)
    b[:n] = vg.gamma(dt)
    try:
        w = np.linalg.solve(A + np.eye(n + 1) * 1e-8, b)
    except np.linalg.LinAlgError:
        return float(v.mean()), float(np.sqrt(vg.sill))
    lam = w[:n]
    est = float(lam @ v)
    var = float(max(lam @ b[:n] + w[n], 0.0))
    return est, float(np.sqrt(var))


@dataclass
class RegressionKriging:
    """趋势(GBDT) + 残差(普通克里金)。predict 同时返回均值与标准差。"""
    covariates: Optional[list] = None
    trend: Optional[GradientBoostingRegressor] = None
    vg: Optional[Variogram] = None
    coords: Optional[np.ndarray] = None
    resid: Optional[np.ndarray] = None
    fallback_mean: float = 0.0
    fallback_std: float = 0.0

    def fit(self, coords: np.ndarray, values: np.ndarray,
            X: Optional[np.ndarray] = None) -> "RegressionKriging":
        values = np.asarray(values, float)
        self.fallback_mean, self.fallback_std = float(values.mean()), float(values.std())
        if X is not None and X.shape[1] > 0 and len(values) >= 30:
            self.trend = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                   learning_rate=0.05, random_state=0)
            self.trend.fit(np.nan_to_num(X), values)
            resid = values - self.trend.predict(np.nan_to_num(X))
        else:
            self.trend, resid = None, values - values.mean()
        self.coords, self.resid = np.asarray(coords, float), resid
        self.vg = fit_variogram(self.coords, resid)
        return self

    def predict(self, coords: np.ndarray, X: Optional[np.ndarray] = None):
        coords = np.atleast_2d(np.asarray(coords, float))
        base = (self.trend.predict(np.nan_to_num(X)) if self.trend is not None and X is not None
                else np.full(len(coords), self.fallback_mean))
        mean, std = [], []
        for i, t in enumerate(coords):
            e, s = _ordinary_kriging(self.coords, self.resid, self.vg, t)
            mean.append(base[i] + e)
            std.append(s)
        return np.asarray(mean), np.asarray(std)

    def variogram_params(self) -> dict:
        return dict(model="exponential", nugget=round(self.vg.nugget, 5),
                    sill=round(self.vg.sill, 5), range_m=round(self.vg.rng, 1))
