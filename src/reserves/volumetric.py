"""容积法储量与蒙特卡洛不确定性（方案 §5.1）。

    N = A · h · φ · So · ρo / Boi

难点从来不在公式，而在于井点参数怎么外推到面上、不确定性怎么给。
外推见 spatial.py；不确定性用蒙特卡洛对各参数分布抽样，输出 P10/P50/P90。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

RHO_OIL_T_M3 = 0.85
BO_DEFAULT = 1.15
HALF_SPACING_M = 150.0
VERTICAL_RADIUS_M = 200.0


def drainage_area_m2(lateral_length: float, half_spacing: float = HALF_SPACING_M) -> float:
    """井控泄油面积。水平井按段长×两侧半宽，直井按圆形泄油半径。"""
    if lateral_length and lateral_length > 0:
        return float(lateral_length * 2.0 * half_spacing)
    return float(np.pi * VERTICAL_RADIUS_M ** 2)


def ooip_t(area_m2: float, net_pay_m: float, porosity_pct: float, so_pct: float,
           rho: float = RHO_OIL_T_M3, bo: float = BO_DEFAULT) -> float:
    pv = area_m2 * net_pay_m * (porosity_pct / 100.0) * (so_pct / 100.0)
    return float(pv * rho / bo)


@dataclass
class ParamSpec:
    """单个参数的不确定性描述。cv 为变异系数，来自空间建模的预测标准差或经验值。"""
    mean: float
    cv: float = 0.15
    dist: str = "lognormal"      # lognormal | normal | triangular
    low: Optional[float] = None
    high: Optional[float] = None

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.dist == "normal":
            v = rng.normal(self.mean, self.mean * self.cv, n)
        elif self.dist == "triangular":
            lo = self.low if self.low is not None else self.mean * (1 - 2 * self.cv)
            hi = self.high if self.high is not None else self.mean * (1 + 2 * self.cv)
            v = rng.triangular(lo, self.mean, hi, n)
        else:
            sigma = np.sqrt(np.log(1 + self.cv ** 2))
            v = rng.lognormal(np.log(max(self.mean, 1e-9)) - sigma ** 2 / 2, sigma, n)
        lo = self.low if self.low is not None else 1e-9
        hi = self.high if self.high is not None else np.inf
        return np.clip(v, lo, hi)


def monte_carlo(area: ParamSpec, net_pay: ParamSpec, porosity: ParamSpec,
                so: ParamSpec, rf: Optional[ParamSpec] = None,
                n: int = 10000, seed: int = 0,
                bo: float = BO_DEFAULT, rho: float = RHO_OIL_T_M3) -> Dict:
    """对各参数抽样，输出 OOIP（及可采储量）的分位数。

    口径见 dca.CONVENTION：p10/p50/p90 是分位数本身（p10 为小值），
    SEC 已证实储量取 low_estimate（= p10）。
    """
    rng = np.random.default_rng(seed)
    A = area.sample(rng, n)
    h = net_pay.sample(rng, n)
    phi = porosity.sample(rng, n)
    s = so.sample(rng, n)
    ooip = A * h * (phi / 100.0) * (s / 100.0) * rho / bo

    out = {
        "ooip": _pcts(ooip),
        "inputs": dict(area_mean=area.mean, net_pay_mean=net_pay.mean,
                       porosity_mean=porosity.mean, so_mean=so.mean,
                       bo=bo, rho=rho, n_samples=n),
    }
    if rf is not None:
        recov = ooip * (rf.sample(rng, n) / 100.0)
        out["recoverable"] = _pcts(recov)
        out["inputs"]["rf_mean_pct"] = rf.mean
    return out


def _pcts(v: np.ndarray) -> Dict[str, float]:
    """口径与 dca 保持一致：p10/p50/p90 是分位数本身，低/高估计另给别名。"""
    return dict(p10=float(np.percentile(v, 10)), p50=float(np.percentile(v, 50)),
                p90=float(np.percentile(v, 90)),
                low_estimate=float(np.percentile(v, 10)),
                high_estimate=float(np.percentile(v, 90)),
                mean=float(v.mean()), std=float(v.std()))


def well_ooip(master_row: pd.Series, static_row: pd.Series, cv: float = 0.18,
              n: int = 10000, seed: int = 0) -> Dict:
    """单井容积法储量的一站式封装。"""
    area = drainage_area_m2(float(master_row.get("lateral_length") or 0.0))
    return monte_carlo(
        area=ParamSpec(area, cv=0.25),
        net_pay=ParamSpec(float(static_row["net_pay_m"]), cv=cv),
        porosity=ParamSpec(float(static_row["porosity_pct"]), cv=cv * 0.8, high=35.0),
        so=ParamSpec(float(static_row["so_pct"]), cv=cv * 0.6, high=95.0),
        n=n, seed=seed)
