"""保形校准 CQR（方案 §4.4）。

分位数回归给出的区间通常不校准：名义 80% 的区间实际可能只覆盖 60%。
CQR 用一个留出校准集把区间整体放宽/收窄，使经验覆盖率逼近名义值。

这一步让"区间覆盖率"成为可写进成果报告的硬指标；
也是 P90 敢对应 SEC"合理确定性"口径的前提（§6.1）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class Conformal:
    """归一化 CQR。

    默认用 **归一化** 一致性分数（除以模型自己给的区间宽度），而不是绝对残差：
    EUR 这类目标跨越 500~36000 t 两个数量级，异方差极强，
    用一个统一的绝对修正量（"所有井的区间都加宽 800 t"）在小井上荒谬、在大井上不够。
    归一化后修正量是个倍数，随预测难度自适应。

    mode="absolute" 保留经典 CQR，便于对照。
    """
    alpha: float = 0.2                       # 名义覆盖率 = 1 - alpha
    mode: str = "normalized"                 # normalized | absolute
    delta: Dict[str, float] = field(default_factory=dict)
    n_calib: Dict[str, int] = field(default_factory=dict)

    def fit_target(self, target: str, y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
        ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        y, lo, hi = y[ok], lo[ok], hi[ok]
        if len(y) < 10:
            self.delta[target], self.n_calib[target] = 0.0, int(len(y))
            return 0.0
        raw = np.maximum(lo - y, y - hi)                 # CQR 一致性分数
        if self.mode == "normalized":
            width = np.maximum(hi - lo, 1e-9)
            score = raw / width
        else:
            score = raw
        n = len(score)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        d = float(np.quantile(score, level, method="higher"))
        self.delta[target], self.n_calib[target] = d, n
        return d

    def apply(self, target: str, lo: np.ndarray, hi: np.ndarray):
        d = self.delta.get(target, 0.0)
        if self.mode == "normalized":
            w = np.maximum(hi - lo, 1e-9)
            return lo - d * w, hi + d * w
        return lo - d, hi + d

    @staticmethod
    def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
        ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if ok.sum() == 0:
            return float("nan")
        return float(((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])).mean())

    @staticmethod
    def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
        ok = np.isfinite(lo) & np.isfinite(hi)
        return float(np.mean(hi[ok] - lo[ok])) if ok.sum() else float("nan")


def calibrate(preds: Dict[str, pd.DataFrame], y: pd.DataFrame, alpha: float,
              lo_col: str = "p10", hi_col: str = "p90") -> Conformal:
    c = Conformal(alpha=alpha)
    for tgt, df in preds.items():
        if tgt in y:
            c.fit_target(tgt, y[tgt].to_numpy(float),
                         df[lo_col].to_numpy(float), df[hi_col].to_numpy(float))
    return c
