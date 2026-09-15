"""模型族融合：梯度提升（早期统计特征） × 序列模型（逐日曲线）。

两个模型看的是同一口井的不同侧面，误差来源不同，融合往往比任何一个单独都稳。
但"融合一定更好"不能预设 —— 每个目标的融合权重在**校准集的 A 半**上按分位数损失选出，
可能选到 0（序列模型对这个目标没帮助）；保形校准只用 **B 半**，选权重与校准不用同一批井。
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TAUS = {"p10": 0.1, "p50": 0.5, "p90": 0.9}


def pinball_loss(y: np.ndarray, pred: pd.DataFrame) -> float:
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() == 0:
        return float("nan")
    tot = 0.0
    for col, tau in TAUS.items():
        e = y[ok] - pred[col].to_numpy(float)[ok]
        tot += float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))
    return tot / len(TAUS)


def mix(pg: pd.DataFrame, ps: pd.DataFrame, w: float) -> pd.DataFrame:
    vals = np.sort((1.0 - w) * pg[list(TAUS)].to_numpy(float) + w * ps[list(TAUS)].to_numpy(float), axis=1)
    return pd.DataFrame(vals, index=pg.index, columns=list(TAUS))


def choose_weights(Y: pd.DataFrame, pg: Dict[str, pd.DataFrame], ps: Dict[str, pd.DataFrame],
                   targets: Iterable[str], grid=GRID) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for t in targets:
        if t not in pg or t not in ps or t not in Y:
            continue
        y = Y[t].to_numpy(float)
        losses = {w: pinball_loss(y, mix(pg[t], ps[t], w)) for w in grid}
        best = min(losses, key=lambda w: (losses[w], w))        # 平手时取更小的序列权重（更简单的模型）
        out[t] = dict(weight=float(best), losses={str(k): round(v, 5) for k, v in losses.items()})
    return out


def blend(pg: Dict[str, pd.DataFrame], ps: Optional[Dict[str, pd.DataFrame]],
          weights: Dict[str, float]) -> Dict[str, pd.DataFrame]:
    if not ps:
        return pg
    return {t: (mix(df, ps[t], float(weights.get(t, 0.0))) if t in ps else df) for t, df in pg.items()}
