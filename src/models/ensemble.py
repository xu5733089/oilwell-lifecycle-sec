"""模型族融合：梯度提升（早期统计特征） × 序列模型（逐日曲线）。

两个模型看的是同一口井的不同侧面，误差来源不同，融合往往比任何一个单独都稳。但"融合一定更好"不能预设：

  · 权重：每个目标在校准集上按分位数损失从网格里选，可能选到 0；
  · 交叉拟合：每口校准井的一致性分数用"另一半校准井选出的权重"计算 —— 全部校准井都能参与保形校准，
    而没有一口井的标签参与决定它自己用的权重（旧做法是 A 半选权重、B 半校准，保形只剩一半样本）；
  · 漂移守卫：输入超出序列模型训练分布的井（src/models/drift.py）逐井把序列权重置 0，退回梯度提升。
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TAUS = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
COLS = list(TAUS)


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


def mix(pg: pd.DataFrame, ps: pd.DataFrame, w) -> pd.DataFrame:
    """w 可以是标量，也可以是逐行权重（漂移守卫把个别井的权重置 0）。"""
    w = np.asarray(w, float)
    if w.ndim:
        w = w[:, None]
    vals = np.sort((1.0 - w) * pg[COLS].to_numpy(float) + w * ps[COLS].to_numpy(float), axis=1)
    return pd.DataFrame(vals, index=pg.index, columns=COLS)


def choose_weights(Y: pd.DataFrame, pg: Dict[str, pd.DataFrame], ps: Dict[str, pd.DataFrame],
                   targets: Iterable[str], grid=GRID) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for t in targets:
        if t not in pg or t not in ps or t not in Y:
            continue
        y = Y[t].to_numpy(float)
        losses = {w: pinball_loss(y, mix(pg[t], ps[t], w)) for w in grid}
        finite = {w: v for w, v in losses.items() if np.isfinite(v)}
        best = min(finite, key=lambda w: (finite[w], w)) if finite else 0.0   # 平手取更小的序列权重
        out[t] = dict(weight=float(best), n=int(np.isfinite(y).sum()),
                      losses={str(k): (round(v, 5) if np.isfinite(v) else None) for k, v in losses.items()})
    return out


def blend(pg: Dict[str, pd.DataFrame], ps: Optional[Dict[str, pd.DataFrame]],
          weights: Dict[str, float], off: Optional[Sequence[bool]] = None) -> Dict[str, pd.DataFrame]:
    if not ps:
        return pg
    out = {}
    for t, df in pg.items():
        if t not in ps:
            out[t] = df
            continue
        w = float(weights.get(t, 0.0))
        out[t] = mix(df, ps[t], w if off is None else np.where(np.asarray(off, bool), 0.0, w))
    return out


def _sub(d: Dict[str, pd.DataFrame], rows) -> Dict[str, pd.DataFrame]:
    return {t: v.loc[rows] for t, v in d.items()}


def select_and_crossfit(Y: pd.DataFrame, pg: Dict[str, pd.DataFrame], ps: Dict[str, pd.DataFrame],
                        targets: Iterable[str], idx_a, idx_b, off: Optional[Sequence[bool]] = None,
                        grid=GRID) -> Tuple[Dict[str, Dict], Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    """返回 (线上权重选择, 交叉拟合的校准集融合预测, 两折各自的权重)。

    线上权重在全部未漂移的校准井上选；A 折的井用 B 折选出的权重融合，B 折反之。
    被漂移守卫标记的井不参与选权重，自身融合权重恒为 0。
    """
    targets = list(targets)
    flag = pd.Series(np.zeros(len(Y), bool) if off is None else np.asarray(off, bool), index=Y.index)
    keep = Y.index[~flag.to_numpy()]
    deployed = choose_weights(Y.loc[keep], _sub(pg, keep), _sub(ps, keep), targets, grid)
    parts, folds = [], {}
    for name, rows, other in (("a", idx_a, idx_b), ("b", idx_b, idx_a)):
        other_ok = [r for r in other if not flag.loc[r]]
        sel = choose_weights(Y.loc[other_ok], _sub(pg, other_ok), _sub(ps, other_ok), targets, grid)
        w = {t: v["weight"] for t, v in sel.items()}
        folds[name] = w
        parts.append(blend(_sub(pg, rows), _sub(ps, rows), w, off=flag.loc[rows].to_numpy()))
    cross = {t: pd.concat([p[t] for p in parts]).loc[Y.index] for t in pg}
    return deployed, cross, folds
