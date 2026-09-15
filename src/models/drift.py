"""输入漂移守卫：序列模型只在训练分布之内使用，超出时该井退回梯度提升。

为什么需要（诊断数据见 docs/dev-plan-phase3.md §6）：
  按区块留出评测时，被留出区块的井 100% 的邻井距离特征落在训练范围之外（最大尺度偏离中位 7.7，
  训练分布内的井约 1.0），区块本身也没见过。神经网络对这种输入做线性外推，EUR 误差翻倍；
  树模型分段常数、自然饱和，不受影响。融合权重在校准集上选 —— 校准集与训练同分布，看不见这种迁移，
  所以必须逐井判断，而不是换一种选权重的方法。

判定（只用训练集与校准集的**输入**，不看任何标签）：
  · 连续特征：score = max_j |x_j − 训练中位数_j| / (训练 P90_j − P10_j)，缺失值不计；
    阈值 = 校准集 score 的 99% 分位 —— 校准集与训练同分布，代表"正常井"的上沿；
  · 类别特征：区块 / 层系在训练集中从未出现，直接判为漂移。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

CATEGORICAL_PREFIXES = ("block_", "layer_")


@dataclass
class DriftGuard:
    cols: List[str] = field(default_factory=list)
    median: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale: np.ndarray = field(default_factory=lambda: np.zeros(0))
    seen: Dict[str, List[str]] = field(default_factory=dict)
    threshold: float = float("inf")
    quantile: float = 0.99

    def fit(self, X_train: pd.DataFrame, feature_cols: List[str]) -> "DriftGuard":
        self.cols = [c for c in feature_cols if not c.startswith(CATEGORICAL_PREFIXES)]
        V = X_train.reindex(columns=self.cols).astype(float)
        self.median = np.nan_to_num(V.median().to_numpy(), nan=0.0)
        spread = (V.quantile(0.9) - V.quantile(0.1)).to_numpy()
        sd = V.std().to_numpy()
        # 训练集里几乎恒定的列：退到标准差，再退到 1，避免把微小差异放大成"漂移"
        self.scale = np.where(np.isfinite(spread) & (spread > 1e-9), spread,
                              np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0))
        self.seen = {}
        for p in CATEGORICAL_PREFIXES:
            cats = [c for c in feature_cols if c.startswith(p)]
            if cats:
                A = X_train.reindex(columns=cats).fillna(0.0).astype(float)
                self.seen[p] = [c for c in cats if bool((A[c] > 0.5).any())]
        return self

    def _parts(self, X: pd.DataFrame):
        n = len(X)
        if self.cols:
            V = X.reindex(columns=self.cols).astype(float).to_numpy()
            Z = np.abs(V - self.median) / self.scale
            Z = np.where(np.isfinite(Z), Z, 0.0)
            score = Z.max(axis=1)
            top = [self.cols[i] for i in Z.argmax(axis=1)]
        else:
            score, top = np.zeros(n), [None] * n
        novel = np.zeros(n, dtype=bool)
        for p, seen in self.seen.items():
            if not any(c.startswith(p) for c in X.columns):
                continue
            active = (X.reindex(columns=seen).fillna(0.0).astype(float).to_numpy() > 0.5) if seen \
                else np.zeros((n, 1), dtype=bool)
            novel |= ~active.any(axis=1)
        return score, top, novel

    def calibrate(self, X_calib: pd.DataFrame, quantile: float | None = None) -> "DriftGuard":
        self.quantile = quantile or self.quantile
        score, _, _ = self._parts(X_calib)
        self.threshold = float(np.quantile(score, self.quantile)) if len(score) else float("inf")
        return self

    def assess(self, X: pd.DataFrame) -> pd.DataFrame:
        score, top, novel = self._parts(X)
        return pd.DataFrame(dict(score=score, top_feature=top, novel_category=novel,
                                 flagged=novel | (score > self.threshold)), index=X.index)
