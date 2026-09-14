"""主力预测模型：分位数梯度提升（方案 §4.2 层级 B）。

选型理由（写进代码注释，答辩时不用回忆）：
  样本量在 10^2~10^3 口井、特征是"短序列统计量 + 异构表格"，
  树模型在这个规模上通常优于深度序列模型；训练秒级，可反复试口径；
  且天然可做特征归因，直接满足比赛的"模型可解释性"硬要求。

默认用 sklearn 的 HistGradientBoostingRegressor(loss="quantile")，零额外依赖；
装了 lightgbm 会自动切换（通常更快更准），接口不变。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:                                   # pragma: no cover - 取决于环境
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False


def _make(quantile: float, seed: int):
    if HAS_LGB:
        return lgb.LGBMRegressor(objective="quantile", alpha=quantile, n_estimators=400,
                                 learning_rate=0.05, num_leaves=15, min_child_samples=12,
                                 subsample=0.9, colsample_bytree=0.8, random_state=seed,
                                 verbose=-1)
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile,
                                         max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=15, min_samples_leaf=8,
                                         l2_regularization=1.0, random_state=seed)


@dataclass
class QuantileModel:
    targets: List[str]
    quantiles: List[float]
    feature_cols: List[str]
    seed: int = 42
    models: Dict[str, Dict[float, object]] = field(default_factory=dict)
    background: Dict[str, float] = field(default_factory=dict)   # 归因用的基线取值
    n_train: Dict[str, int] = field(default_factory=dict)

    # ---------- 训练 ----------
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "QuantileModel":
        Xm = X[self.feature_cols].astype(float)
        self.background = Xm.median(numeric_only=True).to_dict()
        for tgt in self.targets:
            if tgt not in y:
                continue
            mask = y[tgt].notna().to_numpy()
            if mask.sum() < 25:
                continue
            xs, ys = Xm[mask], y.loc[mask, tgt].astype(float)
            self.models[tgt] = {}
            self.n_train[tgt] = int(mask.sum())
            for q in self.quantiles:
                m = _make(q, self.seed)
                m.fit(xs, ys)
                self.models[tgt][q] = m
        return self

    # ---------- 预测 ----------
    def predict(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        Xm = X[self.feature_cols].astype(float)
        out: Dict[str, pd.DataFrame] = {}
        for tgt, per_q in self.models.items():
            cols = {f"p{int(q * 100)}": per_q[q].predict(Xm) for q in self.quantiles}
            df = pd.DataFrame(cols, index=X.index)
            # 分位数交叉是分位回归的常见毛病，强制单调
            ordered = np.sort(df.to_numpy(), axis=1)
            df = pd.DataFrame(ordered, index=X.index, columns=sorted(df.columns,
                              key=lambda c: int(c[1:])))
            out[tgt] = df
        return out

    def predict_point(self, X: pd.DataFrame, target: str, quantile: float = 0.5) -> np.ndarray:
        return self.models[target][quantile].predict(X[self.feature_cols].astype(float))

    @property
    def backend(self) -> str:
        return "lightgbm" if HAS_LGB else "sklearn-hgb"
