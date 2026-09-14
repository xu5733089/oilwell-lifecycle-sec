"""特征归因（方案 §1.2「模型可解释性」）。

本仓库不依赖 shap 包（内网离线环境常装不上），改用
**特征消融归因**：把单个特征换成训练集中位数，看预测怎么变。
这是 leave-one-covariate-out 的局部版本，结论方向与 SHAP 一致，
成本 O(n_features) 次预测，毫秒级。

装上 shap 后可直接替换 local_attribution() 的实现，上层接口不变。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def local_attribution(model, x_row: pd.Series, target: str, quantile: float = 0.5,
                      top_k: int = 8) -> List[Dict]:
    est = model.models[target][quantile]
    cols = model.feature_cols
    base_x = pd.DataFrame([x_row[cols].astype(float)])
    base = float(est.predict(base_x)[0])

    contribs = []
    for c in cols:
        xm = base_x.copy()
        xm.loc[:, c] = model.background.get(c, np.nan)
        try:
            delta = base - float(est.predict(xm)[0])
        except Exception:
            continue
        contribs.append(dict(feature=c, value=_safe(x_row.get(c)),
                             contrib=round(float(delta), 4)))
    contribs.sort(key=lambda d: abs(d["contrib"]), reverse=True)
    return contribs[:top_k]


def global_importance(model, X: pd.DataFrame, y: pd.Series, target: str,
                      quantile: float = 0.5, n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    est = model.models[target][quantile]
    mask = y.notna().to_numpy()
    r = permutation_importance(est, X.loc[mask, model.feature_cols].astype(float),
                               y[mask].astype(float), n_repeats=n_repeats,
                               random_state=seed, scoring="neg_mean_absolute_error")
    return (pd.DataFrame({"feature": model.feature_cols,
                          "importance": r.importances_mean,
                          "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def _safe(v):
    try:
        return None if v is None or pd.isna(v) else round(float(v), 4)
    except Exception:
        return None
