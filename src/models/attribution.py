"""特征归因（方案 §1.2「模型可解释性」）。

主方法：**精确 TreeSHAP**（src/models/treeshap.py，路径依赖口径，自研、零依赖、与穷举 Shapley 逐位对拍）。
  · 单井：基准值 + Σ 各特征贡献 = 模型输出，逐位可加，可以画成瀑布图；
  · 全局：对样本井逐个求 SHAP，按平均 |SHAP| 排序，配合特征取值画蜂群图。
保留的对照方法：特征消融（换成训练集中位数看预测变化）与置换重要性 —— 结论方向一般一致，但不满足可加性。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .treeshap import TreeExplainer

SHAP_METHOD = "精确 TreeSHAP（路径依赖口径，基准值 + 各特征贡献 = 模型输出）"
_EXPLAINERS: Dict[int, TreeExplainer] = {}


def explainer(model, target: str, quantile: float = 0.5) -> TreeExplainer:
    est = model.models[target][quantile]
    ex = _EXPLAINERS.get(id(est))
    if ex is None:
        ex = _EXPLAINERS[id(est)] = TreeExplainer(est)
    return ex


def local_shap(model, x_row: pd.Series, target: str, quantile: float = 0.5, top_k: int = 8) -> Dict:
    """单井 SHAP：前 top_k 个特征 + "其余特征"合计，保证瀑布图首尾闭合。"""
    cols = model.feature_cols
    x = x_row[cols].astype(float).to_numpy()[None, :]
    ex = explainer(model, target, quantile)
    phi = ex.shap_values(x)[0]
    pred = float(model.models[target][quantile].predict(pd.DataFrame(x, columns=cols))[0])
    order = np.argsort(-np.abs(phi))
    top = [dict(feature=cols[j], value=_safe(x_row.get(cols[j])), contrib=round(float(phi[j]), 4))
           for j in order[:top_k]]
    rest = float(phi[order[top_k:]].sum()) if len(order) > top_k else 0.0
    return dict(target=target, quantile=quantile, expected_value=round(float(ex.expected_value), 4),
                prediction=round(pred, 4), contributions=top, rest=round(rest, 4),
                n_rest=int(max(len(order) - top_k, 0)),
                additivity_gap=round(float(ex.expected_value + phi.sum() - pred), 8), method=SHAP_METHOD)


def global_shap(model, X: pd.DataFrame, top: int = 12, max_points: int = 360, seed: int = 0) -> Dict:
    """全局 SHAP 蜂群图数据：每个特征给 (SHAP 值, 特征取值在 5%–95% 分位间的归一化位置)。"""
    Xm = X[model.feature_cols].astype(float)
    sample = Xm.sample(min(len(Xm), max_points), random_state=seed) if len(Xm) > max_points else Xm
    arr = sample.to_numpy()
    out: Dict[str, Dict] = {}
    for tgt in model.models:
        ex = explainer(model, tgt)
        phi = ex.shap_values(arr)
        mean_abs = np.abs(phi).mean(axis=0)
        feats = []
        for j in np.argsort(-mean_abs)[:top]:
            v = arr[:, j]
            lo, hi = (np.nanpercentile(v, 5), np.nanpercentile(v, 95)) if np.isfinite(v).any() else (0.0, 1.0)
            vn = np.clip((v - lo) / (hi - lo if hi > lo else 1.0), 0.0, 1.0)
            feats.append(dict(feature=model.feature_cols[j], mean_abs=round(float(mean_abs[j]), 4),
                              points=[[round(float(phi[i, j]), 4), None if not np.isfinite(vn[i]) else round(float(vn[i]), 3)]
                                      for i in range(len(arr))]))
        out[tgt] = dict(expected_value=round(float(ex.expected_value), 4), n=int(len(arr)), features=feats,
                        method=SHAP_METHOD)
    return out


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
