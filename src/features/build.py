"""特征工程（方案 §4.3）。

铁律：只允许使用投产后前 obs_days 天的动态数据 + 静态/完井参数。
任何越过观测窗口的信息都是泄漏 —— 因为上线时新井就只有这些。

空间邻井特征用的是"邻井的标签"，因此必须只从训练集里取邻居（ref_labels），
否则验证集会通过邻居标签间接看到自己的答案。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SPATIAL_K = 5


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def early_window_features(g: pd.DataFrame, obs_days: int) -> Dict[str, float]:
    w = g[g["day_index"] <= obs_days].sort_values("day_index")
    on = w[w["hours_on"] > 0]
    f: Dict[str, float] = {}
    if on.empty:
        return {k: np.nan for k in EARLY_KEYS}

    d = on["day_index"].to_numpy(float)
    q = on["oil_t"].to_numpy(float)
    whp = on["whp_mpa"].to_numpy(float)

    for k in (30, 60, 90):
        if obs_days >= k:
            f[f"cum_{k}d"] = float(w[w["day_index"] <= k]["oil_t"].sum())
        else:
            f[f"cum_{k}d"] = np.nan
    f["cum_obs"] = float(w["oil_t"].sum())
    f["q_max_obs"] = float(np.nanmax(q))
    f["q_mean_obs"] = float(np.nanmean(q))
    f["q_last14_mean"] = float(np.nanmean(q[d >= d.max() - 14])) if len(d) else np.nan
    f["q_cv"] = float(np.nanstd(q) / max(np.nanmean(q), 1e-6))
    f["uptime"] = float(on["hours_on"].mean() / 24.0)
    f["n_obs_days"] = float(len(on))

    # 上升段斜率与曲率（对数产量对时间）
    ramp = (q > 0.2 * f["q_max_obs"])
    f["ramp_slope"] = _slope(d[ramp], np.log(np.maximum(q[ramp], 1e-3)))
    half = d[q >= 0.5 * f["q_max_obs"]]
    f["days_to_half_max"] = float(half.min()) if len(half) else np.nan
    f["days_to_max_obs"] = float(d[int(np.nanargmax(q))])
    f["q_max_is_at_edge"] = float(f["days_to_max_obs"] >= d.max() - 3)   # 峰可能还没到

    # 压力：压降速率与生产压差代理
    f["whp_first"] = float(np.nanmean(whp[:5])) if len(whp) else np.nan
    f["whp_last"] = float(np.nanmean(whp[-5:])) if len(whp) else np.nan
    f["whp_drop"] = f["whp_first"] - f["whp_last"]
    f["dP_dt"] = _slope(d, whp)
    dd = max(f["whp_drop"], 1e-3)
    f["pi_proxy"] = f["cum_obs"] / dd                      # 采液指数代理 q/Δp
    f["whp_cv"] = float(np.nanstd(whp) / max(np.nanmean(whp), 1e-6))

    f["wc_slope"] = _slope(d, on["water_cut"].to_numpy(float))
    f["wc_last"] = float(np.nanmean(on["water_cut"].to_numpy(float)[-5:]))
    f["gor_mean"] = float(np.nanmean(on["gor"].to_numpy(float)))
    return f


EARLY_KEYS = ["cum_30d", "cum_60d", "cum_90d", "cum_obs", "q_max_obs", "q_mean_obs",
              "q_last14_mean", "q_cv", "uptime", "n_obs_days", "ramp_slope",
              "days_to_half_max", "days_to_max_obs", "q_max_is_at_edge", "whp_first",
              "whp_last", "whp_drop", "dP_dt", "pi_proxy", "whp_cv", "wc_slope",
              "wc_last", "gor_mean"]

STATIC_KEYS = ["porosity_pct", "so_pct", "net_pay_m", "toc_pct", "sweet_spot_idx",
               "brittleness", "pressure_coef", "temp_c", "log_perm"]

COMPLETION_KEYS = ["tvd", "lateral_length", "stage_count", "proppant_t", "frac_fluid_m3",
                   "proppant_per_stage", "is_horizontal", "kh_proxy"]

SPATIAL_KEYS = ["nb_dist_min", "nb_dist_mean", "nb_t_peak_med", "nb_q_peak_med",
                "nb_p_peak_med", "nb_t_peak_iqr", "nb_eur_med", "nb_count"]


def _spatial(master: pd.DataFrame, ref_labels: Optional[pd.DataFrame], k: int) -> pd.DataFrame:
    cols = {c: np.nan for c in SPATIAL_KEYS}
    if ref_labels is None or ref_labels.empty:
        out = master[["well_id"]].copy()
        for c, v in cols.items():
            out[c] = v
        return out

    master = master.copy()
    for c in ("x_off", "y_off"):
        master[c] = pd.to_numeric(master[c], errors="coerce")
    ref = master.merge(ref_labels, on="well_id", how="inner")
    rows: List[Dict] = []
    for _, w in master.iterrows():
        pool = ref[(ref["layer"] == w["layer"]) & (ref["well_id"] != w["well_id"])]
        if len(pool) < 2:
            pool = ref[ref["well_id"] != w["well_id"]]
        if pool.empty:
            rows.append(dict(well_id=w["well_id"], **cols))
            continue
        dist = np.hypot(pool["x_off"] - w["x_off"], pool["y_off"] - w["y_off"]).to_numpy()
        idx = np.argsort(dist)[:k]
        nb, dd = pool.iloc[idx], dist[idx]
        tp = nb["t_peak"].dropna()
        rows.append(dict(
            well_id=w["well_id"],
            nb_dist_min=float(dd.min()), nb_dist_mean=float(dd.mean()),
            nb_t_peak_med=float(nb["t_peak"].median()),
            nb_q_peak_med=float(nb["q_peak"].median()),
            nb_p_peak_med=float(nb["p_peak"].median()),
            nb_t_peak_iqr=float(tp.quantile(.75) - tp.quantile(.25)) if len(tp) > 2 else np.nan,
            nb_eur_med=float(nb["eur"].median()) if "eur" in nb else np.nan,
            nb_count=float(len(nb))))
    return pd.DataFrame(rows)


def build(prod: pd.DataFrame, master: pd.DataFrame, static: pd.DataFrame,
          obs_days: int, ref_labels: Optional[pd.DataFrame] = None,
          k: int = SPATIAL_K, spatial_master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """spatial_master: 找邻井时可用的全量井表。
    单井在线预测时只传该井的 master，邻居必须从全量井里找 —— 否则没有邻居。"""
    early = pd.DataFrame([
        dict(well_id=wid, **early_window_features(g, obs_days))
        for wid, g in prod.groupby("well_id")
    ])

    st = static.copy()
    st["log_perm"] = np.log10(st["perm_md"].clip(lower=1e-4))
    st = st[["well_id"] + STATIC_KEYS]

    m = master.copy()
    m["proppant_per_stage"] = m["proppant_t"] / m["stage_count"].clip(lower=1)
    m["is_horizontal"] = (m["well_type"] == "水平井").astype(float)
    m = m.merge(static[["well_id", "perm_md", "net_pay_m"]], on="well_id", how="left")
    m["kh_proxy"] = m["perm_md"] * m["net_pay_m"]
    keep = ["well_id", "block", "layer", "first_prod_date"] + COMPLETION_KEYS

    if spatial_master is not None:
        sm = spatial_master.copy()
        sm = sm.merge(static[["well_id", "perm_md", "net_pay_m"]], on="well_id", how="left") \
            if "perm_md" not in sm else sm
        pool = pd.concat([m, sm[~sm["well_id"].isin(m["well_id"])]], ignore_index=True)
        sp = _spatial(pool, ref_labels, k)
        sp = sp[sp["well_id"].isin(m["well_id"])]
    else:
        sp = _spatial(m, ref_labels, k)

    df = (early.merge(st, on="well_id", how="left")
               .merge(m[keep], on="well_id", how="left")
               .merge(sp, on="well_id", how="left"))
    df["obs_days"] = obs_days
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    """训练用列：数值特征 + 区块/层位 one-hot。刻意不含投产年份，避免学成时间外推。"""
    base = [c for c in EARLY_KEYS + STATIC_KEYS + COMPLETION_KEYS + SPATIAL_KEYS if c in df]
    return base + [c for c in df.columns if c.startswith(("block_", "layer_"))]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("block", "layer"):
        if col in out:
            out = pd.concat([out, pd.get_dummies(out[col], prefix=col, dtype=float)], axis=1)
    return out


def align(X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """把在线构造的特征对齐到训练时的列集合（缺失的 one-hot 补 0）。"""
    out = X.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0 if c.startswith(("block_", "layer_")) else np.nan
    return out
