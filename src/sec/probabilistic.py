"""老井基础剩余可采的概率汇总（蒙特卡洛 + 单因子高斯 Copula），纯计算、不读库。

为什么需要：已证实储量逐井取低值再相加，等于假设所有井"同时往坏处偏"（完全相关）。
井数一多、误差又不完全相关时，整体的 90% 把握值（10% 分位）会高于逐井低值之和 —— 差多少取决于误差分布和相关性，
这两件事都不能拍脑袋，所以：

  · 单井误差分布：直接取递减回测里"真实剩余可采 / 预测剩余可采"的对数比的**经验分布**（不假设正态，保留左尾）；
    按单井拟合的历史长短分档（短历史的误差更大）；
  · 相关性：单因子高斯 Copula —— 同一组（区块）内的井共享一个因子，z_i = √ρ·Z_组 + √(1−ρ)·ε_i，
    u_i = Φ(z_i) 映射到经验分布的分位数。ρ = 1 时同一组内的井取同一分位、组与组之间仍独立，ρ = 0 时所有井相互独立；
    "所有井取同一分位"（逐井分位直接相加）是更极端的完全同步，结果里单独给出这个参照值 sum_well_p10_t；
  · 相关系数从回测误差的组内相关估计，但组太少时估计不可靠，所以结果对 ρ 做敏感性，不只报一个数。

分位口径与全仓库一致：p10 是 10% 分位（数值小者，对应"P90 低估计"），p90 是 90% 分位。
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


def icc_oneway(values: Sequence[float], groups: Sequence) -> Dict:
    """单因子方差分析的组内相关系数（负值截为 0）。"""
    v = pd.Series(np.asarray(values, float))
    g = pd.Series(np.asarray(groups)).astype(str)
    df = pd.DataFrame(dict(v=v, g=g)).dropna()
    grp = df.groupby("g")["v"]
    n_groups, n = grp.ngroups, len(df)
    if n_groups < 2 or n <= n_groups:
        return dict(icc=0.0, n_groups=int(n_groups), n=int(n))
    k = grp.size().mean()
    msb = float((grp.size() * (grp.mean() - df["v"].mean()) ** 2).sum() / (n_groups - 1))
    msw = float(((df["v"] - grp.transform("mean")) ** 2).sum() / (n - n_groups))
    icc = (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) > 0 else 0.0
    return dict(icc=float(max(0.0, icc)), n_groups=int(n_groups), n=int(n))


def _quantile_map(errors: np.ndarray, u: np.ndarray) -> np.ndarray:
    e = np.sort(np.asarray(errors, float))
    ranks = (np.arange(len(e)) + 0.5) / len(e)
    return np.interp(u, ranks, e)


def aggregate(best_t: Sequence[float], groups: Sequence, bucket: Sequence, errors_by_bucket: Dict,
              rho: float, n_sims: int = 4000, seed: int = 0) -> Dict:
    """best_t：逐井最佳估计剩余可采；bucket：每口井用哪一档误差分布；errors_by_bucket：档 → 对数比样本。"""
    best = np.asarray(best_t, float)
    n = len(best)
    if n == 0:
        return dict(n_wells=0, rho=float(rho))
    rho = float(min(max(rho, 0.0), 1.0))
    gidx = pd.factorize(pd.Series(np.asarray(groups)).astype(str))[0]
    rng = np.random.default_rng(seed)
    z = np.sqrt(rho) * rng.standard_normal((n_sims, gidx.max() + 1))[:, gidx] \
        + np.sqrt(1.0 - rho) * rng.standard_normal((n_sims, n))
    u = norm.cdf(z)
    bucket = np.asarray(bucket)
    err = np.empty_like(u)
    q10_well = np.empty(n)
    for b, sample in errors_by_bucket.items():
        cols = np.flatnonzero(bucket == b)
        if len(cols):
            err[:, cols] = _quantile_map(sample, u[:, cols])
            q10_well[cols] = _quantile_map(sample, np.array([0.10]))[0]
    total = (best[None, :] * np.exp(err)).sum(axis=1)
    sum_well_p10 = float(np.sum(best * np.exp(q10_well)))
    p10 = float(np.quantile(total, 0.10))
    return dict(rho=rho, n_wells=int(n), n_sims=int(n_sims),
                p10_t=p10, p50_t=float(np.quantile(total, 0.50)), p90_t=float(np.quantile(total, 0.90)),
                mean_t=float(total.mean()), low_estimate_t=p10,
                sum_best_t=float(best.sum()), sum_well_p10_t=sum_well_p10,
                aggregation_gain_t=p10 - sum_well_p10,
                aggregation_gain_pct=(100.0 * (p10 - sum_well_p10) / sum_well_p10) if sum_well_p10 > 0 else None)
