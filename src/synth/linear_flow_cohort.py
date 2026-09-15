"""线性流验证队列：用渗流力学解析解生成的月度产量（只供递减模型回测，不入库）。

为什么不用 physics_dca 自己的公式造数据：那样等于让物理约束模型考自己出的题 ——
和"主合成数据的递减按 Arps 生成、天然偏向经验 Arps"是同一个毛病。

生成模型：致密油压裂水平井的"改造区 + 外围基质"复合流动，两个封闭平板恒压解叠加

    q(t) = A · [ S((t+t₀)/τ₁)/√τ₁ + c · S((t+t₀)/τ₂)/√τ₂ ]，   S(t_D) = Σ_{n≥0} exp(−(2n+1)² π² t_D / 4)

  · 线性流阶段两项都是 1/(2√(πt)) 量级，产量–时间双对数斜率 −1/2；
  · t ≈ 0.2 τ₁ 后改造区进入边界控制流、指数衰减；外围基质（τ₂ = 8–15 τ₁、权重 c）继续线性流，形成长尾；
  · t₀（1–3 个月）是返排、油嘴控制与裂缝清洗造成的早期平缓段 —— 没有它，首月被线性流奇异点抬得过高，
    产量第 3 年就跌破经济极限（首版队列的问题）。
这个复合解析解既不是 Arps 双曲，也不是物理约束模型的分段公式，对两种拟合方法是公平的考题。

参数只按生成器自身的合理性统计选定，没有看任何拟合方法的回测结果：
经济寿命中位约 19 年（P10 10 年、P90 34 年），前 72 个月产出约占 EUR 的 68%，改造区线性流结束时间中位约 15 个月。

队列：三个"区域"的 τ₁ 中位数不同（类比先验要能从同区域成熟井学到东西）；首月开井日均对数正态（中位 60 t/d）；
月度产量带 7% 计量噪声；约 20% 的月份有停井（影响月产量，不影响开井日均）；产量跌破经济极限（2 t/d）后关井。
EUR 真值 = 投产到经济极限的累产（50 年封顶）。
"""
from __future__ import annotations

from math import pi
from typing import Dict, List, Sequence

import numpy as np

DAYS_PER_MONTH = 30.4
Q_ECON = 2.0
HORIZON_MONTHS = 600
AREA_TAU_MEDIAN = {"LF-A": 40.0, "LF-B": 80.0, "LF-C": 160.0}      # 改造区特征时间 τ₁（月）
_SUB_FIRST = (np.arange(60) + 0.5) / 60.0
_SUB = (np.arange(10) + 0.5) / 10.0


def slab_sum(t_d: np.ndarray, n_terms: int = 400) -> np.ndarray:
    t_d = np.maximum(np.asarray(t_d, float), 1e-12)
    n = np.arange(n_terms)[:, None]
    s = np.exp(-((2 * n + 1) ** 2) * (pi ** 2) * t_d[None, :] / 4.0).sum(axis=0)
    return np.where(t_d < 1e-3, 0.5 / np.sqrt(pi * t_d), s)          # 极早期用渐近式，避免级数截断


def monthly_shape(taus: Sequence[float], weights: Sequence[float], n_months: int, t0: float = 0.0) -> np.ndarray:
    """第 k 个月的月平均"形状"：Σ w_j · S((t+t₀)/τ_j)/√τ_j（首月子点与级数项数取足）。"""
    out = np.zeros(n_months)
    for tau, w in zip(taus, weights):
        first = (slab_sum((_SUB_FIRST + t0) / tau, 400) / np.sqrt(tau)).mean()
        k = (np.arange(1, n_months)[:, None] + _SUB[None, :]).ravel() + t0
        rest = (slab_sum(k / tau, 128) / np.sqrt(tau)).reshape(n_months - 1, -1).mean(axis=1)
        out += w * np.concatenate([[first], rest])
    return out


def generate(n_wells: int = 330, months: int = 72, seed: int = 20260916) -> List[Dict]:
    rng = np.random.default_rng(seed)
    areas = list(AREA_TAU_MEDIAN)
    wells: List[Dict] = []
    for i in range(n_wells):
        area = areas[i % len(areas)]
        tau1 = float(np.clip(rng.lognormal(np.log(AREA_TAU_MEDIAN[area]), 0.35), 6.0, 400.0))
        tau2 = tau1 * float(rng.uniform(8.0, 15.0))
        c = float(rng.uniform(0.30, 0.60))
        t0 = float(rng.uniform(1.0, 3.0))
        q1 = float(np.clip(rng.lognormal(np.log(60.0), 0.40), 10.0, 200.0))
        shape = monthly_shape((tau1, tau2), (1.0, c), HORIZON_MONTHS, t0)
        truth_rate = q1 / shape[0] * shape
        alive = np.cumprod(truth_rate >= Q_ECON).astype(bool)
        eur = float(np.sum(truth_rate[alive]) * DAYS_PER_MONTH)
        n_obs = int(min(months, alive.sum()))
        rate = truth_rate[:n_obs] * rng.lognormal(0.0, 0.07, n_obs)
        on_frac = np.where(rng.random(n_obs) < 0.2, rng.uniform(0.6, 0.95, n_obs), 1.0)
        wells.append(dict(well_id=f"LF{i:04d}", well_code=f"LF-{area[-1]}-{i:04d}", block=area,
                          order_key=f"{i % 97:03d}-{i:04d}",
                          months=np.arange(n_obs, dtype=float) + 0.5, rates=rate,
                          volumes=rate * DAYS_PER_MONTH * on_frac, eur_true=eur,
                          tau_srv_months=tau1, tau_matrix_months=tau2, matrix_weight=c, t0_months=t0,
                          q_first_t_per_d=q1, t_elf_true_months=0.2 * tau1))
    return wells
