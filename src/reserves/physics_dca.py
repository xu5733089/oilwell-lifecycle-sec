"""带物理约束的递减模型：线性流 → 边界控制流 → 终端递减（LF-BDF）。

经验 Arps 的问题：致密油水平井早期处于长期线性流，拟合出的 b 常常 > 1；外推时这个"瞬态 b"被当成
整个生命周期的规律，EUR 系统性偏高 —— 而 SEC 已证实储量最怕高估。

物理约束（每一条都能在答辩时讲出来源）：
  1. 流态分段：线性流阶段 q ∝ (1 + 2·Di·t)^(−1/2)（等效 b = 2，渗流理论中裂缝线性流的产量–时间关系）；
     线性流结束（t_elf）后进入边界控制流，b ≤ 1（边界控制流下 Arps b 的物理上限）；
     两段在 t_elf 处产量与瞬时递减率都连续；递减率降到终端递减率后转指数递减。
  2. 物质平衡上限：EUR ≤ 地质储量 × 采收率上限（容积法 P50 × 区块经验采收率上限）。以罚函数进入拟合。
  3. 类比先验（贝叶斯最大后验）：同区块成熟井的 b_bdf、t_elf、Di 分布作为先验（analog_prior），
     数据项按观测噪声 σ 标准化 —— 历史越短先验占比越大，历史长了由数据说话；没有类比井时用宽先验。
诊断：产量–时间双对数斜率（线性流约 −1/2，进入边界控制流后变陡）；规整化压力 RNP 与物质平衡时间 MBT 双对数斜率
（线性流约 1/2，边界控制流约 1）。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from . import dca

B_LF = 2.0
B_BDF_MIN, B_BDF_MAX = 0.05, 1.0
HORIZON_MONTHS = 600.0
SIGMA_OBS = 0.12            # 月度对数产量的观测噪声（开井日均口径），决定数据项与先验项的相对权重
# 无类比井时的宽先验：t_elf 以几个月为中心、对数标准差很大 —— 数据显示没有线性流时可以收缩到 0
DEFAULT_PRIOR = dict(b_mean=0.8, b_sd=0.35, t_elf_months=3.0, t_elf_log_sd=1.5,
                     di_log_mean=math.log(0.12), di_log_sd=0.8, strength=1.0)


@dataclass
class PhysicsFit:
    qi: float
    di: float                 # 线性流阶段的名义递减率，/月
    t_elf: float              # 线性流结束时间，月（自拟合起点起算）
    b_bdf: float              # 边界控制流 b
    d_min: float              # 终端递减率，/月
    rmse: float = float("nan")
    r2: float = float("nan")
    n_points: int = 0
    converged: bool = False
    eur_cap_t: Optional[float] = None
    cap_binding: bool = False
    prior_weight: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


def _elf_effective(f: PhysicsFit) -> float:
    """线性流阶段的递减率降到终端递减率时，线性流必须结束（否则会比终端递减还缓）。"""
    t = max(float(f.t_elf), 0.0)
    if f.d_min > 0 and f.di > f.d_min:
        t = min(t, (f.di / f.d_min - 1.0) / (2.0 * f.di))
    return t


def rate(t_month, f: PhysicsFit) -> np.ndarray:
    t = np.maximum(np.asarray(t_month, dtype=float), 0.0)
    te = _elf_effective(f)
    q_lf = f.qi / np.sqrt(1.0 + 2.0 * f.di * t)
    d_elf = f.di / (1.0 + 2.0 * f.di * te)
    q_elf = f.qi / math.sqrt(1.0 + 2.0 * f.di * te)
    tb = np.maximum(t - te, 0.0)
    b = min(max(f.b_bdf, B_BDF_MIN), B_BDF_MAX)
    if f.d_min > 0 and d_elf <= f.d_min:
        q_b = q_elf * np.exp(-f.d_min * tb)
    else:
        q_b = q_elf / np.power(1.0 + b * d_elf * tb, 1.0 / b)
        if f.d_min > 0:
            tsw = (d_elf / f.d_min - 1.0) / (b * d_elf)
            late = tb > tsw
            if late.any():
                q_sw = q_elf / (1.0 + b * d_elf * tsw) ** (1.0 / b)
                q_b = np.where(late, q_sw * np.exp(-f.d_min * (tb - tsw)), q_b)
    return np.where(t < te, q_lf, q_b)


def decline_rate(t_month, f: PhysicsFit) -> np.ndarray:
    """瞬时递减率 D = −d ln q / dt（数值差分），图上看流态转换用。"""
    t = np.asarray(t_month, dtype=float)
    h = 1e-3
    return -(np.log(rate(t + h, f)) - np.log(rate(np.maximum(t - h, 0.0), f))) / (2 * h)


def remaining(f: PhysicsFit, q_econ: float, t_now: float, step: float = 0.5) -> Tuple[float, float, bool]:
    t = np.arange(t_now, HORIZON_MONTHS, step)
    if len(t) == 0:
        return 0.0, t_now, False
    q = rate(t, f)
    alive = np.cumprod(q >= q_econ).astype(bool)
    rem = float(np.sum(q[alive]) * step * dca.DAYS_PER_MONTH) if hasattr(dca, "DAYS_PER_MONTH") \
        else float(np.sum(q[alive]) * step * 30.4)
    t_econ = float(t[alive][-1]) if alive.any() else float(t_now)
    return rem, t_econ, bool(alive[-1])


# --------------------------------------------------------------------------- #
# 诊断
def _rolling_slope(x: np.ndarray, y: np.ndarray, win: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    h = win // 2
    for i in range(len(x)):
        a, b = max(0, i - h), min(len(x), i + h + 1)
        if b - a >= 3:
            out[i] = np.polyfit(x[a:b], y[a:b], 1)[0]
    return out


def flow_regime(t_prod_month: np.ndarray, q: np.ndarray, t_peak_month: float = 0.0) -> Dict:
    """产量–时间双对数诊断。t_prod_month 自投产起算（线性流的时间原点是投产，不是峰值）。"""
    t = np.asarray(t_prod_month, float)
    q = np.asarray(q, float)
    ok = (t > 0) & np.isfinite(q) & (q > 0) & (t >= t_peak_month)
    t, q = t[ok], q[ok]
    if len(t) < 6:
        return dict(points=[], slopes=[], early_slope=None, late_slope=None, t_elf_month=None,
                    regime_now="历史不足，无法判断流态")
    lt, lq = np.log(t), np.log(q)
    sl = _rolling_slope(lt, lq, max(5, len(t) // 5))
    n3 = max(3, len(t) // 3)
    early = float(np.polyfit(lt[:n3], lq[:n3], 1)[0])
    late = float(np.polyfit(lt[-n3:], lq[-n3:], 1)[0])
    t_elf = None
    steep = np.where(np.nan_to_num(sl, nan=0.0) < -0.85)[0]
    for i in steep:                                     # 斜率持续变陡的起点
        if (np.nan_to_num(sl[i:i + 3], nan=0.0) < -0.85).all():
            t_elf = float(t[i])
            break
    regime = "边界控制流（斜率已明显陡于 −1/2）" if late < -0.85 else \
        "线性流（瞬态，斜率接近 −1/2）" if late > -0.7 else "流态过渡"
    stride = max(1, len(t) // 120)
    return dict(points=[[round(float(a), 3), round(float(b), 4)] for a, b in zip(t[::stride], q[::stride])],
                slopes=[[round(float(a), 3), None if not np.isfinite(b) else round(float(b), 4)]
                        for a, b in zip(t[::stride], sl[::stride])],
                early_slope=round(early, 3), late_slope=round(late, 3),
                t_elf_month=None if t_elf is None else round(t_elf, 2), regime_now=regime)


def rnp_diagnostic(cum_t: np.ndarray, q: np.ndarray, whp: np.ndarray) -> Dict:
    """规整化压力 RNP = (p_i − p_wf) / q 对 物质平衡时间 MBT = Q / q 的双对数：线性流斜率≈1/2，边界控制流≈1。
    这里用井口压力代替井底流压、以早期最高井口压力的 1.05 倍作初始压力 —— 只作流态形态诊断，不做定量试井解释。"""
    cum_t, q, whp = (np.asarray(v, float) for v in (cum_t, q, whp))
    ok = np.isfinite(cum_t) & np.isfinite(q) & np.isfinite(whp) & (q > 0) & (cum_t > 0)
    if ok.sum() < 8:
        return dict(points=[], slope=None, p_init=None)
    valid_whp = whp[ok]
    p_i = float(np.nanmax(valid_whp[: max(5, len(valid_whp) // 10)])) * 1.05
    mbt, rnp = cum_t[ok] / q[ok], (p_i - whp[ok]) / q[ok]
    good = (mbt > 0) & (rnp > 0)
    if good.sum() < 8:
        return dict(points=[], slope=None, p_init=round(p_i, 3))
    mbt, rnp = mbt[good], rnp[good]
    slope = float(np.polyfit(np.log(mbt[len(mbt) // 2:]), np.log(rnp[len(rnp) // 2:]), 1)[0])
    stride = max(1, len(mbt) // 120)
    return dict(points=[[round(float(a), 3), round(float(b), 5)] for a, b in zip(mbt[::stride], rnp[::stride])],
                slope=round(slope, 3), p_init=round(p_i, 3))


# --------------------------------------------------------------------------- #
# 拟合
def fit(t_month: np.ndarray, q: np.ndarray, *, q_econ: float, cum_to_now: float,
        eur_cap_t: Optional[float] = None, d_min_year: float = 0.075,
        prior: Optional[Dict] = None, t_elf_guess: Optional[float] = None,
        halflife_months: float = 24.0, starts: Sequence[float] = (0.3, 6.0, 24.0)) -> PhysicsFit:
    """t_month 自拟合起点（峰值）起算的月数，q 为对应日产（t/d）。"""
    pr = dict(DEFAULT_PRIOR, **(prior or {}))
    t = np.asarray(t_month, float)
    q = np.asarray(q, float)
    ok = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t, q = t[ok], q[ok]
    d_min = d_min_year / 12.0
    if len(t) < 4:
        return PhysicsFit(qi=float(q[0]) if len(q) else 0.0, di=math.exp(pr["di_log_mean"]),
                          t_elf=pr["t_elf_months"], b_bdf=pr["b_mean"], d_min=d_min, n_points=len(t),
                          eur_cap_t=eur_cap_t)
    t = t - t.min()
    t_now = float(t.max())
    w = np.power(0.5, (t_now - t) / halflife_months)
    sw = np.sqrt(w / w.mean()) / SIGMA_OBS
    lq = np.log(q)
    k = pr["strength"]
    span = B_BDF_MAX - B_BDF_MIN

    def unpack(th) -> PhysicsFit:
        b = B_BDF_MIN + span * (0.5 * (1.0 + math.tanh(0.5 * th[3])))
        return PhysicsFit(qi=math.exp(th[0]), di=math.exp(th[1]), t_elf=math.exp(th[2]), b_bdf=b, d_min=d_min)

    def resid(th):
        f = unpack(th)
        r = sw * (np.log(np.maximum(rate(t, f), 1e-9)) - lq)
        extra = [k * (f.b_bdf - pr["b_mean"]) / pr["b_sd"],
                 k * (th[2] - math.log(pr["t_elf_months"])) / pr["t_elf_log_sd"],
                 k * (th[1] - pr["di_log_mean"]) / pr["di_log_sd"]]
        if eur_cap_t:
            rem, _, _ = remaining(f, q_econ, t_now, step=1.0)
            extra.append(50.0 * max(0.0, (cum_to_now + rem) / eur_cap_t - 1.0))
        return np.concatenate([r, np.asarray(extra)])

    guesses = list(starts) + ([t_elf_guess] if t_elf_guess else [])
    lo = [math.log(1e-3), math.log(1e-4), math.log(0.05), -8.0]
    hi = [math.log(1e4), math.log(3.0), math.log(240.0), 8.0]
    b0 = min(max((pr["b_mean"] - B_BDF_MIN) / span, 0.02), 0.98)
    best = None
    for te in guesses:
        x0 = [math.log(max(q[:3].mean(), 1e-3)), pr["di_log_mean"], math.log(te), math.log(b0 / (1 - b0))]
        x0 = [min(max(v, l + 1e-6), h - 1e-6) for v, l, h in zip(x0, lo, hi)]
        try:
            res = least_squares(resid, x0=x0, bounds=(lo, hi), max_nfev=400)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    f = unpack(best.x)
    pred = rate(t, f)
    ss_res = float(np.sum((pred - q) ** 2))
    ss_tot = float(np.sum((q - q.mean()) ** 2)) or 1.0
    f.rmse, f.r2, f.n_points, f.converged = float(np.sqrt(ss_res / len(q))), 1.0 - ss_res / ss_tot, len(q), bool(best.success)
    f.eur_cap_t, f.prior_weight = eur_cap_t, round(3.0 / (3.0 + len(t)), 3)
    if eur_cap_t:
        rem, _, _ = remaining(f, q_econ, t_now)
        f.cap_binding = bool(cum_to_now + rem >= 0.995 * eur_cap_t)
    return f


def bootstrap_remaining(t_month, q, f: PhysicsFit, *, q_econ: float, cum_to_now: float,
                        d_min_year: float = 0.075, prior: Optional[Dict] = None,
                        n_boot: int = 40, seed: int = 0) -> Dict[str, float]:
    """对数残差移动块自助法；每次重拟合都带同样的物理约束与先验。"""
    t = np.asarray(t_month, float)
    q = np.asarray(q, float)
    ok = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t, q = t[ok] - t[ok].min(), q[ok]
    t_now = float(t.max())
    base, _, _ = remaining(f, q_econ, t_now)
    if len(t) < 12:
        return dca._band(np.array([base * k for k in (0.6, 1.0, 1.4)]), 0)
    fitted = rate(t, f)
    res = np.log(q) - np.log(np.maximum(fitted, 1e-9))
    n = len(res)
    block = max(3, int(np.sqrt(n)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
        rb = np.concatenate([res[s:s + block] for s in starts])[:n]
        fb = fit(t, fitted * np.exp(rb), q_econ=q_econ, cum_to_now=cum_to_now, eur_cap_t=f.eur_cap_t,
                 d_min_year=d_min_year, prior=prior, starts=(max(f.t_elf, 0.1),))
        vals.append(remaining(fb, q_econ, t_now)[0])
    return dca._band(np.asarray(vals), len(vals))


def analog_prior(fits: Sequence[PhysicsFit], min_n: int = 5) -> Optional[Dict]:
    """同区块成熟井的拟合参数 → 先验（中位数 + 稳健标准差，带下限防止先验过窄）。"""
    fits = [f for f in fits if f.converged and f.n_points >= 24]
    if len(fits) < min_n:
        return None
    b = np.array([f.b_bdf for f in fits])
    te = np.log(np.array([max(f.t_elf, 0.05) for f in fits]))
    di = np.log(np.array([f.di for f in fits]))
    mad = lambda v: 1.4826 * float(np.median(np.abs(v - np.median(v))))
    return dict(b_mean=float(np.median(b)), b_sd=max(mad(b), 0.12),
                t_elf_months=float(np.exp(np.median(te))), t_elf_log_sd=max(mad(te), 0.5),
                di_log_mean=float(np.median(di)), di_log_sd=max(mad(di), 0.25), strength=1.0, n_analogs=len(fits))


def monthly_series(day_index: np.ndarray, oil_t: np.ndarray, hours_on: np.ndarray,
                   bin_days: float = 30.4) -> Tuple[np.ndarray, np.ndarray]:
    """日度 → 月度开井日均产量（停井日不计入分母），降噪并与月度流态诊断口径一致。"""
    d = np.asarray(day_index, float)
    oil = np.asarray(oil_t, float)
    on = np.asarray(hours_on, float) > 0
    m = np.floor((d - 1) / bin_days).astype(int)
    months, rates = [], []
    for k in np.unique(m):
        sel = (m == k) & on & np.isfinite(oil)
        if sel.sum() >= 8:
            months.append((k + 0.5))
            rates.append(float(oil[sel].mean()))
    return np.asarray(months, float), np.asarray(rates, float)
