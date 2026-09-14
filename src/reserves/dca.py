"""递减曲线分析（方案 §5.2）。

支持 指数 / 双曲 / 调和 / 修正双曲 / Duong 五种模型，自动选优。

硬约束（本模块的核心）：
    b > 1 且未设终端递减率 d_min 时，Arps 积分不收敛，EUR 会算到无穷。
    这是储量评估最常见的错误来源，因此这里直接断言拦截，
    并在 fit() 里对 b 施加上界。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

B_MAX = 1.6
CONVENTION = ("p10/p50/p90 一律为分位数本身（p10 = 10% 分位 = 数值小者）；SEC 已证实储量取 low_estimate（= p10，对应行业口径的『P90 低估计』）")
MODELS = ("exponential", "harmonic", "hyperbolic", "modified_hyperbolic", "duong")


@dataclass
class DCAFit:
    model: str
    qi: float
    di: float            # 名义初始递减率，/月
    b: float
    d_min: float         # 终端递减率，/月（0 表示不切换）
    a: float = 0.0       # Duong 参数
    m: float = 0.0       # Duong 参数
    rmse: float = float("nan")
    r2: float = float("nan")
    n_points: int = 0
    converged: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def _assert_convergent(b: float, d_min: float) -> None:
    if b > 1.0 and not (d_min and d_min > 0):
        raise ValueError(
            f"b={b:.3f} > 1 且未设终端递减率 d_min：Arps 积分不收敛，EUR 将发散。"
            " 请设置 d_min（页岩油常用 6%~8%/年）或把 b 约束到 <= 1。"
        )


def rate(t_month: np.ndarray, fit: DCAFit) -> np.ndarray:
    """给定拟合参数，预测 t 时刻（自拟合起点起算，单位月）的日产量。"""
    t = np.asarray(t_month, dtype=float)
    if fit.model == "duong":
        t_safe = np.maximum(t, 1e-6)
        return fit.qi * np.power(t_safe, -fit.m) * np.exp(
            fit.a / (1.0 - fit.m) * (np.power(t_safe, 1.0 - fit.m) - 1.0))
    if fit.b <= 1e-6:
        return fit.qi * np.exp(-fit.di * t)
    _assert_convergent(fit.b, fit.d_min)
    q = fit.qi / np.power(1.0 + fit.b * fit.di * t, 1.0 / fit.b)
    if fit.d_min and fit.d_min > 0:
        t_switch = (fit.di / fit.d_min - 1.0) / (fit.b * fit.di)
        if t_switch > 0:
            late = t > t_switch
            if late.any():
                q_sw = fit.qi / np.power(1.0 + fit.b * fit.di * t_switch, 1.0 / fit.b)
                q[late] = q_sw * np.exp(-fit.d_min * (t[late] - t_switch))
    return q


def _residual_factory(t, q, w, model, d_min):
    def resid(theta):
        if model == "duong":
            qi, a, m = theta
            f = DCAFit("duong", qi=qi, di=0.0, b=0.0, d_min=0.0, a=a, m=m)
        else:
            qi, di, b = theta
            f = DCAFit(model, qi=qi, di=di, b=b, d_min=d_min)
        try:
            pred = rate(t, f)
        except ValueError:
            return np.full_like(q, 1e6)
        return (pred - q) * w
    return resid


def fit(t_month: np.ndarray,
        q: np.ndarray,
        model: str = "modified_hyperbolic",
        d_min_year: float = 0.075,
        recency_halflife_months: Optional[float] = 24.0) -> DCAFit:
    """对（月数, 日产量）序列拟合递减曲线。

    recency_halflife_months: 近期数据加权半衰期。储量评估关心当前趋势，
    远端历史权重指数衰减；设 None 则等权。
    """
    t = np.asarray(t_month, dtype=float)
    q = np.asarray(q, dtype=float)
    ok = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t, q = t[ok], q[ok]
    if len(t) < 6:
        return DCAFit(model, qi=float(q[0]) if len(q) else 0.0, di=0.1, b=0.5,
                      d_min=0.0, n_points=len(t), converged=False)
    t = t - t.min()

    if recency_halflife_months:
        w = np.power(0.5, (t.max() - t) / recency_halflife_months)
    else:
        w = np.ones_like(t)
    w = w / w.mean()

    d_min = (d_min_year or 0.0) / 12.0
    if model == "exponential":
        lo, hi, x0 = [1e-6, 1e-5, 0.0], [1e4, 2.0, 1e-9], [q[0], 0.15, 0.0]
        d_min_eff = 0.0
    elif model == "harmonic":
        lo, hi, x0 = [1e-6, 1e-5, 1.0 - 1e-9], [1e4, 2.0, 1.0], [q[0], 0.15, 1.0]
        d_min_eff = d_min
    elif model == "hyperbolic":
        lo, hi, x0 = [1e-6, 1e-5, 1e-6], [1e4, 2.0, 1.0], [q[0], 0.15, 0.7]
        d_min_eff = 0.0
    elif model == "modified_hyperbolic":
        lo, hi, x0 = [1e-6, 1e-5, 1e-6], [1e4, 2.0, B_MAX], [q[0], 0.15, 0.9]
        d_min_eff = d_min if d_min > 0 else 0.075 / 12.0   # 强制给终端递减，见模块文档
    elif model == "duong":
        lo, hi, x0 = [1e-6, 1e-4, 1e-3], [1e4, 5.0, 2.0], [q[0], 0.8, 1.1]
        d_min_eff = 0.0
    else:
        raise ValueError(f"未知 DCA 模型 {model!r}，可选 {MODELS}")

    res = least_squares(_residual_factory(t, q, w, model, d_min_eff), x0=x0,
                        bounds=(lo, hi), max_nfev=4000)
    if model == "duong":
        f = DCAFit("duong", qi=float(res.x[0]), di=0.0, b=0.0, d_min=0.0,
                   a=float(res.x[1]), m=float(res.x[2]))
    else:
        f = DCAFit(model, qi=float(res.x[0]), di=float(res.x[1]), b=float(res.x[2]),
                   d_min=float(d_min_eff))
    pred = rate(t, f)
    ss_res = float(np.sum((pred - q) ** 2))
    ss_tot = float(np.sum((q - q.mean()) ** 2)) or 1.0
    f.rmse = float(np.sqrt(ss_res / len(q)))
    f.r2 = 1.0 - ss_res / ss_tot
    f.n_points = len(q)
    f.converged = bool(res.success)
    _assert_convergent(f.b, f.d_min)
    return f


def fit_best(t_month, q, d_min_year: float = 0.075,
             candidates: Tuple[str, ...] = ("modified_hyperbolic", "hyperbolic",
                                            "exponential", "duong")) -> DCAFit:
    """多模型择优：以加权 RMSE 最小者胜出。"""
    best = None
    for m in candidates:
        try:
            f = fit(t_month, q, model=m, d_min_year=d_min_year)
        except Exception:
            continue
        if f.converged and (best is None or f.rmse < best.rmse):
            best = f
    if best is None:
        best = fit(t_month, q, model="exponential")
    return best


def eur(f: DCAFit, q_econ: float, t_start_month: float = 0.0,
        horizon_years: float = 50.0, step_month: float = 0.5) -> Dict[str, float]:
    """把递减曲线积分到经济极限。截断是 SEC 口径的硬要求（§6.2）。"""
    _assert_convergent(f.b, f.d_min)
    t = np.arange(0.0, horizon_years * 12.0, step_month)
    q = rate(t, f)
    econ = q >= q_econ
    if not econ.any():
        return dict(eur=0.0, remaining=0.0, t_econ_month=0.0, q_econ=q_econ)
    t_econ = float(t[econ][-1])
    days_per_step = step_month * 30.4
    total = float(np.sum(q[econ]) * days_per_step)
    fut = econ & (t >= t_start_month)
    remaining = float(np.sum(q[fut]) * days_per_step)
    # 曲线到评估期末仍高于经济极限：t_econ 是积分上限，不是真的经济极限时刻。
    # 不标出来的话，"经济极限时刻 600 个月"会被当成结论读，那是假的。
    capped = bool(econ[-1])
    return dict(eur=total, remaining=remaining, t_econ_month=t_econ, q_econ=q_econ,
                t_econ_capped=capped, horizon_month=float(horizon_years * 12.0))


def bootstrap_remaining(t_month, q, f: DCAFit, q_econ: float, t_now_month: float,
                        n_boot: int = 120, seed: int = 0) -> Dict[str, float]:
    """移动块自助法，给「剩余可采量」的分位数。

    两个刻意的选择：

    1) 估的是**剩余可采**而不是 EUR。已经采出来的量是事实，没有不确定性；
       把它算进概率区间会把真实的预测不确定性稀释掉。
       EUR = 累产（确定） + 剩余可采（不确定），在调用方合成。

    2) 用**移动块**而不是 iid 重采样。生产数据的残差是自相关的
       （停井、措施、季节性），iid 重采样会破坏这种相关性，
       把区间压得远比真实不确定性窄 —— 看起来漂亮，但不可信。
       块长取 ~sqrt(n)，是块自助法的常用经验值。

    换成真正的贝叶斯后验（emcee / numpyro）时只替换本函数，接口不变。
    """
    t = np.asarray(t_month, float)
    q = np.asarray(q, float)
    ok = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t, q = t[ok] - t[ok].min(), q[ok]
    base = eur(f, q_econ, t_start_month=t_now_month)["remaining"]
    if len(t) < 12:
        return _band(np.array([base * k for k in (0.6, 1.0, 1.5)]), 0)

    resid = q - rate(t, f)
    n = len(resid)
    block = max(4, int(np.sqrt(n)))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)

    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        rb = np.concatenate([resid[s:s + block] for s in starts])[:n]
        qb = np.maximum(rate(t, f) + rb, 1e-6)
        try:
            fb = fit(t, qb, model=f.model, d_min_year=(f.d_min * 12.0) or 0.075)
            vals.append(eur(fb, q_econ, t_start_month=t_now_month)["remaining"])
        except Exception:
            continue
    if len(vals) < 5:
        return _band(np.array([base * k for k in (0.6, 1.0, 1.5)]), 0)
    return _band(np.asarray(vals), len(vals))


def eur_from_history(t_month, q, f: DCAFit, q_econ: float, cum_to_date: float,
                     n_boot: int = 120, seed: int = 0) -> Dict[str, float]:
    """EUR = 已累产（确定） + 剩余可采（概率区间）。

    绝不用「从投产起把拟合曲线整条积分」来算 EUR：近期加权会让回溯段偏低，
    结果可能出现 EUR < 累产 这种一眼假的数。
    """
    t = np.asarray(t_month, float)
    t_now = float(t.max() - t.min())
    band = bootstrap_remaining(t, q, f, q_econ, t_now, n_boot=n_boot, seed=seed)
    e = eur(f, q_econ, t_start_month=t_now)
    out = {k: (band[k] + cum_to_date if k in ("p10", "p50", "p90",
                                              "low_estimate", "high_estimate") else band[k])
           for k in band}
    out["remaining_p50"] = float(band["p50"])
    out["cum_to_date"] = float(cum_to_date)
    out["t_econ_month"] = float(e["t_econ_month"])
    out["t_econ_capped"] = bool(e.get("t_econ_capped"))
    out["t_now_month"] = t_now
    return out


def _band(v: np.ndarray, n_boot: int) -> Dict[str, float]:
    """统一口径：键名 p10/p50/p90 就是分位数本身，另给 low/high_estimate 别名。

    两套口径混用是储量工作里最常见的低级错误来源 ——
    ML 侧习惯 p10=小值，储量侧习惯 P90=低估计。本仓库只认前者，
    需要储量口径时一律读 low_estimate / high_estimate。
    """
    return dict(p10=float(np.percentile(v, 10)),
                p50=float(np.percentile(v, 50)),
                p90=float(np.percentile(v, 90)),
                low_estimate=float(np.percentile(v, 10)),
                high_estimate=float(np.percentile(v, 90)),
                convention=CONVENTION, n_boot=int(n_boot))


def synthesize_curve(q_peak: float, t_peak_day: float, eur_t: float,
                     b: float = 0.9, d_min_year: float = 0.075,
                     horizon_months: int = 120) -> Dict[str, list]:
    """由预测出的（达峰产量、达峰时间、EUR）反推一条完整生命周期曲线。

    做法：固定 b 与终端递减率，对初始递减率 Di 做二分，使积分后的 EUR 命中目标。
    上升段用幂律爬坡。这样界面上画出的曲线与三个标量预测值自洽 ——
    否则图和数字对不上，演示时一眼被看穿。
    """
    t_peak_m = max(t_peak_day, 1.0) / 30.4
    # 上升段累产 = ∫q_peak·(t/t_peak)^0.62 dt = q_peak·t_peak/1.62
    pre_cum = q_peak * t_peak_day / 1.62
    target_post = max(eur_t - pre_cum, q_peak * 30.4)

    def post_cum(di: float) -> float:
        f = DCAFit("modified_hyperbolic", qi=q_peak, di=di, b=b, d_min=d_min_year / 12.0)
        t = np.arange(0.0, horizon_months, 0.5)
        return float(np.sum(rate(t, f)) * 0.5 * 30.4)

    lo, hi = 1e-3, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if post_cum(mid) > target_post:
            lo = mid
        else:
            hi = mid
    di = 0.5 * (lo + hi)
    f = DCAFit("modified_hyperbolic", qi=q_peak, di=di, b=b, d_min=d_min_year / 12.0)

    months = np.arange(1, horizon_months + 1, dtype=float)
    q = np.where(months * 30.4 <= t_peak_day,
                 q_peak * np.power(np.maximum(months * 30.4, 1.0) / t_peak_day, 0.62),
                 rate(np.maximum(months - t_peak_m, 0.0), f))
    return dict(month=[int(m) for m in months], q=[round(float(x), 3) for x in q],
                di=round(di, 5), b=b, d_min=round(d_min_year / 12.0, 5))
