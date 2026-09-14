"""工作量剥离：提采新井识别、措施效果评价、"新-老-措"产量构成、自然/综合递减率。

纯计算：不读库，不知道大模型的存在。输入是月度生产表
    well_id, ym(YYYY-MM), oil_t, water_m3, days_on, days
由服务层从 prod_daily 聚合后传入；测试里用 to_monthly() 从日度数据聚合。

两个口径：
  · 产量构成与递减率用**日历日产**（月产 / 当月天数）—— 停井也是递减的一部分；
  · 措施基线用**生产日产**（月产 / 生产天数）—— 措施前后的停井天数不同，
    用日历日产会把停井差异算成措施效果。
"""
from __future__ import annotations

import calendar
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import dca

DAYS_PER_MONTH = 30.4


# --------------------------------------------------------------------------- #
# 月份工具
def ym_to_idx(ym: str) -> int:
    return int(ym[:4]) * 12 + int(ym[5:7]) - 1


def idx_to_ym(i: int) -> str:
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def days_in_month(ym: str) -> int:
    return calendar.monthrange(int(ym[:4]), int(ym[5:7]))[1]


def period_bounds(as_of: str, months: int) -> Tuple[str, str]:
    """评估期：基准日所在月往前数 months 个月（含当月）。"""
    end = str(as_of)[:7]
    return idx_to_ym(ym_to_idx(end) - int(months) + 1), end


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.assign(ym=daily["dt"].str[:7], on=(daily["hours_on"] > 0).astype(int))
    return (d.groupby(["well_id", "ym"], as_index=False)
             .agg(oil_t=("oil_t", "sum"), water_m3=("water_m3", "sum"),
                  days_on=("on", "sum"), days=("dt", "size")))


def _remaining_above(rates: np.ndarray, q_econ: float) -> float:
    """逐月日产序列截到首次跌破经济极限为止的累计产量（t）。"""
    rates = np.asarray(rates, dtype=float)
    alive = np.cumprod(rates >= q_econ).astype(bool)
    return float(np.sum(rates[alive]) * DAYS_PER_MONTH)


# --------------------------------------------------------------------------- #
# 提采新井 / 扩边井识别
NEW_WELL_COLUMNS = ["well_id", "first_prod_date", "category", "n_old_neighbors",
                    "nearest_old_m", "radius_m", "min_old_neighbors"]


def identify_new_wells(master: pd.DataFrame, as_of: str, period_months: int,
                       radius_m: float, min_old_neighbors: int) -> pd.DataFrame:
    """评估期内投产的井，按"周边有没有评估期前已投产的老井"分成提采新井与扩边井。

    提采新井打在已探明范围之内，周边老井多；扩边井打在已探明范围之外，周边没有老井。
    邻井不限层系：已探明范围是平面概念。master 须是**全部**井，不能只传单元内的井 ——
    单元边界上的新井，邻井可能在隔壁单元。
    """
    start_ym, end_ym = period_bounds(as_of, period_months)
    fp = master["first_prod_date"].astype(str).str[:7]
    new = master[(fp >= start_ym) & (fp <= end_ym)]
    old = master[fp < start_ym]
    ox, oy = old["x_off"].to_numpy(float), old["y_off"].to_numpy(float)
    rows: List[Dict] = []
    for r in new.itertuples(index=False):
        dist = np.hypot(ox - float(r.x_off), oy - float(r.y_off))
        n = int((dist <= radius_m).sum())
        rows.append(dict(well_id=r.well_id, first_prod_date=r.first_prod_date,
                         category="infill" if n >= min_old_neighbors else "extension",
                         n_old_neighbors=n,
                         nearest_old_m=float(dist.min()) if len(dist) else None,
                         radius_m=float(radius_m), min_old_neighbors=int(min_old_neighbors)))
    return pd.DataFrame(rows, columns=NEW_WELL_COLUMNS)


# --------------------------------------------------------------------------- #
# 措施效果评价
def measure_increment_remaining(effect: Dict, q_econ: float, horizon_months: int = 360) -> float:
    """措施增加的剩余可采 = 有措施时截到经济极限的剩余 − 无措施（基线）截到经济极限的剩余。

    两边都截经济极限才对：措施可能让一口本该关井的井多活几年，这几年的全部产量都算措施带来的。
    因此它**不随经济极限单调**：经济极限抬高时基线更早关井，措施延寿的占比反而更大。
    """
    base = dca.DCAFit(**effect["baseline_fit"])
    h = np.arange(1, horizon_months + 1, dtype=float)
    b = dca.rate(effect["t_now_month"] + h, base)
    total = b + effect["inc_now_t_per_d"] * np.exp(-effect["d_inc_month"] * h)
    return max(_remaining_above(total, q_econ) - _remaining_above(b, q_econ), 0.0)


def measure_effect(wm: pd.DataFrame, event_ym: str, as_of_ym: str, q_econ: float, *,
                   max_pre_months: int = 24, min_pre_months: int = 6,
                   min_post_months: int = 3, min_days_on: int = 10) -> Dict:
    """单次措施的效果：措施前递减外推为基线，措施后实际与基线之差为增油。

    wm：单井月度表。返回 status：
      ok                 可评：给出已实现增油、增加的剩余可采、增加可采储量
      insufficient_pre   措施前有效生产月数不足，基线不可信，不剥离增油
      insufficient_post  措施后观察期不足，给出已实现增油，但不计增储
    """
    w = wm[wm["ym"] <= as_of_ym].copy()
    w["i"] = w["ym"].map(ym_to_idx)
    w = w.sort_values("i")
    w["rate"] = w["oil_t"] / w["days_on"].clip(lower=1)
    ev = ym_to_idx(event_ym)
    valid = w["days_on"] >= min_days_on
    pre = w[valid & (w["i"] < ev) & (w["i"] >= ev - max_pre_months)]
    out: Dict = dict(event_ym=event_ym, n_pre_months=int(len(pre)),
                     min_pre_months=int(min_pre_months), min_post_months=int(min_post_months))
    if len(pre) < min_pre_months:
        return dict(out, status="insufficient_pre", n_post_months=0, effective=None,
                    realized_inc_t=None, inc_remaining_t=None, inc_eur_t=None, monthly_inc=[],
                    reason=f"措施前有效生产仅 {len(pre)} 个月，少于 {min_pre_months} 个月，"
                           "基线不可信，未剥离增油")

    t0 = int(pre["i"].min())
    base = dca.fit_best(pre["i"].to_numpy(float) - t0, pre["rate"].to_numpy(float),
                        candidates=("modified_hyperbolic", "exponential"))

    def base_rate(idx) -> np.ndarray:
        return dca.rate(np.asarray(idx, dtype=float) - t0, base)

    # 已实现增油：措施当月起，实际月产 − 基线日产 × 同样的生产天数（当月措施前的天数增量自然为零）
    since = w[w["i"] >= ev]
    inc_month = since["oil_t"].to_numpy(float) - base_rate(since["i"]) * since["days_on"].to_numpy(float)
    post = w[valid & (w["i"] > ev)]
    out.update(baseline_fit=base.to_dict(), baseline_start_ym=idx_to_ym(t0),
               pre_rate_t_per_d=float(pre["rate"].tail(3).mean()),
               realized_inc_t=float(inc_month.sum()), n_post_months=int(len(post)),
               monthly_inc=[dict(ym=y, inc_t=float(v)) for y, v in zip(since["ym"], inc_month)])
    if len(post) < min_post_months:
        return dict(out, status="insufficient_post", effective=None, inc_remaining_t=None,
                    inc_eur_t=None,
                    reason=f"措施后有效生产仅 {len(post)} 个月，少于 {min_post_months} 个月，暂不计增储")

    inc = post["rate"].to_numpy(float) - base_rate(post["i"])
    k = post["i"].to_numpy(float) - ev
    now = int(post["i"].max())
    # 增量递减：对正的增量做对数线性回归；拟合不出递减（见效爬坡期）时退回基线的当前递减率
    d_inc = None
    pos = inc > 0
    if pos.sum() >= 3:
        slope = float(np.polyfit(k[pos], np.log(inc[pos]), 1)[0])
        if slope < 0:
            d_inc = -slope
    if d_inc is None:
        q1, q2 = base_rate([now, now + 1])
        d_inc = float(np.log(max(q1, 1e-9) / max(q2, 1e-9)))
    out.update(status="ok", t_now_month=float(now - t0), d_inc_month=max(d_inc, 0.005),
               inc_now_t_per_d=max(float(np.mean(inc[-3:])), 0.0),
               post_rate_t_per_d=float(post["rate"].head(3).mean()),
               rate_gain_t_per_d=float(np.mean(inc[:3])),
               reason="措施前递减外推为基线，措施后实际与基线之差为增油")
    out["inc_remaining_t"] = measure_increment_remaining(out, q_econ)
    out["inc_eur_t"] = out["realized_inc_t"] + out["inc_remaining_t"]
    out["effective"] = bool(out["realized_inc_t"] > 0)
    return out


def summarize_measures(effects: Sequence[Dict], names: Dict[str, str]) -> Dict:
    """按措施类型汇总：井数、措施前后单井日产、日产增幅、单井与总的增加可采储量。"""
    def agg(g: pd.DataFrame, kind: str) -> Dict:
        ok = g[g["status"] == "ok"]
        n_eff = int(ok["effective"].sum()) if len(ok) else 0
        mean = lambda c: float(ok[c].mean()) if len(ok) else None   # noqa: E731
        return dict(event_type=kind, name=names.get(kind, kind), n=int(len(g)),
                    n_evaluated=int(len(ok)), n_effective=n_eff,
                    effective_rate_pct=(100.0 * n_eff / len(ok)) if len(ok) else None,
                    pre_rate_avg_t_per_d=mean("pre_rate_t_per_d"),
                    post_rate_avg_t_per_d=mean("post_rate_t_per_d"),
                    rate_gain_avg_t_per_d=mean("rate_gain_t_per_d"),
                    inc_eur_avg_t=mean("inc_eur_t"),
                    inc_eur_total_t=float(ok["inc_eur_t"].sum()) if len(ok) else 0.0,
                    inc_remaining_total_t=float(ok["inc_remaining_t"].sum()) if len(ok) else 0.0,
                    realized_total_t=float(ok["realized_inc_t"].sum()) if len(ok) else 0.0)

    if not effects:
        return dict(by_type=[], overall=agg(pd.DataFrame(columns=["status", "effective"]), "all"))
    df = pd.DataFrame(list(effects))
    return dict(by_type=[agg(g, k) for k, g in df.groupby("event_type")],
                overall=agg(df, "all"))


# --------------------------------------------------------------------------- #
# "新-老-措"产量构成与递减率
def monthly_composition(monthly: pd.DataFrame, master: pd.DataFrame, effects: Sequence[Dict], *,
                        start_ym: str, end_ym: str, new_since_ym: str,
                        measure_since_ym: str) -> pd.DataFrame:
    """逐月拆分：总产 = 新井产量 + 措施增油 + 老井基础产量。

    new_since_ym 之后投产的井整体算新井；measure_since_ym 之后实施的措施，其增油单列。
    新井上的措施不再单列（已整体计入新井），否则会重复扣减。
    effects 须带 well_id / event_ym / monthly_inc。
    """
    fp = master.set_index("well_id")["first_prod_date"].astype(str).str[:7]
    m = monthly[(monthly["ym"] >= start_ym) & (monthly["ym"] <= end_ym)].copy()
    m["is_new"] = m["well_id"].map(fp).fillna("") >= new_since_ym
    m["new_oil"] = np.where(m["is_new"], m["oil_t"], 0.0)
    m["producing"] = (m["days_on"] > 0).astype(int)
    m["new_producing"] = m["producing"] * m["is_new"].astype(int)
    agg = m.groupby("ym").agg(total_oil_t=("oil_t", "sum"), new_oil_t=("new_oil", "sum"),
                              n_producing=("producing", "sum"),
                              n_new_producing=("new_producing", "sum"))

    inc_rows: List[Tuple[str, float]] = []
    event_months: List[str] = []
    for e in effects:
        if e.get("event_ym", "") < measure_since_ym or not e.get("monthly_inc"):
            continue
        if fp.get(e["well_id"], "") >= new_since_ym:
            continue
        event_months.append(e["event_ym"])
        inc_rows += [(r["ym"], r["inc_t"]) for r in e["monthly_inc"] if start_ym <= r["ym"] <= end_ym]
    inc = (pd.DataFrame(inc_rows, columns=["ym", "inc_t"]).astype({"inc_t": float})
             .groupby("ym")["inc_t"].sum())
    agg["measure_inc_t"] = inc.reindex(agg.index).fillna(0.0)
    agg["old_base_t"] = agg["total_oil_t"] - agg["new_oil_t"] - agg["measure_inc_t"]
    agg["n_old_producing"] = agg["n_producing"] - agg["n_new_producing"]
    agg["n_measures_cum"] = [sum(1 for x in event_months if x <= ym) for ym in agg.index]
    days = np.array([days_in_month(ym) for ym in agg.index], dtype=float)
    agg["old_base_rate_t_per_d"] = agg["old_base_t"] / days
    return agg.reset_index()


def decline_rates(comp: pd.DataFrame, window_start_ym: str, window_end_ym: str) -> Dict:
    """自然递减率（扣新井、扣措施增油）与综合递减率（只扣新井），均为指数拟合的年化值。

    年递减率 = 1 − (1 − 月递减率)^12。
    """
    s = comp[(comp["ym"] >= window_start_ym) & (comp["ym"] <= window_end_ym)]
    days = np.array([days_in_month(ym) for ym in s["ym"]], dtype=float)
    x = np.array([ym_to_idx(ym) for ym in s["ym"]], dtype=float)
    out: Dict = dict(window_start_ym=window_start_ym, window_end_ym=window_end_ym,
                     n_months=int(len(s)))
    for key, y in (("natural", s["old_base_t"].to_numpy(float) / np.maximum(days, 1)),
                   ("comprehensive", (s["old_base_t"] + s["measure_inc_t"]).to_numpy(float)
                    / np.maximum(days, 1))):
        ok = y > 0
        if ok.sum() < 6:
            out.update({f"{key}_monthly_pct": None, f"{key}_annual_pct": None, f"{key}_r2": None})
            continue
        slope, icpt = np.polyfit(x[ok], np.log(y[ok]), 1)
        pred = icpt + slope * x[ok]
        ss_res = float(np.sum((np.log(y[ok]) - pred) ** 2))
        ss_tot = float(np.sum((np.log(y[ok]) - np.log(y[ok]).mean()) ** 2)) or 1.0
        out.update({f"{key}_monthly_pct": float((1 - np.exp(slope)) * 100),
                    f"{key}_annual_pct": float((1 - np.exp(12 * slope)) * 100),
                    f"{key}_r2": 1 - ss_res / ss_tot})
    return out


def well_change_evidence(monthly: pd.DataFrame, well_ids, open_ym: str, close_ym: str, *,
                         months: int = 3, oil_density: float = 0.85, top_k: int = 3) -> List[Dict]:
    """两期之间日产降幅最大的老井及其含水变化 —— 技术修订的单井层面依据。"""
    ids = set(well_ids)

    def window(end: str) -> pd.DataFrame:
        lo = idx_to_ym(ym_to_idx(end) - months + 1)
        s = monthly[monthly["well_id"].isin(ids) & (monthly["ym"] >= lo) & (monthly["ym"] <= end)]
        days = sum(days_in_month(idx_to_ym(i)) for i in range(ym_to_idx(lo), ym_to_idx(end) + 1))
        g = s.groupby("well_id").agg(oil=("oil_t", "sum"), water=("water_m3", "sum"))
        g["rate"] = g["oil"] / days
        liquid = g["water"] + g["oil"] / oil_density
        g["wc"] = np.where(liquid > 0, g["water"] / liquid.where(liquid > 0, 1) * 100, np.nan)
        return g

    a, b = window(open_ym), window(close_ym)
    j = a.join(b, lsuffix="_open", rsuffix="_close", how="inner")
    j = j[j["rate_open"] > 0]
    j["delta"] = j["rate_close"] - j["rate_open"]
    j = j.sort_values("delta").head(top_k)
    return [dict(well_id=wid, rate_open_t_per_d=float(r["rate_open"]),
                 rate_close_t_per_d=float(r["rate_close"]),
                 rate_change_t_per_d=float(r["delta"]),
                 water_cut_open_pct=None if pd.isna(r["wc_open"]) else float(r["wc_open"]),
                 water_cut_close_pct=None if pd.isna(r["wc_close"]) else float(r["wc_close"]))
            for wid, r in j.iterrows()]
