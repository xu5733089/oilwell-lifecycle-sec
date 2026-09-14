"""SEC 单元"新-老-措"构成评估（docs/dev-plan-sec-unit.md）。

把一个评估单元的已证实已开发储量拆成来源可追溯的四部分：
    老井基础 + 措施增储 + 提采新井 + 扩边井
并在此之上做自动对账、敏感性分析、变化归因与折耗率。

纯计算：输入是服务层准备好的月度表、井表、事件表与经济参数，不读库。
老井基础为什么逐井算、不做单元整体递减拟合：单元产量是不同投产批次的井叠加出来的，
只要近几年还在陆续投产，整体曲线里就混着各批新井的早期陡降，不是一条 Arps 曲线 ——
短尾巴外推会在相邻两期之间大起大落，技术修订随之失真。单元整体曲线只用来算递减率。

已证实口径：老井逐井取指数递减（保守）与最佳估计（修正双曲等择优）两者之低值；
措施增储与新井按下文各自的规则取值。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..reserves import dca
from ..reserves import workload as wl
from . import reconcile as rec

COMPONENTS = {"old_base": "老井基础", "measure": "措施增储",
              "new_infill": "提采新井", "extension": "扩边井"}
HORIZON_MONTHS = 360


# --------------------------------------------------------------------------- #
# 类型曲线与新井储量
def type_curve(monthly: pd.DataFrame, master: pd.DataFrame, as_of_ym: str, *,
               min_history_months: int = 24, min_wells: int = 5) -> Dict:
    """同单元老井按投产后月份对齐的日历日产中位数曲线（类比法的依据）。

    只取历史足够长的老井；某个月份样本井数不足就截断 —— 类型曲线必须连续，
    不能用少数几口"长寿井"把尾部拉高。
    """
    fp = master.set_index("well_id")["first_prod_date"].astype(str).str[:7].map(wl.ym_to_idx)
    hist = wl.ym_to_idx(as_of_ym) - fp + 1
    ids = hist[hist >= min_history_months].index
    m = monthly[monthly["well_id"].isin(ids) & (monthly["ym"] <= as_of_ym)].copy()
    if m.empty:
        return dict(k=[], rate=[], n_wells=0)
    m["k"] = m["ym"].map(wl.ym_to_idx) - m["well_id"].map(fp) + 1
    m["rate"] = m["oil_t"] / m["ym"].map(wl.days_in_month)
    g = m.groupby("k")["rate"].agg(["median", "size"]).sort_index()
    rate: List[float] = []
    for k, row in g.iterrows():
        if k != len(rate) + 1 or row["size"] < min_wells:
            break
        rate.append(float(row["median"]))
    return dict(k=list(range(1, len(rate) + 1)), rate=rate, n_wells=int(len(ids)))


def _analog_forecast(tc_rate: np.ndarray, months_on: int, scale: float) -> np.ndarray:
    """类型曲线从第 months_on+1 个月起按比例缩放；曲线用完后按其末 12 个月的递减率外推。"""
    tail = tc_rate[months_on:] * scale
    last = tc_rate[-12:] if len(tc_rate) >= 12 else tc_rate
    slope = float(np.polyfit(np.arange(len(last)), np.log(np.maximum(last, 1e-6)), 1)[0]) \
        if len(last) >= 2 else -0.02
    ext = tc_rate[-1] * scale * np.exp(-max(-slope, 0.005) *
                                       np.arange(1, HORIZON_MONTHS - len(tail) + 1))
    return np.concatenate([tail, ext])


def _pick_new_well_reserves(methods: Dict, q_econ: float) -> Dict:
    """新井储量取值规则（按经济极限重算，敏感性分析复用同一规则）：

    最佳估计：自身历史够长用动态法，否则用类比法；
    合理确定性：有模型法低估计时，取"最佳估计"与"模型低估计"两者之低值。
    """
    best, best_key = None, None
    if "dynamic" in methods:
        d = methods["dynamic"]
        best = dca.eur(dca.DCAFit(**d["fit"]), q_econ, t_start_month=d["t_now_month"])["remaining"]
        best_key = "dynamic"
    elif "analog" in methods:
        best = wl._remaining_above(np.asarray(methods["analog"]["forecast"]), q_econ)
        best_key = "analog"
    low = methods.get("model", {}).get("remaining_low_t")
    if best is None and low is None:
        return dict(reserves_t=None, basis=None, best_estimate_t=None)
    if best is None:
        return dict(reserves_t=low, basis="model_low", best_estimate_t=None)
    if low is None or best <= low:
        return dict(reserves_t=best, basis=best_key, best_estimate_t=best)
    return dict(reserves_t=low, basis="model_low", best_estimate_t=best)


def new_well_reserves(wm: pd.DataFrame, first_prod_ym: str, as_of_ym: str, tc: Dict,
                      q_econ: float, *, model: Optional[Dict] = None,
                      dyn_min_months: int = 12) -> Dict:
    """单口新井的剩余可采：动态法 / 类比法 / 模型法三法并列，按规则取值。"""
    w = wm[wm["ym"] <= as_of_ym].sort_values("ym")
    months_on = wl.ym_to_idx(as_of_ym) - wl.ym_to_idx(first_prod_ym) + 1
    rate = (w["oil_t"] / w["ym"].map(wl.days_in_month)).to_numpy(float)
    cum = float(w["oil_t"].sum())
    methods: Dict = {}

    tcr = np.asarray(tc.get("rate", []), dtype=float)
    if len(w) and len(tcr) > months_on:
        k = (w["ym"].map(wl.ym_to_idx) - wl.ym_to_idx(first_prod_ym) + 1).to_numpy(int)
        sel = (k >= 1) & (k <= len(tcr))
        denom = float(tcr[k[sel] - 1].sum())
        if denom > 0:
            scale = float(rate[sel].sum() / denom)
            methods["analog"] = dict(scale=scale,
                                     forecast=_analog_forecast(tcr, months_on, scale).tolist())

    if len(rate) >= dyn_min_months:
        kp = int(np.argmax(rate))
        post = rate[kp:]
        if len(post) >= 6 and np.isfinite(post).all() and (post > 0).sum() >= 6:
            t = np.arange(len(post), dtype=float)
            # 必须带终端递减：不设 d_min 的双曲在 b→1 时近似调和递减，
            # 50 年评估期内一直高于经济极限，短历史井的剩余可采会被放大数倍
            fit = dca.fit_best(t, post, candidates=("modified_hyperbolic", "exponential"))
            methods["dynamic"] = dict(fit=fit.to_dict(), t_now_month=float(len(post) - 1),
                                      r2=fit.r2)

    if model:
        methods["model"] = dict(remaining_low_t=max(float(model["p10"]) - cum, 0.0),
                                remaining_p50_t=max(float(model["p50"]) - cum, 0.0))

    pick = _pick_new_well_reserves(methods, q_econ)
    for key in ("analog", "dynamic"):          # 各法单独的结果也给出来，便于交叉核对
        if key in methods:
            methods[key]["remaining_t"] = _pick_new_well_reserves(
                {key: methods[key]}, q_econ)["best_estimate_t"]
    return dict(first_prod_ym=first_prod_ym, months_on=int(months_on), cum_t=cum,
                rate_now_t_per_d=float(rate[-3:].mean()) if len(rate) else 0.0,
                methods=methods, **pick)


# --------------------------------------------------------------------------- #
# 老井基础
def _well_decline_fit(t: np.ndarray, q: np.ndarray, min_tail: int) -> Optional[Dict]:
    """单井峰后递减拟合：从峰值月起，再用分段回归跳过早期措施造成的台阶。"""
    ok = np.isfinite(q) & (q > 0)
    if ok.sum() < min_tail:
        return None
    kp = int(np.argmax(np.where(ok, q, -1.0)))
    tt, qq = t[kp:], q[kp:]
    good = np.isfinite(qq) & (qq > 0)
    tt, qq = tt[good], qq[good]
    if len(tt) < min_tail:
        return None
    s0 = dca.select_start(tt - tt[0], qq, min_tail=min_tail)["start_index"]
    xs, qs = tt[s0:], qq[s0:]
    best = dca.fit_best(xs - xs[0], qs, candidates=("modified_hyperbolic", "exponential"))
    low = dca.fit(xs - xs[0], qs, model="exponential")
    return dict(best=best.to_dict(), low=low.to_dict(), fit_start_idx=float(xs[0]),
                n_fit_months=int(len(xs)))


def _well_remaining(fit: Dict, q_econ: float, now_idx: float,
                    scale: float = 1.0, decline_delta: float = 0.0) -> Dict[str, float]:
    """按经济极限截断的剩余可采；scale / decline_delta 供敏感性分析扰动。"""
    t_now = now_idx - fit["fit_start_idx"]
    out = {}
    for key in ("best", "low"):
        f = dca.DCAFit(**fit[key])
        if scale != 1.0 or decline_delta:
            f = replace(f, qi=f.qi * scale, di=max(f.di + decline_delta / 12.0, 1e-5),
                        d_min=max(f.d_min + decline_delta / 12.0, 1e-5) if f.d_min > 0 else 0.0)
        out[key] = float(dca.eur(f, q_econ, t_start_month=t_now)["remaining"])
    out["low"] = min(out["low"], out["best"])
    return out


def old_wells_reserves(by_well: Dict[str, pd.DataFrame], old_ids: Sequence[str],
                       effects: Sequence[Dict], as_of_ym: str, q_econ: float, *,
                       min_fit_months: int = 12, idle_months: int = 3,
                       tc: Optional[Dict] = None, first_prod: Optional[Dict[str, str]] = None,
                       fit_cache: Optional[Dict] = None) -> Dict:
    """老井基础 PDP：逐井动态法，截到单井经济极限后加总。

    · 近 idle_months 个月没有产量的井不计入 PDP（井筒在、未生产，属 PDNP）；
    · 本期做过措施且有基线的井，用**措施前**的数据外推 —— 措施带来的部分单列在措施增储里，
      这里再用措施后的数据拟合就会重复计算；
    · 峰后历史不够做递减拟合的井（上一评估期才投产），与新井同规则用类型曲线类比，
      不能因为拟合不了就记成 0。
    fit_cache：递减拟合与经济参数无关，同一基准日的多个价格情景可共用，由调用方传入同一个 dict。
    """
    now = wl.ym_to_idx(as_of_ym)
    cutoff = {e["well_id"]: e["event_ym"] for e in effects
              if e["in_period"] and e["status"] in ("ok", "insufficient_post")}
    rows: List[Dict] = []
    for wid in old_ids:
        wm = by_well.get(wid)
        if wm is None:
            continue
        idx = wm["ym"].map(wl.ym_to_idx).to_numpy(float)
        oil = wm["oil_t"].to_numpy(float)
        if oil[idx > now - idle_months].sum() <= 0:
            rows.append(dict(well_id=wid, status="not_producing", remaining_low_t=0.0,
                             remaining_best_t=0.0))
            continue
        key = (wid, as_of_ym, cutoff.get(wid, ""), min_fit_months)
        if fit_cache is not None and key in fit_cache:
            fit = fit_cache[key]
        else:
            sel = idx < wl.ym_to_idx(cutoff[wid]) if wid in cutoff else np.ones(len(idx), bool)
            rate = oil / wm["ym"].map(wl.days_in_month).to_numpy(float)
            fit = _well_decline_fit(idx[sel], rate[sel], min_fit_months)
            if fit_cache is not None:
                fit_cache[key] = fit
        if fit is None:
            # 类比法同样只看措施前的数据：否则措施后的抬升会进老井基础，与措施增储重复
            base_wm = wm[wm["ym"] < cutoff[wid]] if wid in cutoff else wm
            res = (new_well_reserves(base_wm, first_prod[wid], as_of_ym, tc, q_econ)
                   if tc and first_prod and wid in first_prod and len(base_wm) else None)
            if res is None or res["reserves_t"] is None:
                rows.append(dict(well_id=wid, status="insufficient", remaining_low_t=0.0,
                                 remaining_best_t=0.0))
            else:
                rows.append(dict(well_id=wid, status="short_history", basis=res["basis"],
                                 methods=res["methods"], remaining_low_t=res["reserves_t"],
                                 remaining_best_t=res["reserves_t"],
                                 rate_now_t_per_d=res["rate_now_t_per_d"]))
            continue
        rem = _well_remaining(fit, q_econ, now)
        rate_now = float(dca.rate(np.array([now - fit["fit_start_idx"]]),
                                  dca.DCAFit(**fit["best"]))[0])
        rows.append(dict(well_id=wid, status="ok",
                         basis="measure_baseline" if wid in cutoff else "own_history",
                         fit=fit, remaining_low_t=rem["low"], remaining_best_t=rem["best"],
                         rate_now_t_per_d=rate_now))
    ok = [r for r in rows if r["status"] in ("ok", "short_history")]
    return dict(status="ok" if ok else "insufficient",
                n_evaluated=len(ok),
                n_short_history=sum(1 for r in rows if r["status"] == "short_history"),
                n_not_producing=sum(1 for r in rows if r["status"] == "not_producing"),
                n_insufficient=sum(1 for r in rows if r["status"] == "insufficient"),
                n_measure_baseline=sum(1 for r in ok if r["basis"] == "measure_baseline"),
                remaining_low_t=float(sum(r["remaining_low_t"] for r in rows)),
                remaining_best_t=float(sum(r["remaining_best_t"] for r in rows)),
                rate_now_t_per_d=float(sum(r["rate_now_t_per_d"] for r in ok)),
                wells=rows)


# --------------------------------------------------------------------------- #
# 单元评估
def evaluate_unit(*, monthly: pd.DataFrame, master: pd.DataFrame, events: pd.DataFrame,
                  new_wells: pd.DataFrame, as_of: str, q_econ_well: float, period_months: int,
                  measure_cfg: Dict, min_fit_months: int, type_curve_min_history: int,
                  model_estimates: Optional[Dict[str, Dict]] = None,
                  fit_cache: Optional[Dict] = None) -> Dict:
    """一个 SEC 单元在一个基准日、一套经济参数下的"新-老-措"构成评估。

    master / events / monthly 传单元内的井；new_wells 是 identify_new_wells 的结果（可含单元外的井）。
    """
    as_of_ym = str(as_of)[:7]
    start_ym, _ = wl.period_bounds(as_of, period_months)
    ids = set(master["well_id"])
    m = monthly[monthly["well_id"].isin(ids) & (monthly["ym"] <= as_of_ym)]
    by_well = {w: g.sort_values("ym") for w, g in m.groupby("well_id")}

    fp = master.set_index("well_id")["first_prod_date"].astype(str).str[:7]
    ev = events[events["well_id"].isin(ids) & (events["dt"].astype(str).str[:7] <= as_of_ym)]
    effects: List[Dict] = []
    for r in ev.itertuples(index=False):
        wm = by_well.get(r.well_id)
        if wm is None:
            continue
        e = wl.measure_effect(wm, str(r.dt)[:7], as_of_ym, q_econ_well, **measure_cfg)
        # in_period = 计入本期措施增储。本期新井上的措施不算：新井储量已包含其效果，
        # 与 monthly_composition 跳过新井措施是同一条规则
        on_new = fp.get(r.well_id, "") >= start_ym
        e.update(well_id=r.well_id, event_type=r.event_type, on_new_well=on_new,
                 in_period=str(r.dt)[:7] >= start_ym and not on_new)
        effects.append(e)

    first_ym = m["ym"].min() if len(m) else start_ym
    comp = wl.monthly_composition(m, master, effects, start_ym=first_ym, end_ym=as_of_ym,
                                  new_since_ym=start_ym, measure_since_ym=start_ym)
    tc = type_curve(m, master, as_of_ym, min_history_months=type_curve_min_history)
    old_ids = [w for w in fp.index if fp[w] < start_ym]
    old = old_wells_reserves(by_well, old_ids, effects, as_of_ym, q_econ_well,
                             min_fit_months=min_fit_months, tc=tc, first_prod=fp.to_dict(),
                             fit_cache=fit_cache)

    nw = new_wells[new_wells["well_id"].isin(ids)]
    wells: List[Dict] = []
    for r in nw.itertuples(index=False):
        wm = by_well.get(r.well_id)
        if wm is None:
            continue
        res = new_well_reserves(wm, str(r.first_prod_date)[:7], as_of_ym, tc, q_econ_well,
                                model=(model_estimates or {}).get(r.well_id))
        res.update(well_id=r.well_id, category=r.category, n_old_neighbors=int(r.n_old_neighbors))
        wells.append(res)

    in_period = m[m["ym"] >= start_ym]
    prod_by_well = in_period.groupby("well_id")["oil_t"].sum()
    period_measures = [e for e in effects if e["in_period"]]
    components = [
        dict(key="old_base", name=COMPONENTS["old_base"], reserves_t=old["remaining_low_t"],
             best_estimate_t=old["remaining_best_t"], n_items=old["n_evaluated"],
             basis="逐井动态法（指数递减与最佳估计取低值）；峰后历史不足的井用类比法"),
        dict(key="measure", name=COMPONENTS["measure"],
             reserves_t=float(sum(e["inc_remaining_t"] for e in period_measures
                                  if e["status"] == "ok")),
             best_estimate_t=None,
             n_items=sum(1 for e in period_measures if e["status"] == "ok"),
             basis="本期措施增加的剩余可采（观察期不足的措施不计）"),
    ]
    prod_by_comp = {}
    for key, cat in (("new_infill", "infill"), ("extension", "extension")):
        sub = [w for w in wells if w["category"] == cat]
        components.append(dict(key=key, name=COMPONENTS[key],
                               reserves_t=float(sum(w["reserves_t"] or 0.0 for w in sub)),
                               best_estimate_t=None, n_items=len(sub),
                               basis="动态法/类比法最佳估计与模型法低估计取低值"))
        prod_by_comp[key] = float(prod_by_well.reindex([w["well_id"] for w in sub]).fillna(0).sum())

    return dict(as_of=str(as_of), period_start_ym=start_ym, period_end_ym=as_of_ym,
                q_econ_well_t_per_d=float(q_econ_well), n_wells=len(ids),
                components=components,
                total_t=float(sum(c["reserves_t"] for c in components)),
                production_in_period_t=float(in_period["oil_t"].sum()),
                production_by_component=prod_by_comp,
                old_base=old, measures=effects, new_wells=wells, type_curve=tc,
                composition=comp.to_dict("records"))


# --------------------------------------------------------------------------- #
# 对账、折耗、敏感性、归因
def reconcile_auto(opening: Dict, closing: Dict, closing_at_opening_price: Dict) -> Dict:
    """期初 → 期末自动对账。各行项由构成评估独立算出，技术修订为轧差项。

    closing_at_opening_price：用期末的数据、期初的价格与成本重算的期末储量，
    二者之差即价格/成本修订 —— 把"油价变了"和"井变了"分开。
    """
    comp = {c["key"]: c for c in closing["components"]}
    measure_add = float(sum(e["inc_remaining_t"] + e["realized_inc_t"] for e in closing["measures"]
                            if e["in_period"] and e["status"] == "ok"))
    values = dict(
        opening=float(opening["total_t"]),
        production=-float(closing["production_in_period_t"]),
        price_revision=float(closing["total_t"] - closing_at_opening_price["total_t"]),
        new_wells=float(comp["new_infill"]["reserves_t"]
                        + closing["production_by_component"]["new_infill"]),
        measure=measure_add,
        extension=float(comp["extension"]["reserves_t"]
                        + closing["production_by_component"]["extension"]),
        category_change=0.0,
    )
    values["technical_revision"] = float(closing["total_t"]) - sum(values.values())
    values["closing"] = float(closing["total_t"])
    table = rec.build(values)
    prod = float(closing["production_in_period_t"])
    denom = float(closing["total_t"]) + prod
    return dict(**table,
                depletion_rate_pct=(100.0 * prod / denom) if denom > 0 else None,
                depletion_note="产量法折耗率 = 本期产量 /（期末已证实已开发储量 + 本期产量）；"
                               "折耗额还需资产账面价值，本平台不掌握",
                category_change_note="合成数据不含复产与 PUD 转 PDP 记录，类别调整计 0")


def sensitivity(ev: Dict, q_econ_fn: Callable[[float, float], float], base_price: float) -> Dict:
    """油价、成本、产量、递减率四个参数对单元 PDP 的影响（最佳估计口径的变化量）。

    q_econ_fn(price_usd_bbl, opex_factor) -> 单井经济极限 t/d。
    产量与递减率只作用于老井基础；油价与成本通过经济极限作用于全部构成。
    模型法低估计与经济参数无关，保持不变。
    """
    old = ev["old_base"]
    now = wl.ym_to_idx(ev["period_end_ym"])
    q_now = old.get("rate_now_t_per_d", 0.0)
    fitted = [r for r in old["wells"] if r["status"] == "ok"]
    short = [r for r in old["wells"] if r["status"] == "short_history"]

    def total(price: float = base_price, opex_f: float = 1.0,
              rate_delta_t_month: float = 0.0, decline_delta: float = 0.0) -> float:
        qe = q_econ_fn(price, opex_f)
        scale = max(q_now + rate_delta_t_month / wl.DAYS_PER_MONTH, 1e-6) / max(q_now, 1e-9)
        val = sum(_well_remaining(r["fit"], qe, now, scale, decline_delta)["best"] for r in fitted)
        # 类比法的井没有自身递减参数，产量与递减率扰动不作用于它们，只随经济极限变化
        val += sum(_pick_new_well_reserves(r["methods"], qe)["reserves_t"] or 0.0 for r in short)
        val += sum(wl.measure_increment_remaining(e, qe) for e in ev["measures"]
                   if e["in_period"] and e["status"] == "ok")
        val += sum(_pick_new_well_reserves(w["methods"], qe)["reserves_t"] or 0.0
                   for w in ev["new_wells"])
        return float(val)

    base = total()
    monthly_prod = q_now * wl.DAYS_PER_MONTH
    specs = [
        ("price", "油价", "USD/bbl", [base_price * k for k in np.linspace(0.8, 1.2, 9)],
         lambda v: total(price=v), (base_price * 1.1, base_price * 0.9)),
        ("opex", "操作成本系数", "倍", list(np.linspace(0.8, 1.2, 9)),
         lambda v: total(opex_f=v), (1.1, 0.9)),
        ("rate", "老井月产增幅", "t/月", [monthly_prod * k for k in np.linspace(-0.2, 0.2, 9)],
         lambda v: total(rate_delta_t_month=v), (monthly_prod * 0.1, -monthly_prod * 0.1)),
        ("decline", "递减率变化", "年化百分点", [float(k) for k in range(-3, 4)],
         lambda v: total(decline_delta=v / 100.0), (1.0, -1.0)),
    ]
    curves, swings = [], {}
    for key, name, unit, grid, fn, (hi, lo) in specs:
        curves.append(dict(param=key, name=name, unit=unit, x=[float(v) for v in grid],
                           delta_t=[fn(v) - base for v in grid]))
        # 敏感权重：典型扰动（油价/成本/产量 ±10%、递减率 ±1 个百分点）下的 PDP 摆幅
        swings[key] = abs(fn(hi) - fn(lo))
    tot = sum(swings.values()) or 1.0
    return dict(base_best_estimate_t=base, curves=curves,
                weights=[dict(param=c["param"], name=c["name"], swing_t=swings[c["param"]],
                              weight=swings[c["param"]] / tot) for c in curves],
                note="敏感性按最佳估计计算变化量；权重为典型扰动下 PDP 摆幅的占比")


def change_attribution(recon: Dict, closing: Dict, tech_evidence: List[Dict],
                       econ_open: Dict, econ_close: Dict, top_k: int = 3) -> Dict:
    """单元 PDP 为什么变了：按行项排序，每项落到具体的井、措施或参数。"""
    rows = {r["key"]: r["value"] for r in recon["table"]}
    drivers: List[Dict] = []

    def add(key: str, evidence: List[Dict], kind: str):
        drivers.append(dict(key=key, item=next(r["item"] for r in recon["table"] if r["key"] == key),
                            value_t=float(rows.get(key, 0.0)), evidence_kind=kind, evidence=evidence))

    for key, cat in (("new_wells", "infill"), ("extension", "extension")):
        ws = sorted((w for w in closing["new_wells"] if w["category"] == cat),
                    key=lambda w: -(w["reserves_t"] or 0.0))[:top_k]
        add(key, [dict(well_id=w["well_id"], reserves_t=w["reserves_t"], basis=w["basis"],
                       months_on=w["months_on"]) for w in ws], "wells")
    ms = sorted((e for e in closing["measures"] if e["in_period"] and e["status"] == "ok"),
                key=lambda e: -abs(e["inc_eur_t"]))[:top_k]
    add("measure", [dict(well_id=e["well_id"], event_type=e["event_type"], event_ym=e["event_ym"],
                         inc_eur_t=e["inc_eur_t"], rate_gain_t_per_d=e["rate_gain_t_per_d"])
                    for e in ms], "measures")
    add("price_revision", [dict(price_open_usd_bbl=econ_open["price_usd_bbl"],
                                price_close_usd_bbl=econ_close["price_usd_bbl"],
                                q_econ_open_t_per_d=econ_open["q_econ"],
                                q_econ_close_t_per_d=econ_close["q_econ"])], "economics")
    add("technical_revision", tech_evidence, "wells_rate_change")
    add("production", [], "none")

    change = float(rows["closing"] - rows["opening"])
    denom = sum(abs(d["value_t"]) for d in drivers) or 1.0
    for d in drivers:
        d["share_pct"] = 100.0 * abs(d["value_t"]) / denom
    drivers.sort(key=lambda d: -abs(d["value_t"]))
    return dict(opening_t=float(rows["opening"]), closing_t=float(rows["closing"]),
                change_t=change, drivers=drivers,
                note="占比为各行项绝对值占全部行项绝对值之和的比例")
