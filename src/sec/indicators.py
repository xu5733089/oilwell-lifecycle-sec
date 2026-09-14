"""开发与经营指标计算与评分（运行监控雷达图）。

指标值由调用方用内核结果组装后传入，这里只做算术与按 conf/indicators.yaml 打分。
"""
from __future__ import annotations

from typing import Dict, Optional


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return 100.0 * float(a) / float(b)


def compute(*, production_t: float, plan_production_t: Optional[float],
            natural_decline_pct: Optional[float], comprehensive_decline_pct: Optional[float],
            water_cut_pct: Optional[float], water_cut_prev_pct: Optional[float],
            n_wells: int, n_producing: int, new_oil_t: float, plan_new_oil_t: Optional[float],
            n_measures_evaluated: int, n_measures_effective: int, pdp_t: float,
            producing_well_months: float, opex_per_well_month_usd: float,
            net_revenue_usd_per_t: float) -> Dict[str, Optional[float]]:
    opex_t = (producing_well_months * opex_per_well_month_usd / production_t) if production_t > 0 else None
    return dict(
        natural_decline_pct=natural_decline_pct,
        comprehensive_decline_pct=comprehensive_decline_pct,
        water_cut_pct=water_cut_pct,
        water_cut_rise_pct=(None if water_cut_pct is None or water_cut_prev_pct is None
                            else float(water_cut_pct - water_cut_prev_pct)),
        open_well_rate_pct=_pct(n_producing, n_wells),
        measure_effective_rate_pct=_pct(n_measures_effective, n_measures_evaluated),
        reserve_production_ratio=(float(pdp_t / production_t) if production_t > 0 else None),
        production_plan_attainment_pct=_pct(production_t, plan_production_t),
        new_well_plan_attainment_pct=_pct(new_oil_t, plan_new_oil_t),
        opex_usd_per_t=opex_t,
        net_revenue_usd_per_t=float(net_revenue_usd_per_t),
        profit_usd_per_t=(None if opex_t is None else float(net_revenue_usd_per_t - opex_t)),
    )


def score(values: Dict[str, Optional[float]], spec: Dict) -> Dict:
    """按锚点线性打分到 0~100；取不到值的指标不参与评分，也不参与分组综合分。"""
    rows, groups = [], {}
    for key, s in spec["indicators"].items():
        v = values.get(key)
        sc = None
        if v is not None and s["good"] != s["bad"]:
            sc = max(0.0, min(100.0, (float(v) - s["bad"]) / (s["good"] - s["bad"]) * 100.0))
        rows.append(dict(key=key, name=s["name"], group=s["group"], unit=s["unit"],
                         value=v, score=sc, good=s["good"], bad=s["bad"], weight=s["weight"]))
        if sc is not None:
            g = groups.setdefault(s["group"], [0.0, 0.0])
            g[0] += sc * s["weight"]
            g[1] += s["weight"]
    return dict(indicators=rows,
                groups=[dict(group=k, name=spec.get("groups", {}).get(k, k),
                             score=(v[0] / v[1]) if v[1] else None)
                        for k, v in groups.items()])
