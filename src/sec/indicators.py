"""开发与经营指标计算与评分（运行监控雷达图）。

指标值由调用方用内核结果组装后传入，这里只做算术与按 conf/indicators.yaml 打分。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional


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


def validate_spec(spec: Dict, reference: Dict) -> List[str]:
    """锚点方案校验：指标集合须与内置口径一致（算法只会算这些指标），锚点为数值且优值≠差值，权重非负。"""
    errors: List[str] = []
    ind = (spec or {}).get("indicators") or {}
    ref = reference["indicators"]
    missing, extra = sorted(set(ref) - set(ind)), sorted(set(ind) - set(ref))
    if missing:
        errors.append("缺少指标：" + "、".join(ref[k]["name"] for k in missing))
    if extra:
        errors.append("未知指标：" + "、".join(extra))
    groups: Dict[str, float] = {}
    for key, s in ind.items():
        if key not in ref:
            continue
        name = ref[key]["name"]
        try:
            good, bad, weight = float(s["good"]), float(s["bad"]), float(s.get("weight", 0))
        except (KeyError, TypeError, ValueError):
            errors.append(f"{name}：优值锚点、差值锚点、权重须为数值")
            continue
        if not all(map(math.isfinite, (good, bad, weight))):
            errors.append(f"{name}：锚点与权重须为有限数值")
        elif good == bad:
            errors.append(f"{name}：优值锚点与差值锚点不能相等")
        if weight < 0:
            errors.append(f"{name}：权重不能为负")
        groups[ref[key]["group"]] = groups.get(ref[key]["group"], 0.0) + max(weight, 0.0)
    for g, w in groups.items():
        if w <= 0:
            errors.append(f"{reference.get('groups', {}).get(g, g)}：组内权重之和须大于 0")
    return errors


def normalize_spec(spec: Dict, reference: Dict) -> Dict:
    """只保留可调的锚点与权重；名称、分组、单位一律取内置口径，防止被改名混淆。"""
    ind = {}
    for key, r in reference["indicators"].items():
        s = spec["indicators"][key]
        ind[key] = dict(r, good=float(s["good"]), bad=float(s["bad"]), weight=float(s["weight"]))
    return dict(version=reference.get("version"), indicators=ind, groups=dict(reference.get("groups", {})))


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
