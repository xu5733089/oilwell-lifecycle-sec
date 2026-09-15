"""产量法折耗与减值测试（纯计算，不读库）。

金额单位统一为万元；价格与成本来自价格册（美元），按配置汇率折算。

· 产量法折耗率 = 本期产量 /（期末证实已开发储量 + 本期产量）
  证实已开发储量 = PDP + PDNP —— 已经投入的井筒对应的是全部已开发储量，不只是在产的那部分。
· 折耗额 =（期初资产净值 + 本期资本化投入）× 折耗率
· 可收回金额 = 减值测试价下证实已开发储量未来净现金流的折现值。
  剖面简化为单元级指数递减：初始月产取近 3 个月实际月均，递减率由"剖面累计 = 储量"反解，
  净现金流 =（吨油净收入 − 吨油操作成本）× 月产。不含 PUD 未来投资与弃置费。
· 减值额 = max(0, 折耗后账面价值 − 可收回金额)
· 标准化计量（FASB ASC 932 "standardized measure of discounted future net cash flows"）：
  未来现金流入 − 未来生产成本 − 未来开发成本 − 未来所得税 = 未来净现金流，按 10% 年率折现（年中折现）。
  剖面与减值测试一致：已开发部分用单元级指数剖面；未开发部分逐井位用类型曲线，从计划钻井月起投产，投资记在投产前一个月。
  所得税按年对正的税前现金流乘税率，不考虑税基、亏损结转与抵扣 —— 简化口径，在返回体里写明。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


def depletion(opening_nbv_wan: float, capex_additions_wan: float, production_t: float,
              pd_reserves_t: float) -> Dict[str, Optional[float]]:
    base = float(opening_nbv_wan) + float(capex_additions_wan)
    denom = float(pd_reserves_t) + float(production_t)
    rate = float(production_t) / denom if denom > 0 else None
    amount = base * rate if rate is not None else 0.0
    return dict(depletion_base_wan=base, depletion_rate=rate, depletion_wan=amount,
                carrying_after_depletion_wan=base - amount)


def recoverable_amount(pd_reserves_t: float, monthly_rate_t: float, net_revenue_usd_per_t: float,
                       opex_usd_per_t: float, discount_rate: float, fx_cny_per_usd: float) -> Dict:
    """证实已开发储量折现现金流。月产或储量为零时可收回金额为 0，并说明原因。"""
    reserves, q0 = float(pd_reserves_t), float(monthly_rate_t)
    cash_per_t = float(net_revenue_usd_per_t) - float(opex_usd_per_t)
    base = dict(cash_usd_per_t=cash_per_t, discount_rate=float(discount_rate))
    if reserves <= 0 or q0 <= 0:
        return dict(base, recoverable_wan=0.0, decline_month=None, discounted_t=0.0, life_years=None,
                    note="无证实已开发储量或近期无产量")
    q0 = min(q0, reserves * 0.999)
    decline = -math.log(1.0 - q0 / reserves)                    # 使 Σ q0·e^(−D·t) = 储量
    a = math.exp(-decline) * (1.0 + float(discount_rate)) ** (-1.0 / 12.0)
    discounted_t = q0 / (1.0 - a)
    value = max(cash_per_t, 0.0) * discounted_t * float(fx_cny_per_usd) / 1e4
    return dict(base, recoverable_wan=value, decline_month=decline, discounted_t=discounted_t,
                life_years=math.log(20.0) / decline / 12.0,
                note=None if cash_per_t > 0 else "吨油净收入不足以覆盖操作成本，现金流为负，可收回金额按 0 计")


def impairment(carrying_wan: float, recoverable_wan: float) -> Dict[str, Optional[float]]:
    loss = max(float(carrying_wan) - float(recoverable_wan), 0.0)
    headroom = (float(recoverable_wan) - float(carrying_wan)) / float(carrying_wan) * 100.0 \
        if carrying_wan > 0 else None
    return dict(impairment_wan=loss, closing_nbv_wan=float(carrying_wan) - loss, headroom_pct=headroom,
                impaired=loss > 0)


SM_DISCOUNT_RATE = 0.10          # ASC 932 规定的标准化计量折现率


def exponential_profile(reserves_t: float, monthly_volume_t: float, horizon_months: int = 600) -> np.ndarray:
    """与可收回金额同一剖面：第 t 个月产量 q0·e^(−D·t)，D 使无穷级数之和等于储量。"""
    reserves, q0 = float(reserves_t), float(monthly_volume_t)
    if reserves <= 0 or q0 <= 0:
        return np.zeros(0)
    q0 = min(q0, reserves * 0.999)
    decline = -math.log(1.0 - q0 / reserves)
    return q0 * np.exp(-decline * np.arange(horizon_months))


def cash_flow_schedule(*, pd_volume_t: Sequence[float], pd_opex_usd_per_t: float, pud_wells: List[Dict],
                       opex_per_well_month_usd: float, net_revenue_usd_per_t: float, fx_cny_per_usd: float,
                       development_now_wan: float = 0.0, horizon_months: int = 600) -> Dict[str, np.ndarray]:
    """逐月现金流（万元）：已开发剖面 + 逐井位未开发剖面；pud_wells 每项 {start_month, volume_t, capex_wan}。"""
    n, k = int(horizon_months), float(fx_cny_per_usd) / 1e4
    vol, prod, dev = np.zeros(n), np.zeros(n), np.zeros(n)
    pv = np.asarray(pd_volume_t, float)[:n]
    vol[:len(pv)] += pv
    prod[:len(pv)] += pv * float(pd_opex_usd_per_t) * k
    for w in pud_wells:
        s = int(w["start_month"])
        if s >= n:
            continue
        wv = np.asarray(w["volume_t"], float)[: n - s]
        vol[s:s + len(wv)] += wv
        prod[s:s + len(wv)] += np.where(wv > 0, float(opex_per_well_month_usd) * k, 0.0)
        dev[max(s - 1, 0)] += float(w["capex_wan"])
    dev[0] += float(development_now_wan)
    return dict(volume_t=vol, inflow_wan=vol * float(net_revenue_usd_per_t) * k,
                production_cost_wan=prod, development_cost_wan=dev)


def standardized_measure(schedule: Dict[str, np.ndarray], tax_rate: float,
                         discount_rate: float = SM_DISCOUNT_RATE) -> Dict:
    n = len(schedule["volume_t"])
    starts = np.arange(0, n, 12)
    annual = {key: np.add.reduceat(np.asarray(schedule[key], float), starts)
              for key in ("volume_t", "inflow_wan", "production_cost_wan", "development_cost_wan")}
    alive = np.flatnonzero((annual["volume_t"] > 1e-9) | (annual["development_cost_wan"] > 0))
    life = int(alive.max()) + 1 if len(alive) else 0
    annual = {key: v[:life] for key, v in annual.items()}
    pre = annual["inflow_wan"] - annual["production_cost_wan"] - annual["development_cost_wan"]
    tax = float(tax_rate) * np.maximum(pre, 0.0)
    net = pre - tax
    factor = (1.0 + float(discount_rate)) ** -(np.arange(life) + 0.5)
    sm = float(np.sum(net * factor))
    vol_cum = np.cumsum(annual["volume_t"])
    life_vol = int(np.searchsorted(vol_cum, 0.995 * vol_cum[-1]) + 1) if life and vol_cum[-1] > 0 else 0
    dev_years = np.flatnonzero(annual["development_cost_wan"] > 0)
    life_report = max(life_vol, int(dev_years.max()) + 1 if len(dev_years) else 0)     # 产出 99.5% 所需年数（指数尾巴不算寿命）
    return dict(future_cash_inflows_wan=float(annual["inflow_wan"].sum()),
                future_production_costs_wan=float(annual["production_cost_wan"].sum()),
                future_development_costs_wan=float(annual["development_cost_wan"].sum()),
                future_income_tax_wan=float(tax.sum()), future_net_cash_flows_wan=float(net.sum()),
                discount_wan=float(net.sum()) - sm, standardized_measure_wan=sm,
                pre_tax_discounted_wan=float(np.sum(pre * factor)), discounted_tax_wan=float(np.sum(tax * factor)),
                volume_t=float(annual["volume_t"].sum()), life_years=life_report, tax_rate=float(tax_rate),
                discount_rate=float(discount_rate),
                annual=[dict(year=i + 1, volume_t=float(annual["volume_t"][i]), inflow_wan=float(annual["inflow_wan"][i]),
                             production_cost_wan=float(annual["production_cost_wan"][i]),
                             development_cost_wan=float(annual["development_cost_wan"][i]),
                             income_tax_wan=float(tax[i]), net_cash_flow_wan=float(net[i]),
                             discounted_wan=float(net[i] * factor[i])) for i in range(life)])
