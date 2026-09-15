"""产量法折耗与减值测试（纯计算，不读库）。

金额单位统一为万元；价格与成本来自价格册（美元），按配置汇率折算。

· 产量法折耗率 = 本期产量 /（期末证实已开发储量 + 本期产量）
  证实已开发储量 = PDP + PDNP —— 已经投入的井筒对应的是全部已开发储量，不只是在产的那部分。
· 折耗额 =（期初资产净值 + 本期资本化投入）× 折耗率
· 可收回金额 = 减值测试价下证实已开发储量未来净现金流的折现值。
  剖面简化为单元级指数递减：初始月产取近 3 个月实际月均，递减率由"剖面累计 = 储量"反解，
  净现金流 =（吨油净收入 − 吨油操作成本）× 月产。不含 PUD 未来投资与弃置费。
· 减值额 = max(0, 折耗后账面价值 − 可收回金额)
"""
from __future__ import annotations

import math
from typing import Dict, Optional


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
