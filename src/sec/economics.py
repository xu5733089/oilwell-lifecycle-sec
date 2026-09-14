"""SEC 经济极限计算（方案 §6.2）。

价格口径按 Reg S-X 4-10：报告期前 12 个月每月首日价格的**未加权算术平均**，
不含合同约定的价格递增。成本用当前成本，不考虑通胀预期。

产量剖面必须在"净收入 = 操作成本"处截断 —— 不截断的 EUR 在准则下不能算储量。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..config import price_decks
from ..reserves import dca


def deck(price_deck_id: str) -> Dict:
    decks = price_decks()
    if price_deck_id not in decks:
        raise KeyError(f"未知价格册 {price_deck_id!r}，可选：{list(decks)}")
    return decks[price_deck_id]


def avg_12m_price(price_deck_id: str) -> float:
    d = deck(price_deck_id)
    prices = d["monthly_first_day_prices_usd_bbl"]
    if len(prices) != 12:
        raise ValueError(f"价格册 {price_deck_id} 需要恰好 12 个月首日价格，当前 {len(prices)} 个")
    return float(np.mean(prices))          # 未加权算术平均


def deck_for(as_of: str) -> str:
    """按评估基准日找价格册。找不到就明确报错，不拿别的年份顶替。"""
    for k, d in price_decks().items():
        if str(d["as_of"]) == str(as_of):
            return k
    raise KeyError(f"评估基准日 {as_of} 没有对应的价格册，可选基准日："
                   f"{sorted(str(d['as_of']) for d in price_decks().values())}")


def scenario_params(price_deck_id: str, scenario: str = "sec") -> Dict:
    """三套价格与成本参数。sec 情景的价格固定取 12 个月首日均价，配置里写了也不采用。"""
    d = deck(price_deck_id)
    sc = d.get("scenarios") or {"sec": {"label": "SEC 价"}}
    if scenario not in sc:
        raise KeyError(f"价格册 {price_deck_id} 没有情景 {scenario!r}，可选：{list(sc)}")
    s = sc[scenario] or {}
    price = avg_12m_price(price_deck_id) if scenario == "sec" else float(s["price_usd_bbl"])
    return dict(scenario=scenario, label=s.get("label", scenario), price_usd_bbl=price,
                opex_factor=float(s.get("opex_factor", 1.0)))


def net_revenue_per_tonne(price_deck_id: str, price_usd_bbl: Optional[float] = None) -> float:
    d = deck(price_deck_id)
    price = avg_12m_price(price_deck_id) if price_usd_bbl is None else float(price_usd_bbl)
    net_bbl = (price - d["transport_treat_usd_bbl"]) * (1 - d["tax_rate"])
    return float(net_bbl * d["bbl_per_tonne"])


def economic_limit_rate(price_deck_id: str, scenario: str = "sec", opex_factor: float = 1.0,
                        price_usd_bbl: Optional[float] = None) -> Dict[str, float]:
    """q_econ = 月操作成本 / (单位净收入 × 30.4)，单位 t/d（单井口径）。

    opex_factor：单元成本系数（叠乘在情景成本系数之上）。
    price_usd_bbl：仅供敏感性分析覆盖价格；正式评估不传。
    """
    d = deck(price_deck_id)
    sc = scenario_params(price_deck_id, scenario)
    price = sc["price_usd_bbl"] if price_usd_bbl is None else float(price_usd_bbl)
    net_t = net_revenue_per_tonne(price_deck_id, price)
    opex = float(d["opex_per_well_month_usd"]) * sc["opex_factor"] * float(opex_factor)
    q_econ = opex / max(net_t * 30.4, 1e-9)
    return dict(q_econ=float(q_econ),
                avg_12m_price_usd_bbl=round(avg_12m_price(price_deck_id), 3),
                price_usd_bbl=round(price, 3),
                net_revenue_usd_per_tonne=round(net_t, 2),
                opex_per_well_month_usd=round(opex, 2),
                price_deck_id=price_deck_id, as_of=d["as_of"],
                scenario=scenario, scenario_label=sc["label"])


def truncate_reserves(fit: dca.DCAFit, price_deck_id: str, t_now_month: float = 0.0,
                      quantile_scale: float = 1.0) -> Dict[str, float]:
    """把递减曲线截断到经济极限，返回 EUR 与剩余可采。

    quantile_scale: 用 EUR 概率区间的比例缩放曲线，得到 P90 口径的剩余储量。
    """
    econ = economic_limit_rate(price_deck_id)
    e = dca.eur(fit, econ["q_econ"], t_start_month=t_now_month)
    return dict(eur=e["eur"] * quantile_scale,
                remaining=e["remaining"] * quantile_scale,
                t_econ_month=e["t_econ_month"], **econ)
