"""储量对账表（方案 §6.5）。

期初 → 产量消耗 → 价格修订 → 技术修订 → 新增钻井 → 类别调整 → 期末，
自动核对是否闭合。这是 SEC 20-F 里真实存在的表，也是管理者最想看的一页。
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

ROWS = [
    ("opening", "期初已证实储量", "上一评估期结果"),
    ("production", "本期产量消耗", "实际累产"),
    ("price_revision", "价格修订", "价格册变化导致的经济极限移动"),
    ("technical_revision", "技术修订", "模型更新、递减参数重拟合导致的调整"),
    ("new_wells", "新增钻井", "新投产井带入"),
    ("category_change", "类别调整", "PUD 转 PDP 等"),
]


def build(values: Dict[str, float], tolerance: float = 1.0) -> Dict:
    opening = float(values.get("opening", 0.0))
    deltas = {k: float(values.get(k, 0.0)) for k, _, _ in ROWS if k != "opening"}
    closing_calc = opening + sum(deltas.values())
    closing_reported = float(values.get("closing", closing_calc))

    table: List[Dict] = []
    for key, label, note in ROWS:
        v = opening if key == "opening" else deltas[key]
        table.append(dict(key=key, item=label, value=round(v, 1), note=note))
    table.append(dict(key="closing", item="期末已证实储量",
                      value=round(closing_reported, 1), note="自动核对是否闭合"))

    diff = closing_reported - closing_calc
    return dict(table=table, closing_calculated=round(closing_calc, 1),
                closing_reported=round(closing_reported, 1),
                difference=round(diff, 1), balanced=abs(diff) <= tolerance,
                unit="t")


def to_frame(result: Dict) -> pd.DataFrame:
    return pd.DataFrame(result["table"])[["item", "value", "note"]]
