"""储量对账表（方案 §6.5）。

期初 → 产量消耗 → 价格/成本修订 → 新井增储 → 措施增储 → 扩边 → 类别调整 → 技术修订 → 期末，
自动核对是否闭合。这是 SEC 20-F 里真实存在的表，也是管理者最想看的一页。

自动对账（src/sec/composition.reconcile_auto）里技术修订是轧差项，与 20-F 的
"revisions of previous estimates" 口径一致；手工传入全部行项时闭合检查才有校验意义。
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

ROWS = [
    ("opening", "期初已证实储量", "上一评估期结果"),
    ("production", "本期产量消耗", "实际累产"),
    ("price_revision", "价格/成本修订", "价格册与成本变化导致的经济极限移动"),
    ("new_wells", "提采新井增储", "本期投产的提采新井：期末储量 + 本期产量"),
    ("measure", "措施增储", "本期措施：增加的剩余可采 + 本期已实现增油"),
    ("extension", "扩边与新发现", "本期投产的扩边井：期末储量 + 本期产量"),
    ("category_change", "类别调整", "PUD 转 PDP 等"),
    ("technical_revision", "技术修订", "递减规律变化、停井等导致的调整"),
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
