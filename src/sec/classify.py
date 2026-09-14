"""储量分类规则引擎（方案 §6.3）。

纯规则实现，**不交给大模型判断**。
每条判定都返回依据字段清单与准则条款号，可被逐条核对。
"""
from __future__ import annotations

from typing import Dict, List, Optional

PDP, PDNP, PUD, NOT_PROVED = "PDP", "PDNP", "PUD", "NOT_PROVED"
PUD_MAX_YEARS = 5


def classify(well: Dict, prod_days: int, economic: bool,
             in_five_year_plan: Optional[bool] = None,
             offset_continuity_evidence: Optional[bool] = None) -> Dict:
    """
    well: 至少含 status（producing / shut_in / completed_not_producing / planned）
    prod_days: 有产量记录的天数
    economic: 当前产量是否高于经济极限
    """
    ev: List[str] = []
    status = str(well.get("status") or "").lower()

    if prod_days > 0 and status == "producing":
        if not economic:
            return _r(NOT_PROVED, ["当前产量已低于经济极限"], "Rule 4-10(a)(22)",
                      False, "产量低于经济极限，不满足『可经济开采』")
        ev = [f"有产量记录 {prod_days} 天", "status=producing", "当前产量高于经济极限"]
        return _r(PDP, ev, "Rule 4-10(a)(6)", False,
                  "已完井且正在生产，属已开发已生产储量")

    if prod_days > 0 and status == "shut_in":
        ev = [f"有产量记录 {prod_days} 天", "status=shut_in（井筒已存在，暂未生产）"]
        return _r(PDNP, ev, "Rule 4-10(a)(6)", True,
                  "井筒已存在但当前未生产，需人工确认复产条件与时间")

    if status == "completed_not_producing":
        ev = ["已完井未投产（待接管线 / 层未射开）"]
        return _r(PDNP, ev, "Rule 4-10(a)(6)", True,
                  "已开发未生产，需人工确认投产计划")

    if status == "planned" or prod_days == 0:
        if in_five_year_plan and offset_continuity_evidence:
            ev = ["未钻井位", f"已纳入 {PUD_MAX_YEARS} 年内开发计划", "相邻已探明井存在连续性证据"]
            return _r(PUD, ev, "Rule 4-10(a)(31)", True,
                      "满足五年规则与连续性证据，可初判为已探明未开发")
        missing = []
        if not in_five_year_plan:
            missing.append(f"未确认纳入 {PUD_MAX_YEARS} 年内开发计划")
        if not offset_continuity_evidence:
            missing.append("缺少相邻已探明井的连续性证据")
        return _r(NOT_PROVED, missing, "Rule 4-10(a)(31)", True,
                  "不满足 PUD 条件，不计入已证实储量")

    return _r(NOT_PROVED, [f"状态未知：{status!r}"], None, True, "状态字段缺失，需人工确认")


def _r(category: str, evidence: List[str], citation: Optional[str],
       needs_human: bool, rationale: str) -> Dict:
    return dict(category=category, evidence=evidence, citation=citation,
                needs_human=needs_human, rationale=rationale)
