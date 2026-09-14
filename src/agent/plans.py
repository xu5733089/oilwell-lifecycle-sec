"""计划模板（方案 §7.2 第 3 段）。

每个意图对应一条**预定义的工具调用链**，大模型只填参数，不改结构。
好处是行为可控、可审计、可回归测试：同一个问题永远走同一条链路。

只保留一个 general_analysis 走受限 ReAct（最多 5 步、白名单工具）兜底开放问题；
本仓库默认不开，需要时在 config 里打开。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

MAX_REACT_STEPS = 5


@dataclass
class Step:
    tool: str
    args_from: Dict[str, str]                 # 目标参数 -> 槽位名
    const: Dict[str, Any] = None              # 固定参数
    optional: bool = False                    # 失败不阻断后续步骤

    def build_args(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        args = dict(self.const or {})
        for target, src in self.args_from.items():
            if src in slots:
                args[target] = slots[src]
        return args


PLANS: Dict[str, List[Step]] = {
    "query_well": [
        Step("query_well", {"well_code": "well_code", "fields": "fields"}),
    ],
    "predict_lifecycle": [
        Step("predict_lifecycle", {"well_code": "well_code", "obs_days": "obs_days",
                                   "targets": "targets"}),
        Step("find_analog_wells", {"well_code": "well_code"}, {"top_k": 3}, optional=True),
    ],
    "find_analogs": [
        Step("find_analog_wells", {"well_code": "well_code", "top_k": "top_k"}),
    ],
    "fit_dca": [
        Step("fit_dca", {"well_code": "well_code", "model": "model",
                         "price_deck_id": "price_deck_id"}),
    ],
    "estimate_reserves": [
        Step("estimate_reserves_volumetric", {"well_code": "well_code",
                                              "mc_samples": "mc_samples"}),
    ],
    "cross_check": [
        Step("cross_check_reserves", {"well_code": "well_code"}),
    ],
    # SEC 是唯一必须带条款检索的意图：合规结论不许无引用
    "sec_screen": [
        Step("sec_screen", {"well_code": "well_code", "as_of": "as_of",
                            "price_deck_id": "price_deck_id"}),
        Step("search_standard", {}, {"query": "已证实储量 合理确定性 经济可采 价格口径 储量分类",
                                     "top_k": 3}),
    ],
    "gen_report": [
        Step("query_well", {"well_code": "well_code"}),
        Step("predict_lifecycle", {"well_code": "well_code"}, optional=True),
        # 年轻井做不了递减分析是正常业务状态；sec_screen 自己会把"不可得"写进清单，
        # 这里设为必需步骤会让整份报告中断。
        Step("fit_dca", {"well_code": "well_code", "price_deck_id": "price_deck_id"},
             optional=True),
        Step("cross_check_reserves", {"well_code": "well_code"}, optional=True),
        Step("sec_screen", {"well_code": "well_code", "as_of": "as_of",
                            "price_deck_id": "price_deck_id"}),
        Step("search_standard", {}, {"query": "已证实储量 分类 五年规则 可靠技术", "top_k": 3}),
    ],
    # ---- SEC 单元级：储量构成与对账属于合规结论，必须带条款检索 ----
    "unit_composition": [
        Step("unit_sec_composition", {"scope": "scope", "as_of": "as_of", "scenario": "scenario"}),
        Step("search_standard", {}, {"query": "已证实已开发储量 PDP 合理确定性 经济可采 价格口径",
                                     "top_k": 3}),
    ],
    "unit_decline": [
        Step("unit_base_decline", {"scope": "scope", "as_of": "as_of",
                                   "exclude_years": "exclude_years"}),
    ],
    "measure_effect": [
        Step("unit_measure_effects", {"scope": "scope", "as_of": "as_of",
                                      "event_type": "event_type"}),
    ],
    "new_well_identify": [
        Step("unit_new_wells", {"scope": "scope", "as_of": "as_of"}),
    ],
    "unit_reconcile": [
        Step("unit_reconcile", {"scope": "scope", "from_as_of": "from_as_of",
                                "to_as_of": "to_as_of", "scenario": "scenario"}),
        Step("unit_change_attribution", {"scope": "scope", "from_as_of": "from_as_of",
                                         "to_as_of": "to_as_of", "scenario": "scenario"},
             optional=True),
        Step("search_standard", {}, {"query": "储量变化 产量 修订 扩边 新发现 价格", "top_k": 3}),
    ],
    "unit_sensitivity": [
        Step("unit_sensitivity", {"scope": "scope", "as_of": "as_of", "scenario": "scenario"}),
    ],
    "fallback": [],
}


def plan_for(intent: str) -> List[Step]:
    return PLANS.get(intent, [])


def describe(intent: str) -> List[str]:
    return [s.tool for s in plan_for(intent)]
