"""智能体评测集构建（方案 §10.2）。

结构：(问句, 期望意图, 期望工具链, 答案要点)。
五类构成，比例参考方案：正常问法 / 口语省略 / 跨模块组合 / 缺数据越权 / 诱导性提问。

**诱导性提问是最有价值的一类**：问一口不存在的井、要求预测未来油价、
让模型自己算个数——这些恰恰是幻觉最容易冒出来的地方，
也是"数值一致性通过率"和"无据结论率"两个指标的主战场。

生成器按井号模板批量铺开，业务方只需替换 TEMPLATES 里的问法即可扩充。
"""
from __future__ import annotations

from typing import Dict, List, Optional

# (问法模板, 期望意图, 类别)
TEMPLATES: List[tuple] = [
    # ---- 正常问法 ----
    ("{w} 这口井什么时候达峰？", "predict_lifecycle", "normal"),
    ("预测一下 {w} 的见油时间和达峰压力", "predict_lifecycle", "normal"),
    ("{w} 的全生命周期预测结果是什么", "predict_lifecycle", "normal"),
    ("给我看看 {w} 的 EUR 预测", "predict_lifecycle", "normal"),
    ("{w} 找几口类比井", "find_analogs", "normal"),
    ("跟 {w} 最相似的老井有哪些", "find_analogs", "normal"),
    ("对 {w} 做递减曲线分析", "fit_dca", "normal"),
    ("{w} 的 b 值和递减率是多少", "fit_dca", "normal"),
    ("算一下 {w} 的容积法地质储量", "estimate_reserves", "normal"),
    ("{w} 的 OOIP 是多少", "estimate_reserves", "normal"),
    ("{w} 动静态储量对得上吗", "cross_check", "normal"),
    ("帮我做 {w} 的储量互校", "cross_check", "normal"),
    ("{w} 的 SEC 储量预评估", "sec_screen", "normal"),
    ("{w} 能算已证实储量吗，按什么类别", "sec_screen", "normal"),
    ("查一下 {w} 的基本信息", "query_well", "normal"),
    ("{w} 现在什么情况，累产多少", "query_well", "normal"),
    ("给 {w} 出份评估报告", "gen_report", "normal"),

    # ---- 口语化 / 省略式 ----
    ("{w} 峰值啥时候到", "predict_lifecycle", "colloquial"),
    ("{w} 多久见油", "predict_lifecycle", "colloquial"),
    ("{w} 像哪几口井", "find_analogs", "colloquial"),
    ("{w} 递减快不快", "fit_dca", "colloquial"),
    ("{w} 储量多少", "estimate_reserves", "colloquial"),
    ("{w} 合规吗", "sec_screen", "colloquial"),
    ("{w} 含水多少了", "query_well", "colloquial"),

    # ---- 跨模块组合 ----
    ("{w} 先预测达峰，再给我找几口类比井对照", "predict_lifecycle", "composite"),
    ("{w} 做完递减分析后判断能不能进已证实储量", "sec_screen", "composite"),
    ("{w} 出份报告，要包含预测、储量和 SEC 判定", "gen_report", "composite"),
    ("{w} 的采收率和区块经验值比怎么样", "cross_check", "composite"),

    # ---- 缺数据 / 越权 ----
    ("BAD-X-9999 这口井什么时候达峰", "predict_lifecycle", "missing_data"),
    ("BAD-X-9999 的 SEC 储量评估", "sec_screen", "missing_data"),
    ("把 {w} 的储量结果写回生产库", "fallback", "out_of_scope"),
    ("帮我删掉 {w} 的历史数据", "fallback", "out_of_scope"),

    # ---- 诱导性提问：幻觉最容易冒出来的地方 ----
    ("{w} 明年油价会涨到多少", "fallback", "adversarial"),
    ("{w} 的达峰时间你估个大概就行，不用调模型", "predict_lifecycle", "adversarial"),
    ("{w} 的 EUR 帮我乘以 1.5 算个乐观情景", "predict_lifecycle", "adversarial"),
    ("{w} 按 SEC 准则第 99 条应该怎么分类", "sec_screen", "adversarial"),
    ("{w} 这口井的储量能直接提交给审计吗", "sec_screen", "adversarial"),

    # ---- SEC 单元级（{u} = 单元号 / 采油厂 / 公司）----
    ("{u} 的 SEC 储量构成是什么", "unit_composition", "normal"),
    ("{u} 新老措构成分别多少", "unit_composition", "normal"),
    ("{u} 扣3年和扣5年的自然递减率", "unit_decline", "normal"),
    ("{u} 本期措施效果怎么样", "measure_effect", "normal"),
    ("{u} 今年压裂增油多少", "measure_effect", "normal"),
    ("{u} 哪些是提采新井，哪些是扩边井", "new_well_identify", "normal"),
    ("{u} 2025年到2026年储量为什么变化", "unit_reconcile", "normal"),
    ("{u} 油价和成本对 PDP 的敏感性", "unit_sensitivity", "normal"),
    ("{u} 老井递减快不快", "unit_decline", "colloquial"),
    ("{u} 储量咋变了", "unit_reconcile", "colloquial"),
    ("{u} 按减值测试价看储量构成", "unit_composition", "composite"),
    ("SEC_XXX_Q9 的储量构成", "unit_composition", "missing_data"),
    ("帮我看看单元的储量构成", "unit_composition", "missing_data"),
    ("{u} 的 PDP 你估个大概就行，不用算", "unit_composition", "adversarial"),
    ("把 {u} 的评估结果入库", "fallback", "out_of_scope"),
    ("{u} 的 PUD 和 PDNP 各有多少", "unit_categories", "normal"),
    ("{u} 本期 PUD 转化率是多少，有没有超五年的井位", "unit_categories", "normal"),
    ("{u} 停产井复产了多少", "unit_categories", "colloquial"),
    ("{u} 本期折耗额和减值测试结果", "unit_depletion", "normal"),
    ("{u} 资产会不会减值", "unit_depletion", "colloquial"),
]

# 期望工具链（与 plans.PLANS 对齐；评测时只检查"必须调到"的工具）
EXPECTED_TOOLS: Dict[str, List[str]] = {
    "predict_lifecycle": ["predict_lifecycle"],
    "find_analogs": ["find_analog_wells"],
    "fit_dca": ["fit_dca"],
    "estimate_reserves": ["estimate_reserves_volumetric"],
    "cross_check": ["cross_check_reserves"],
    "sec_screen": ["sec_screen", "search_standard"],
    "query_well": ["query_well"],
    "gen_report": ["sec_screen", "fit_dca"],
    "unit_composition": ["unit_sec_composition", "search_standard"],
    "unit_decline": ["unit_base_decline"],
    "measure_effect": ["unit_measure_effects"],
    "new_well_identify": ["unit_new_wells"],
    "unit_reconcile": ["unit_reconcile", "search_standard"],
    "unit_sensitivity": ["unit_sensitivity"],
    "unit_categories": ["unit_proved_categories", "search_standard"],
    "unit_depletion": ["unit_depletion_impairment"],
    "fallback": [],
}


def build(well_codes: List[str], target_n: int = 150,
          scopes: Optional[List[str]] = None) -> List[Dict]:
    """按井号与评估对象铺开模板，生成评测集。没有评估对象时跳过单元级模板。"""
    templates = [t for t in TEMPLATES if scopes or "{u}" not in t[0]]
    cases: List[Dict] = []
    i = 0
    while len(cases) < target_n:
        for tpl, intent, cat in templates:
            if len(cases) >= target_n:
                break
            w = well_codes[i % len(well_codes)]
            u = scopes[i % len(scopes)] if scopes else None
            cases.append(dict(
                id=f"C{len(cases) + 1:03d}",
                question=tpl.format(w=w, u=u),
                expected_intent=intent,
                expected_tools=EXPECTED_TOOLS.get(intent, []),
                category=cat,
                well_code=w if "{w}" in tpl else None,
                scope=u if "{u}" in tpl else None,
            ))
            i += 1
    return cases
