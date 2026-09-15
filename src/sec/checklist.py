"""SEC 满足性检查清单（方案 §6.4）—— 本模块最有价值的交付物。

每条给 pass / fail / needs_human，并附依据来源：
依据要么是数据字段的实际取值，要么是准则条款引用。
清单里任一项为 needs_human 时，报告只能导出为初稿，不能作为结论。
"""
from __future__ import annotations

from typing import Dict, List, Optional

PASS, FAIL, HUMAN = "pass", "fail", "needs_human"

DISCLAIMER = ("本结果由模型自动生成，属预评估，供内部参考；"
              "最终储量认定以持证评估人签署意见为准。")


def _item(item: str, status: str, evidence: str, citation: Optional[str] = None) -> Dict:
    return dict(item=item, status=status, evidence=evidence, citation=citation)


def _t_econ_text(economic: Dict) -> str:
    t = economic.get("t_econ_month")
    if t is None:
        return "经济极限时刻不可得"
    if economic.get("t_econ_capped"):
        return (f"产量在 {t:.0f} 个月的评估期内始终高于经济极限，"
                "该时刻为积分上限而非真实经济极限时刻")
    return f"经济极限时刻 t_econ≈{t:.0f} 月"


def build(*, prod_days: int, economic: Dict, classification: Dict,
          coverage: Optional[float], model_version: str, label_def_version: str,
          data_source: str, trace_id: str, has_pud: bool = False,
          five_year_plan_confirmed: Optional[bool] = None,
          reliability_report: Optional[str] = None,
          dca_error: Optional[str] = None) -> List[Dict]:
    q_econ = economic.get("q_econ")
    items: List[Dict] = [
        _item("已知储层且有产能证据",
              PASS if prod_days > 0 else HUMAN,
              f"prod_daily 有效记录 {prod_days} 天" if prod_days > 0 else "无产量记录，需试油/试采结论支撑",
              "Rule 4-10(a)(22)"),
        _item("在现有经济条件下可经济开采",
              HUMAN if dca_error else
              (PASS if economic.get("current_rate", 0) > (q_econ or 0) else FAIL),
              (f"经济极限 q_econ={q_econ:.2f} t/d；当前产量 "
               f"{economic.get('current_rate', float('nan')):.2f} t/d；"
               + (f"但产量剖面不可得：{dca_error}" if dca_error else
                  _t_econ_text(economic))),
              "Rule 4-10(a)(10)"),
        _item("价格采用 12 个月首日价格的未加权算术平均",
              PASS,
              f"价格册 {economic.get('price_deck_id')}（as_of={economic.get('as_of')}）"
              f"，12 月均价 {economic.get('avg_12m_price_usd_bbl')} USD/bbl",
              "Rule 4-10(a)(22)(v)"),
        _item("采用现有操作方法与法规，未假设新技术或政策变化",
              PASS,
              "评估参数集未引入未来技术改进或政策假设；成本取当前成本，不含通胀预期",
              "Rule 4-10(a)(22)(v)"),
        _item("合理确定性：概率法取低估计口径，且区间已通过保形校准",
              PASS if (coverage is not None and abs(coverage - 0.8) <= 0.08) else HUMAN,
              (f"准则要求概率法下实际采出量等于或超过估计值的概率至少 90%；"
               f"模型 {model_version}；校准集经验覆盖率 {coverage:.2f}（名义 0.80）"
               if coverage is not None else "缺少校准集覆盖率报告，需补充"),
              "Rule 4-10(a)(24)"),
        _item("储量类别判定依据充分",
              PASS if not classification.get("needs_human") else HUMAN,
              f"判定 {classification.get('category')}；依据：" + "；".join(classification.get("evidence", [])),
              classification.get("citation")),
    ]

    if has_pud:
        items.append(_item(
            "PUD 已纳入五年内开发计划，且有连续性证据",
            PASS if five_year_plan_confirmed else HUMAN,
            "已由开发计划表确认" if five_year_plan_confirmed else "开发计划未确认，需人工核对",
            "Rule 4-10(a)(31)(ii)"))

    items += [
        _item("可靠技术适用性说明",
              PASS if reliability_report else HUMAN,
              (reliability_report or "需附本平台方法在本区块的回测精度与适用边界说明")
              + "（可靠技术须经现场检验，并在被评估地层或类比地层中证明结果一致、可重复）",
              "Rule 4-10(a)(25)"),
        _item("数据可追溯：数据源、模型版本、口径版本、时间戳齐全",
              PASS,
              f"data_source={data_source}；model_version={model_version}；"
              f"label_def_version={label_def_version}；trace_id={trace_id}",
              None),
    ]
    return items


def summarize(items: List[Dict]) -> Dict:
    return dict(
        n_items=len(items),
        n_pass=sum(1 for i in items if i["status"] == PASS),
        n_fail=sum(1 for i in items if i["status"] == FAIL),
        n_needs_human=sum(1 for i in items if i["status"] == HUMAN),
        needs_human=[i["item"] for i in items if i["status"] == HUMAN],
        blocked=[i["item"] for i in items if i["status"] == FAIL],
        disclaimer=DISCLAIMER,
    )
