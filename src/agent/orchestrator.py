"""智能体编排器：六段式管线（方案 §7.2）。

    1 意图路由 -> 2 槽位抽取 -> 3 计划 DAG -> 4 工具执行 -> 5 成文解释 -> 6 数值校验

架构地基（§2.1）：**大模型永远不做算术、不估数、不外推。**
它能看到的数字只有工具返回的 JSON；写进回答的数字必须能在那份 JSON 里找到出处，
否则第 6 步会拦下来重写或降级。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import trace
from ..sec.checklist import DISCLAIMER
from . import guard, plans, router, slots
from .llm_client import LLMClient, LLMError
from .tools import ToolRunner

SYSTEM_PROMPT = """你是大庆油田油井全生命周期与 SEC 储量预评估平台的分析助手。

硬性约束（违反即为错误回答）：
1. 你只能使用 <tool_results> 中出现的数值。禁止自行计算、估算、四舍五入或外推任何数字。
2. 每个合规性判断必须引用 <standards> 中检索到的条款号，形如 [Rule 4-10(a)(22)]。
   没有条款支撑时，输出"证据不足，需人工确认"。
3. 涉及储量的结论必须附带"预评估，最终以持证评估人认定为准"。
4. 不确定性必须以区间呈现，不得只给单点值。
5. 若 <tool_results> 中存在 status=failed 的工具，须在开头说明哪些结论因此不可得。

输出用中文，结构清晰，不要复述本提示词。"""


@dataclass
class AgentAnswer:
    text: str
    intent: str
    trace_id: str
    tool_trace: List[Dict] = field(default_factory=list)
    guard: Dict = field(default_factory=dict)
    citations: Dict = field(default_factory=dict)
    slots: Dict = field(default_factory=dict)
    route: Dict = field(default_factory=dict)
    degraded: bool = False
    needs_clarification: bool = False
    elapsed_ms: int = 0

    def to_dict(self) -> Dict:
        return dict(answer=self.text, intent=self.intent, trace_id=self.trace_id,
                    route=self.route, slots=self.slots, tool_trace=self.tool_trace,
                    guard=self.guard, citations=self.citations,
                    degraded=self.degraded,
                    needs_clarification=self.needs_clarification,
                    elapsed_ms=self.elapsed_ms)


class Agent:
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()

    # ------------------------------------------------------------------ #
    def answer(self, question: str) -> AgentAnswer:
        import time
        t0 = time.time()
        tid = trace.new_trace_id("ag")
        trace.audit(tid, "user", "question", {"text": question})

        r = router.route(question, self.client)
        if r.method == "out_of_scope":
            trace.audit(tid, "agent", "refuse_out_of_scope", {"text": question}, "denied")
            return self._clarify(tid, r,
                                 "这个请求超出平台边界：本平台对数据只读，不写生产库、"
                                 "不修改或删除任何原始数据。如需变更数据，请走对应业务系统的流程。"
                                 "我可以做的是：达峰预测、递减分析、储量互校、SEC 预评估、出报告。",
                                 t0)
        if r.intent == "fallback":
            return self._clarify(tid, r, "没听懂这个问题。可以问：某口井的达峰预测、"
                                          "递减分析、储量互校、SEC 预评估，或让我出份报告。", t0)

        s = slots.extract(question, r.intent, self.client)
        if not s.complete:
            return self._clarify(tid, r, s.question(), t0, slots_=s)

        runner = ToolRunner(tid)
        for step in plans.plan_for(r.intent):
            call = runner.run(step.tool, **step.build_args(s.values))
            if call.status != "ok" and not step.optional:
                break

        results = runner.results()
        evidence = _evidence(results, runner)
        text = self._compose(question, r.intent, results, runner)
        g = guard.check(text, evidence)
        if not g.ok:
            # 一次重写机会；仍不通过就降级为模板成文（模板的数字直接取自 JSON）
            text = self._compose(question, r.intent, results, runner, strict=True)
            g = guard.check(text, evidence)
        degraded = False
        if not g.ok:
            text = _template(r.intent, results, runner)
            g = guard.check(text, evidence)
            degraded = True

        valid = _valid_citations(results)
        cit = guard.check_citations(text, valid)

        ans = AgentAnswer(text=text, intent=r.intent, trace_id=tid,
                          tool_trace=runner.trace(), guard=g.to_dict(), citations=cit,
                          slots=s.to_dict(), route=r.to_dict(), degraded=degraded,
                          elapsed_ms=int((time.time() - t0) * 1000))
        trace.audit(tid, "agent", "answer",
                    {"intent": r.intent, "guard_ok": g.ok, "degraded": degraded})
        return ans

    # ------------------------------------------------------------------ #
    def _clarify(self, tid, r, msg, t0, slots_=None) -> AgentAnswer:
        import time
        return AgentAnswer(text=msg, intent=r.intent, trace_id=tid, route=r.to_dict(),
                           slots=slots_.to_dict() if slots_ else {},
                           needs_clarification=True,
                           guard=dict(ok=True, numbers_checked=0, violations=[], pass_rate=1.0),
                           elapsed_ms=int((time.time() - t0) * 1000))

    def _compose(self, question: str, intent: str, results: Dict,
                 runner: ToolRunner, strict: bool = False) -> str:
        if self.client.backend == "mock":
            return _template(intent, results, runner)
        import json
        std = results.get("search_standard", {}).get("results", [])
        payload = {k: v for k, v in results.items() if k != "search_standard"}
        extra = ("\n【上一次回答出现了工具结果中不存在的数字，请严格只引用 tool_results 里的数值】"
                 if strict else "")
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT + extra},
            {"role": "user", "content":
                f"用户问题：{question}\n\n"
                f"<tool_results>\n{json.dumps(payload, ensure_ascii=False, indent=1)}\n</tool_results>\n\n"
                f"<standards>\n{json.dumps(std, ensure_ascii=False, indent=1)}\n</standards>\n\n"
                f"<tool_failures>\n{json.dumps([c.to_dict() for c in runner.failures()], ensure_ascii=False)}\n</tool_failures>"},
        ]
        try:
            return self.client.chat(msgs).text
        except LLMError as exc:
            return _template(intent, results, runner) + f"\n\n（注：大模型成文不可用，已降级为模板输出。原因：{exc}）"


# ---------------------------------------------------------------------- #
def _valid_citations(results: Dict) -> List[str]:
    """条款存在性对**全知识库**校验。

    只拿本次检索回来的 top-k 当白名单是错的：检查清单里的条款是规则引擎写死的，
    未必出现在这次检索结果里，但它确实存在于准则中。
    存在性与相关性是两回事，这里管存在性。
    """
    cites = [c.get("citation") for c in
             results.get("search_standard", {}).get("results", []) if c.get("citation")]
    try:
        from .rag.retriever import Retriever
        global _CORPUS_CITATIONS
        if _CORPUS_CITATIONS is None:
            _CORPUS_CITATIONS = Retriever().citations()
        cites += _CORPUS_CITATIONS
    except Exception:
        pass
    return list(dict.fromkeys(cites))


_CORPUS_CITATIONS = None


def _evidence(results: Dict, runner: ToolRunner) -> Dict:
    """数值回查的依据：工具成功返回的 JSON + 工具失败时给出的说明。

    失败说明同样出自内核或工具边界（"峰后历史仅 44 天，不足 180 天"），
    不是模型写的；把它排除在外，会让如实转述拒绝原因被误判为幻觉。
    """
    return dict(results=results, failures=[c.error for c in runner.failures() if c.error])


def _fmt(v, unit: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        s = f"{v:,.2f}".rstrip("0").rstrip(".")
    else:
        s = str(v)
    return f"{s} {unit}".strip()


def _template(intent: str, results: Dict, runner: ToolRunner) -> str:
    """模板成文：每个数字直接取自工具 JSON，天然通过数值一致性校验。

    这是 mock 后端的正常输出，也是大模型失灵时的降级路径 ——
    信息量不打折，只是没有自然语言的润色。
    """
    lines: List[str] = []
    fails = runner.failures()
    if fails:
        lines.append("以下工具未能完成，相关结论不可得：" +
                     "；".join(f"{c.name}（{c.error}）" for c in fails))
        lines.append("")

    p = results.get("predict_lifecycle")
    if p:
        lines.append(f"【全生命周期预测】{p['well_code']}（观测窗 {p['obs_days']} 天，"
                     f"模型 {p['model_version']}）")
        name = dict(t_oil_break="见油时间", t_peak="达峰时间", q_peak="达峰产量",
                    p_peak="达峰压力", eur="EUR")
        for k, v in p["results"].items():
            lines.append(f"  · {name.get(k, k)}：P50 {_fmt(v['p50'], v['unit'])}，"
                         f"区间 {_fmt(v['p10'])} ~ {_fmt(v['p90'])} {v['unit']}")
        top = p["explain"]["top_features"][:3]
        if top:
            lines.append("  · 主要影响特征（" + p["explain"]["method"] + "）：" +
                         "，".join(f"{t['feature']}({t['contrib']:+g})" for t in top))
        lines.append("")

    a = results.get("find_analog_wells")
    if a:
        lines.append(f"【类比井】{a['method']}")
        for h in a["analogs"]:
            lines.append(f"  · {h['well_code']}：相似度 {h['similarity']}，"
                         f"实际达峰 {_fmt(h['t_peak_actual'], 'd')}，"
                         f"峰值 {_fmt(h['q_peak_actual'], 't/d')}")
        lines.append("")

    d = results.get("fit_dca")
    if d:
        f = d["fit"]
        lines.append(f"【递减分析】{d['well_code']}：{f['model']} 模型，"
                     f"b={_fmt(f['b'])}，Di={_fmt(f['di_per_month'])}/月，R²={_fmt(f['r2'])}")
        lines.append(f"  · EUR：P50 {_fmt(d['eur']['p50'], 't')}，"
                     f"区间 {_fmt(d['eur']['p10'])} ~ {_fmt(d['eur']['p90'])} t")
        lines.append(f"  · 已累产 {_fmt(d['cum_to_date_t'], 't')}，"
                     f"经济极限 {_fmt(d['economics']['q_econ_t_per_d'], 't/d')}，"
                     f"预计第 {_fmt(d['economics']['t_econ_month'])} 个月到达")
        lines.append("")

    v = results.get("estimate_reserves_volumetric")
    if v:
        o = v["ooip_t"]
        lines.append(f"【容积法储量】{v['well_code']}：OOIP P50 {_fmt(o['p50'], 't')}，"
                     f"区间 {_fmt(o['p10'])} ~ {_fmt(o['p90'])} t（{v['method']}）")
        lines.append("")

    c = results.get("cross_check_reserves")
    if c:
        lines.append(f"【动静态互校】{c['well_code']}：判定 {c['consistency']}，"
                     f"隐含采收率 {_fmt(c['rf_implied_pct'])}%")
        for fnd in c["findings"]:
            # 严重度用全角括号：方括号是条款引用的专用标记，混用会让引用校验器误判
            lines.append(f"  · 〔{fnd['severity']}〕{fnd['finding']}")
            if fnd.get("action"):
                lines.append(f"    建议：{fnd['action']}")
        lines.append("")

    s = results.get("sec_screen")
    if s:
        lines.append(f"【SEC 储量预评估】{s['well_code']}（基准日 {s['as_of']}）")
        lines.append(f"  · 类别：{s['category']} —— {s['category_rationale']}")
        lines.append(f"  · 已证实储量：{_fmt(s['proved_reserves_t'], 't')}，口径 {s['basis']}")
        sm = s["summary"]
        lines.append(f"  · 满足性检查：{sm['n_pass']}/{sm['n_items']} 项通过，"
                     f"{sm['n_needs_human']} 项需人工确认")
        for it in s["checklist"]:
            tag = {"pass": "通过", "fail": "不通过", "needs_human": "需人工确认"}[it["status"]]
            cite = f" [{it['citation']}]" if it.get("citation") else ""
            lines.append(f"    - {tag}｜{it['item']}｜依据：{it['evidence']}{cite}")
        if sm["needs_human"]:
            lines.append("  · 待人工确认项：" + "；".join(sm["needs_human"]))
        lines.append("")
        lines.append(DISCLAIMER)

    lines += _unit_template(results)

    q = results.get("query_well")
    if q and not lines:
        lines.append(f"【单井概况】{q['well_code']}｜{q['block']} {q['layer']}｜{q['well_type']}")
        lines.append(f"  · 投产 {q['first_prod_date']}，生产 {q['prod_days']} 天，"
                     f"累产油 {_fmt(q['cum_oil_t'], 't')}")
        lines.append(f"  · 近 {q['rate_window_days']} 天日产 "
                     f"{_fmt(q['current_rate_t_per_d'], 't/d')}，"
                     f"井口压力 {_fmt(q['latest_whp_mpa'], 'MPa')}，"
                     f"含水 {_fmt(q['water_cut_pct'], '%')}")

    if not lines:
        lines.append("工具未返回可用结果，无法作答。")
    return "\n".join(lines).strip()


LEVEL_CN = {"unit": "SEC 单元", "plant": "采油厂", "company": "公司"}


def _scope_title(r: Dict) -> str:
    s = r["scope"]
    return f"{s['name']}（{LEVEL_CN.get(s['level'], s['level'])}，{s['n_units']} 个单元）"


def _citations_line(results: Dict) -> List[str]:
    std = results.get("search_standard", {}).get("results", [])
    cites = [c["citation"] for c in std if c.get("citation")]
    return ["  · 准则依据：" + "；".join(f"[{c}]" for c in dict.fromkeys(cites))] if cites else []


def _unit_template(results: Dict) -> List[str]:
    """SEC 单元级模板成文。与单井模板同一规矩：每个数字都直接取自工具字段。"""
    L: List[str] = []

    c = results.get("unit_sec_composition")
    if c:
        L.append(f"【SEC 储量构成】{_scope_title(c)}｜基准日 {c['as_of']}｜{c['scenario_label']}")
        L.append(f"  · 已证实已开发储量（{c['category']}）合计 {_fmt(c['total_t'], 't')}；"
                 f"评估期 {c['period_start_ym']} ~ {c['period_end_ym']}，本期产量 {_fmt(c['production_in_period_t'], 't')}")
        for x in c["components"]:
            L.append(f"  · {x['name']}：{_fmt(x['reserves_t'], 't')}，占 {_fmt(x['share_pct'])}%"
                     f"（{x['n_items']} 项；{x['basis']}）")
        o, m, n = c["old_wells"], c["measures"], c["new_wells"]
        L.append(f"  · 老井逐井评估 {o['n_evaluated']} 口（其中峰后历史不足改用类比法 {o['n_short_history']} 口），"
                 f"近期未生产 {o['n_not_producing']} 口不计入；老井最佳估计 {_fmt(o['best_estimate_t'], 't')}")
        L.append(f"  · 本期措施 {m['n_in_period']} 次，可评 {m['n_evaluated']} 次、有效 {m['n_effective']} 次；"
                 f"本期新井：提采新井 {n['n_infill']} 口、扩边井 {n['n_extension']} 口")
        L.append(f"  · 经济参数：油价 {_fmt(c['economics']['price_usd_bbl'])} USD/bbl，"
                 f"单井经济极限 {_fmt(c['economics']['q_econ_t_per_d'], 't/d')}（{c['economics']['note']}）")
        if c.get("aggregation_note"):
            L.append(f"  · {c['aggregation_note']}")
        L += _citations_line(results)
        L += ["", DISCLAIMER, ""]

    d = results.get("unit_base_decline")
    if d:
        L.append(f"【老井基础递减】{_scope_title(d)}｜基准日 {d['as_of']}")
        for r in d["results"]:
            L.append(f"  · 扣近 {r['exclude_years']} 年新井与措施（{r['window_start_ym']} ~ {r['window_end_ym']}，"
                     f"{r['n_months']} 个月）：自然递减率 月 {_fmt(r['natural_monthly_pct'])}% / "
                     f"年 {_fmt(r['natural_annual_pct'])}%；综合递减率 月 {_fmt(r['comprehensive_monthly_pct'])}% / "
                     f"年 {_fmt(r['comprehensive_annual_pct'])}%（拟合 R² {_fmt(r['natural_fit_r2'])}）")
        L.append(f"  · 口径：{d['definition']}")
        L.append("")

    me = results.get("unit_measure_effects")
    if me:
        ov = me["overall"]
        L.append(f"【本期措施效果】{_scope_title(me)}｜基准日 {me['as_of']}")
        L.append(f"  · 共 {ov['n']} 次，可评 {ov['n_evaluated']} 次，有效 {ov['n_effective']} 次"
                 f"（有效率 {_fmt(ov['effective_rate_pct'])}%）；措施前单井日产 {_fmt(ov['pre_rate_avg_t_per_d'], 't/d')}，"
                 f"措施后 {_fmt(ov['post_rate_avg_t_per_d'], 't/d')}，日产增幅 {_fmt(ov['rate_gain_avg_t_per_d'], 't/d')}")
        L.append(f"  · 增加可采储量合计 {_fmt(ov['inc_eur_total_t'], 't')}（已实现增油 {_fmt(ov['realized_total_t'], 't')}，"
                 f"增加的剩余可采 {_fmt(ov['inc_remaining_total_t'], 't')}），单次平均 {_fmt(ov['inc_eur_avg_t'], 't')}")
        for x in me["by_type"]:
            L.append(f"    - {x['name']}：{x['n']} 次，有效 {x['n_effective']} 次，"
                     f"单次增加可采 {_fmt(x['inc_eur_avg_t'], 't')}，合计 {_fmt(x['inc_eur_total_t'], 't')}")
        top = sorted((x for x in me["measures"] if x["inc_eur_t"] is not None),
                     key=lambda x: -x["inc_eur_t"])[:3]
        if top:
            L.append("  · 增加可采最多的措施：" + "；".join(
                f"{x['well_code']}（{x['event_name']}，{x['event_ym']}）{_fmt(x['inc_eur_t'], 't')}" for x in top))
        L.append(f"  · 口径：{me['definition']}")
        L.append("")

    nw = results.get("unit_new_wells")
    if nw:
        L.append(f"【本期新井识别】{_scope_title(nw)}｜基准日 {nw['as_of']}")
        L.append(f"  · 规则：{nw['rule']['text']}")
        L.append(f"  · 提采新井 {nw['n_infill']} 口，扩边井 {nw['n_extension']} 口")
        for code, title in (("infill", "提采新井"), ("extension", "扩边井")):
            ws = [w for w in nw["wells"] if w["category_code"] == code][:3]
            if ws:
                L.append(f"  · {title}（储量前三）：" + "；".join(
                    f"{w['well_code']}（{w['unit_id']}，投产 {w['first_prod_ym']}，周边老井 {w['n_old_neighbors']} 口，"
                    f"最近老井 {_fmt(w['nearest_old_m'], 'm')}）{_fmt(w['reserves_t'], 't')}，{w['basis']}" for w in ws))
        L.append("")

    rc = results.get("unit_reconcile")
    if rc:
        L.append(f"【储量对账】{_scope_title(rc)}｜{rc['from_as_of']} → {rc['to_as_of']}｜"
                 f"情景 {rc['scenario']}｜期初来源：{rc['opening_source']}")
        for r in rc["table"]:
            L.append(f"  · {r['item']}：{_fmt(r['value'], 't')}")
        L.append(f"  · 闭合检查：{'闭合' if rc['balanced'] else '不闭合'}（差额 {_fmt(rc['difference'], 't')}）；"
                 f"产量法折耗率 {_fmt(rc['depletion_rate_pct'])}%")
        L.append(f"  · 说明：{rc['note']}")

    at = results.get("unit_change_attribution")
    if at:
        L.append(f"【变化归因】PDP 由 {_fmt(at['opening_t'], 't')} 变为 {_fmt(at['closing_t'], 't')}，"
                 f"变化 {_fmt(at['change_t'], 't')}")
        for dr in at["drivers"]:
            ev = "；".join(_evidence_text(dr["evidence_kind"], e) for e in dr["evidence"])
            L.append(f"  · {dr['item']} {_fmt(dr['value_t'], 't')}（占 {_fmt(dr['share_pct'])}%）"
                     + (f"：{ev}" if ev else ""))
    if rc or at:
        L += _citations_line(results)
        L += ["", DISCLAIMER, ""]

    pc = results.get("unit_proved_categories")
    if pc:
        from ..api.services import PDNP_STATUS_CN, PUD_STATUS_CN
        L.append(f"【证实储量类别】{_scope_title(pc)}｜基准日 {pc['as_of']}｜{pc['scenario_label']}")
        L.append(f"  · 证实储量合计 {_fmt(pc['total_proved_t'], 't')}")
        for c in pc["categories"]:
            L.append(f"  · {c['name']}：{_fmt(c['reserves_t'], 't')}，占 {_fmt(c['share_pct'])}%（{c['n_items']} 项；{c['basis']}）")
        p = pc["pud"]
        L.append(f"  · PUD 规则：{p['rule']}")
        L.append("  · 部署井位判定：" + "，".join(f"{PUD_STATUS_CN.get(k, k)} {v} 个" for k, v in p["n_by_status"].items()))
        d = pc["pdnp"]
        L.append(f"  · PDNP 规则：{d['rule']}")
        L.append("  · 停产井判定：" + "，".join(f"{PDNP_STATUS_CN.get(k, k)} {v} 口" for k, v in d["n_by_status"].items()))
        for w in pc["warnings"][:3]:
            L.append(f"  · 预警：{w['text']}")
    tr = results.get("unit_category_tracking")
    if tr and tr.get("pud"):
        u = tr["pud"]
        L.append(f"【PUD 滚动】{tr['from_as_of']} → {tr['to_as_of']}：期初入账 {u['n_opening']} 个井位，钻井转化 {u['n_converted']} 个"
                 f"（转化率 {_fmt(u['conversion_rate_pct'])}%），五年规则移出 {u['n_expired']} 个，其他移出 {u['n_removed']} 个，"
                 f"新入账 {u['n_new']} 个，期末 {u['n_closing']} 个")
        for r in u["table"]:
            L.append(f"    - {r['item']}：{_fmt(r['value'], 't')}")
    if tr and tr.get("pdnp"):
        d = tr["pdnp"]
        L.append(f"【PDNP 滚动】期初 {d['n_opening']} 口，复产转 PDP {d['n_reactivated']} 口，移出 {d['n_removed']} 口，"
                 f"新增停产 {d['n_new']} 口，期末 {d['n_closing']} 口")
        for r in d["table"]:
            L.append(f"    - {r['item']}：{_fmt(r['value'], 't')}")
    if pc or tr:
        L.append(f"  · 说明：{(tr or pc)['note']}")
        L += _citations_line(results)
        L += ["", DISCLAIMER, ""]

    dp = results.get("unit_depletion_impairment")
    if dp:
        sm, asm = dp["summary"], dp["assumptions"]
        L.append(f"【折耗与减值】{_scope_title(dp)}｜基准日 {dp['as_of']}｜金额单位 {dp['currency']}")
        L.append(f"  · 期初资产净值 {_fmt(sm['opening_nbv_wan'])}，本期资本化投入 {_fmt(sm['capex_additions_wan'])}；"
                 f"产量法折耗率 {_fmt(sm['depletion_rate_pct'])}%，折耗额 {_fmt(sm['depletion_wan'])}")
        L.append(f"  · 折耗后账面价值 {_fmt(sm['carrying_wan'])}；可收回金额 {_fmt(sm['recoverable_wan'])}"
                 f"（{asm['impairment_scenario']}，折现率 {_fmt(asm['discount_rate'])}）；"
                 f"减值额 {_fmt(sm['impairment_wan'])}，{sm['n_impaired']} 个单元减值；期末净值 {_fmt(sm['closing_nbv_wan'])}")
        imp = [x for x in dp["units"] if x["impaired"]]
        if imp:
            L.append("  · 减值单元：" + "；".join(
                f"{x['unit_id']} 减值 {_fmt(x['impairment_wan'])}（账面 {_fmt(x['carrying_wan'])}，可收回 {_fmt(x['recoverable_wan'])}）"
                for x in imp))
        L.append("  · 口径：" + "；".join(dp["method"]))
        L.append(f"  · 说明：{dp['note']}")
        L.append("")

    s = results.get("unit_sensitivity")
    if s:
        L.append(f"【PDP 敏感性】{_scope_title(s)}｜基准日 {s['as_of']}｜{s['scenario_label']}，"
                 f"基准油价 {_fmt(s['base_price_usd_bbl'])} USD/bbl，最佳估计 {_fmt(s['base_best_estimate_t'], 't')}")
        L.append(f"  · 敏感权重（{s['perturbation']}）：" + "；".join(
            f"{w['name']} {_fmt(w['weight_pct'])}%（PDP 摆幅 {_fmt(w['swing_t'], 't')}）" for w in s["weights"]))
        for cv in s["curves"]:
            L.append(f"  · {cv['name']}（{cv['unit']}）从 {_fmt(cv['x'][0])} 到 {_fmt(cv['x'][-1])}："
                     f"PDP 变化 {_fmt(cv['delta_t'][0], 't')} ~ {_fmt(cv['delta_t'][-1], 't')}")
        L.append("  · 各单元最敏感参数：" + "；".join(f"{u['unit_id']} {u['top_param']}" for u in s["units"]))
        L.append(f"  · 口径：{s['note']}")
        L.append("")
    return L


def _evidence_text(kind: str, e: Dict) -> str:
    if kind == "wells":
        return f"{e['well_code']} {_fmt(e['reserves_t'], 't')}（{e['basis']}，投产 {e['months_on']} 个月）"
    if kind == "measures":
        return f"{e['well_code']} {e['event_name']}（{e['event_ym']}）增加可采 {_fmt(e['inc_eur_t'], 't')}"
    if kind == "economics":
        return (f"油价 {_fmt(e['price_open_usd_bbl'])} → {_fmt(e['price_close_usd_bbl'])} USD/bbl，"
                f"单井经济极限 {_fmt(e['q_econ_open_t_per_d'])} → {_fmt(e['q_econ_close_t_per_d'])} t/d")
    if kind == "category":
        return f"{e['well_code']} {e['change']} {_fmt(e['reserves_t'], 't')}"
    if kind == "wells_rate_change":
        return (f"{e['well_code']} 日产 {_fmt(e['rate_open_t_per_d'])} → {_fmt(e['rate_close_t_per_d'])} t/d，"
                f"含水 {_fmt(e['water_cut_open_pct'])}% → {_fmt(e['water_cut_close_pct'])}%")
    return ""
