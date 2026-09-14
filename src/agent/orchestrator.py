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
        text = self._compose(question, r.intent, results, runner)
        g = guard.check(text, results)
        if not g.ok:
            # 一次重写机会；仍不通过就降级为模板成文（模板的数字直接取自 JSON）
            text = self._compose(question, r.intent, results, runner, strict=True)
            g = guard.check(text, results)
        degraded = False
        if not g.ok:
            text = _template(r.intent, results, runner)
            g = guard.check(text, results)
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
