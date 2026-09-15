"""智能体层测试：路由、槽位、任务边界、数值校验、端到端。

这些测试跑在 mock 后端上，不需要任何大模型 —— 这正是 mock 后端存在的理由。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import guard, plans, router, slots      # noqa: E402
from src.agent.orchestrator import Agent               # noqa: E402
from src.agent.rag.retriever import Retriever          # noqa: E402
from src.agent.tools import REGISTRY, ToolRunner       # noqa: E402
from src.api import services as S                      # noqa: E402


def _a_well() -> str:
    m = S._tables()["master"]
    return m[m["status"] == "producing"].iloc[0]["well_code_anon"]


class TestRouter(unittest.TestCase):
    CASES = [
        ("GL-A-0001 什么时候达峰", "predict_lifecycle"),
        ("给 GL-A-0001 找几口类比井", "find_analogs"),
        ("对 GL-A-0001 做递减曲线分析", "fit_dca"),
        ("算一下 GL-A-0001 的容积法地质储量", "estimate_reserves"),
        ("GL-A-0001 动静态储量互校", "cross_check"),
        ("GL-A-0001 的 SEC 储量预评估", "sec_screen"),
    ]

    def test_keyword_routing(self):
        for q, expect in self.CASES:
            self.assertEqual(router.keyword_route(q).intent, expect, q)

    def test_unknown_falls_back(self):
        self.assertEqual(router.keyword_route("今天天气怎么样").intent, "fallback")


class TestSlots(unittest.TestCase):
    def test_extract_well_and_obs_days(self):
        s = slots.extract("GL-A-0123 用前 60 天数据预测达峰", "predict_lifecycle")
        self.assertEqual(s.values["well_code"], "GL-A-0123")
        self.assertEqual(s.values["obs_days"], 60)
        self.assertTrue(s.complete)

    def test_missing_slot_asks_instead_of_guessing(self):
        s = slots.extract("帮我预测一下达峰时间", "predict_lifecycle")
        self.assertFalse(s.complete)
        self.assertIn("well_code", s.missing)
        self.assertIn("井号", s.question())


class TestToolBoundary(unittest.TestCase):
    """任务边界要落成代码，不能只写在文档里。"""

    def test_unknown_tool_denied(self):
        r = ToolRunner("t1")
        call = r.run("drop_table", table="prod_daily")
        self.assertEqual(call.status, "denied")
        self.assertIn("白名单", call.error)

    def test_call_budget_enforced(self):
        r = ToolRunner("t2", max_calls=2)
        w = _a_well()
        for _ in range(2):
            r.run("query_well", well_code=w)
        call = r.run("query_well", well_code=w)
        self.assertEqual(call.status, "denied")
        self.assertIn("上限", call.error)

    def test_kernel_error_reported_not_invented(self):
        r = ToolRunner("t3")
        call = r.run("predict_lifecycle", well_code="NOPE-9999")
        self.assertEqual(call.status, "failed")
        self.assertIsNone(call.result)
        self.assertIn("不存在", call.error)

    def test_all_tools_are_read_only(self):
        for name in REGISTRY:
            self.assertFalse(any(k in name for k in ("delete", "write", "update", "drop")))


class TestGuard(unittest.TestCase):
    def test_accepts_numbers_from_tool_results(self):
        g = guard.check("达峰 P50 为 128 天，区间 96 ~ 171 天。",
                        {"r": {"p50": 128.334, "p10": 96.1, "p90": 170.8}})
        self.assertTrue(g.ok)

    def test_catches_invented_number(self):
        g = guard.check("达峰时间大约 250 天。", {"r": {"p50": 128.3}})
        self.assertFalse(g.ok)
        self.assertEqual(g.violations[0]["value"], 250.0)

    def test_ignores_citations_and_dates(self):
        g = guard.check("依据 [Rule 4-10(a)(22)]，基准日 2026-12-31。", {"r": {}})
        self.assertTrue(g.ok)

    def test_citation_validation(self):
        c = guard.check_citations("见 [Rule 4-10(a)(22)] 与 [Rule 9-99]", ["Rule 4-10(a)(22)"])
        self.assertFalse(c["ok"])
        self.assertEqual(c["invalid"], ["Rule 9-99"])


class TestRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = Retriever()

    def test_chunked_by_clause(self):
        self.assertGreaterEqual(len(self.r.chunks), 8)
        self.assertTrue(all(c.citation for c in self.r.chunks))

    def test_retrieves_five_year_rule(self):
        hits = self.r.search("PUD 未钻井位 五年 开发计划", top_k=3)
        self.assertTrue(any("31" in h["citation"] for h in hits), [h["citation"] for h in hits])

    def test_retrieves_reasonable_certainty(self):
        hits = self.r.search("合理确定性 概率法 90%", top_k=3)
        self.assertIn("Rule 4-10(a)(24)", [h["citation"] for h in hits])

    def test_official_text_and_precise_citations(self):
        """语料必须是官方原文，且条款号精确到段落层级。"""
        c = self.r.get("Rule 4-10(a)(31)(ii)")
        self.assertIn("scheduled to be drilled within five years", c["text"])
        self.assertEqual(c["cfr"], "17 CFR 210.4-10(a)(31)(ii)")
        self.assertEqual(c["ancestors"][-1]["citation"], "Rule 4-10(a)(31)")
        price = self.r.get("Rule 4-10(a)(22)(v)")["text"]
        self.assertIn("first-day-of-the-month", price)
        self.assertIn("Rule 4-10(a)(17)", self.r.citations())            # 解析器不能把 (17) 当成下级分项
        for must in ("Item 1202(b)(1)", "Item 1203(b)", "Item 1203(d)", "Rule 4-10(a)(22)(i)(B)"):
            self.assertIn(must, self.r.citations())

    def test_chinese_query_hits_price_rule(self):
        hits = self.r.search("价格口径 12个月 首日价格 平均", top_k=3)
        self.assertIn("Rule 4-10(a)(22)(v)", [h["citation"] for h in hits])

    def test_explicit_citation_in_query(self):
        hits = self.r.search("Item 1203(d) 说的是什么", top_k=3)
        self.assertEqual(hits[0]["citation"], "Item 1203(d)")


class TestPlans(unittest.TestCase):
    def test_sec_plan_always_retrieves_standards(self):
        """合规结论不许无引用，所以 SEC 计划必须带条款检索。"""
        self.assertIn("search_standard", plans.describe("sec_screen"))

    def test_every_intent_has_plan(self):
        for intent in router.INTENTS:
            self.assertIsNotNone(plans.plan_for(intent))


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = Agent()
        cls.well = _a_well()

    def test_prediction_answer_is_numerically_consistent(self):
        ans = self.agent.answer(f"{self.well} 什么时候达峰")
        self.assertEqual(ans.intent, "predict_lifecycle")
        self.assertTrue(ans.guard["ok"], ans.guard.get("violations"))
        self.assertGreater(ans.guard["numbers_checked"], 0)

    def test_sec_answer_cites_only_real_clauses(self):
        ans = self.agent.answer(f"{self.well} 的 SEC 储量预评估")
        self.assertEqual(ans.intent, "sec_screen")
        self.assertTrue(ans.guard["ok"], ans.guard.get("violations"))
        self.assertEqual(ans.citations["invalid"], [], ans.citations)
        self.assertIn("预评估", ans.text)

    def test_missing_well_asks_instead_of_answering(self):
        ans = self.agent.answer("帮我预测达峰时间")
        self.assertTrue(ans.needs_clarification)
        self.assertIn("井号", ans.text)

    def test_nonexistent_well_reports_failure_honestly(self):
        ans = self.agent.answer("NOPE-9999 什么时候达峰")
        self.assertTrue(any(t["status"] != "ok" for t in ans.tool_trace))
        self.assertIn("未能完成", ans.text)

    def test_out_of_scope_request_refused(self):
        ans = self.agent.answer(f"把 {self.well} 的储量结果写回生产库")
        self.assertTrue(ans.needs_clarification or not ans.tool_trace)

    def test_trace_is_recorded(self):
        ans = self.agent.answer(f"查一下 {self.well} 的基本信息")
        self.assertTrue(ans.trace_id.startswith("ag_"))
        from src import db
        n = db.read_df("SELECT COUNT(*) n FROM audit_log WHERE trace_id = ?",
                       (ans.trace_id,))["n"].iloc[0]
        self.assertGreater(int(n), 0)


class TestUnitAgent(unittest.TestCase):
    """SEC 单元级问答：路由、槽位、计划、端到端数值一致性。"""

    @classmethod
    def setUpClass(cls):
        cls.agent = Agent()
        u = S.list_units()
        cls.unit = u["plants"][0]["units"][0]["unit_id"]
        cls.plant = u["plants"][0]["plant_name"]
        cls.company = u["company"]["name"]

    def test_scope_words_route_to_unit_intents(self):
        cases = [(f"{self.unit} 的储量构成", "unit_composition"),
                 (f"{self.plant} 扣3年自然递减率", "unit_decline"),
                 (f"{self.unit} 本期压裂增油多少", "measure_effect"),
                 (f"{self.company} 哪些是扩边井", "new_well_identify"),
                 (f"{self.plant} 储量为什么变化", "unit_reconcile"),
                 (f"{self.unit} 油价敏感性", "unit_sensitivity")]
        for q, expect in cases:
            self.assertEqual(router.keyword_route(q).intent, expect, q)

    def test_well_questions_do_not_leak_into_unit_intents(self):
        """"递减率"对一口井是递减分析，对一个单元才是老井基础递减。"""
        self.assertEqual(router.keyword_route("GL-A-0357 的递减率是多少").intent, "fit_dca")

    def test_scope_slots(self):
        v = slots.rule_extract(f"{self.unit} 2025年到2026年按减值价对账，扣3年")
        self.assertEqual(v["scope"], self.unit)
        self.assertEqual((v["from_as_of"], v["to_as_of"]), ("2025-12-31", "2026-12-31"))
        self.assertEqual(v["scenario"], "impairment")
        self.assertEqual(v["exclude_years"], "3")
        self.assertEqual(slots.rule_extract(f"{self.plant} 的储量构成")["scope"], self.plant)
        self.assertEqual(slots.rule_extract("全公司的储量构成")["scope"], self.company)

    def test_year_sets_matching_price_deck(self):
        v = slots.rule_extract("GL-A-0357 2025年能进已证实储量吗")
        self.assertEqual((v["as_of"], v["price_deck_id"]), ("2025-12-31", "deck_2025_12"))

    def test_unit_scope_not_mistaken_for_well_code(self):
        self.assertNotIn("well_code", slots.rule_extract(f"{self.unit} 的储量构成"))

    def test_compliance_plans_retrieve_standards(self):
        self.assertIn("search_standard", plans.describe("unit_composition"))
        self.assertIn("search_standard", plans.describe("unit_reconcile"))

    def test_composition_answer_is_consistent_and_cited(self):
        ans = self.agent.answer(f"{self.unit} 的 SEC 储量构成")
        self.assertEqual(ans.intent, "unit_composition")
        self.assertTrue(ans.guard["ok"], ans.guard.get("violations"))
        self.assertGreater(ans.guard["numbers_checked"], 10)
        self.assertEqual(ans.citations["invalid"], [], ans.citations)
        self.assertIn("预评估", ans.text)

    def test_reconcile_and_sensitivity_answers_are_consistent(self):
        for q in (f"{self.plant} 2025年到2026年储量为什么变化", f"{self.unit} 油价和成本敏感性",
                  f"{self.unit} 本期措施效果", f"{self.company} 扣3年和扣5年自然递减率",
                  f"{self.unit} 哪些是提采新井"):
            ans = self.agent.answer(q)
            self.assertTrue(ans.guard["ok"], (q, ans.guard.get("violations")))
            self.assertTrue(all(t["status"] == "ok" for t in ans.tool_trace), (q, ans.tool_trace))

    def test_missing_scope_asks(self):
        ans = self.agent.answer("帮我看看单元的储量构成")
        self.assertTrue(ans.needs_clarification)
        self.assertIn("评估对象", ans.text)

    def test_unknown_unit_reported_honestly(self):
        ans = self.agent.answer("SEC_XXX_Q9 的储量构成")
        self.assertTrue(any(t.get("error_kind") == "kernel" for t in ans.tool_trace))
        self.assertTrue(ans.guard["ok"], ans.guard.get("violations"))


class TestServiceEnvelope(unittest.TestCase):
    REQUIRED = ("model_version", "label_def_version", "data_source", "trace_id")

    def test_every_service_returns_traceability_fields(self):
        w = _a_well()
        for fn, kw in [(S.query_well, {}), (S.predict_lifecycle, {}),
                       (S.find_analog_wells, {"top_k": 2}), (S.fit_dca, {}),
                       (S.estimate_reserves_volumetric, {"mc_samples": 500}),
                       (S.cross_check_reserves, {}), (S.sec_screen, {})]:
            out = fn(w, **kw)
            for k in self.REQUIRED:
                self.assertIn(k, out, f"{fn.__name__} 缺少追溯字段 {k}")

    def test_unit_services_return_traceability_fields(self):
        unit = S.list_units()["plants"][0]["units"][0]["unit_id"]
        for fn in (S.list_units, S.unit_sec_composition, S.unit_production_composition,
                   S.unit_base_decline, S.unit_new_wells, S.unit_measure_effects, S.unit_reconcile,
                   S.unit_sensitivity, S.unit_change_attribution, S.unit_indicators):
            out = fn() if fn is S.list_units else fn(unit)
            for k in self.REQUIRED:
                self.assertIn(k, out, f"{fn.__name__} 缺少追溯字段 {k}")

    def test_well_list_has_one_row_per_well(self):
        """标签表存多版口径：井表不能因此出现重复井，也不能混入非当前口径的标签。"""
        wells = S.list_wells(limit=100000)["wells"]
        codes = [w["well_code"] for w in wells]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertEqual(len(codes), len(S._tables()["master"]))
        self.assertEqual(set(S._tables()["labels"]["label_def_version"]),
                         {S._bundle()["meta"]["label_def_version"]})


if __name__ == "__main__":
    unittest.main()
