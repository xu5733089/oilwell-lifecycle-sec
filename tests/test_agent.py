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
        self.assertTrue(any("4-10(a)(22)" in h["citation"] for h in hits))


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


if __name__ == "__main__":
    unittest.main()
