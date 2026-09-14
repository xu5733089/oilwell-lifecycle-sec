"""内核算法测试。用 stdlib unittest，不依赖 pytest（内网常装不上）。

    python -m unittest discover -s tests -v
    pytest tests            # 装了 pytest 也能跑
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.labeling.labels import extract_all                       # noqa: E402
from src.quality.gate import check_wells                          # noqa: E402
from src.reserves import crosscheck, dca, volumetric              # noqa: E402
from src.sec import checklist, classify, economics, reconcile     # noqa: E402
from src.synth.well_generator import generate                     # noqa: E402


class TestDCA(unittest.TestCase):
    def test_b_gt_1_without_dmin_raises(self):
        """b>1 且无终端递减 -> Arps 积分发散。这是储量评估最常见的错误来源，必须拦。"""
        f = dca.DCAFit("hyperbolic", qi=10.0, di=0.2, b=1.2, d_min=0.0)
        with self.assertRaises(ValueError):
            dca.rate(np.array([1.0, 2.0]), f)
        with self.assertRaises(ValueError):
            dca.eur(f, q_econ=1.0)

    def test_b_gt_1_with_dmin_converges(self):
        f = dca.DCAFit("modified_hyperbolic", qi=10.0, di=0.2, b=1.2, d_min=0.075 / 12)
        e = dca.eur(f, q_econ=1.0)
        self.assertGreater(e["eur"], 0)
        self.assertLess(e["eur"], 1e7)
        self.assertGreater(e["t_econ_month"], 0)

    def test_fit_recovers_parameters(self):
        truth = dca.DCAFit("modified_hyperbolic", qi=20.0, di=0.18, b=0.85, d_min=0.075 / 12)
        t = np.arange(0, 60, 0.5)
        q = dca.rate(t, truth) * np.random.default_rng(0).lognormal(0, 0.03, len(t))
        f = dca.fit(t, q, model="modified_hyperbolic", recency_halflife_months=None)
        self.assertGreater(f.r2, 0.97)
        self.assertAlmostEqual(f.b, truth.b, delta=0.25)
        self.assertAlmostEqual(f.qi, truth.qi, delta=2.0)

    def test_eur_never_below_cumulative(self):
        """EUR = 已累产 + 剩余可采，结构上不可能小于累产。"""
        truth = dca.DCAFit("modified_hyperbolic", qi=15.0, di=0.2, b=0.9, d_min=0.075 / 12)
        t = np.arange(0, 40, 0.5)
        q = dca.rate(t, truth)
        cum = float(np.sum(q) * 0.5 * 30.4)
        out = dca.eur_from_history(t, q, truth, q_econ=1.0, cum_to_date=cum, n_boot=20)
        self.assertGreaterEqual(out["p50"], cum)
        self.assertGreaterEqual(out["p90"], out["p10"])

    def test_quantile_convention(self):
        """p10 必须是数值小者；low_estimate 与 p10 同值。"""
        truth = dca.DCAFit("modified_hyperbolic", qi=15.0, di=0.2, b=0.9, d_min=0.075 / 12)
        t = np.arange(0, 40, 0.5)
        out = dca.eur_from_history(t, dca.rate(t, truth), truth, 1.0, 0.0, n_boot=20)
        self.assertLessEqual(out["p10"], out["p50"])
        self.assertLessEqual(out["p50"], out["p90"])
        self.assertEqual(out["p10"], out["low_estimate"])
        self.assertEqual(out["p90"], out["high_estimate"])


class TestVolumetric(unittest.TestCase):
    def test_ooip_scales_linearly(self):
        a = volumetric.ooip_t(100_000, 10, 8, 60)
        b = volumetric.ooip_t(200_000, 10, 8, 60)
        self.assertAlmostEqual(b / a, 2.0, places=6)

    def test_monte_carlo_ordering(self):
        r = volumetric.monte_carlo(
            volumetric.ParamSpec(400_000, 0.2), volumetric.ParamSpec(15, 0.15),
            volumetric.ParamSpec(8, 0.12), volumetric.ParamSpec(60, 0.08), n=4000)
        o = r["ooip"]
        self.assertLess(o["p10"], o["p50"])
        self.assertLess(o["p50"], o["p90"])


class TestCrossCheck(unittest.TestCase):
    REF = dict(p10=1.5, p50=3.0, p90=8.0, low_pct=1.5, high_pct=8.0)

    def test_in_range(self):
        d = crosscheck.diagnose(dict(p10=2400, p50=3000, p90=3600),
                                dict(p10=80_000, p50=100_000, p90=120_000),
                                self.REF, dict(b=0.8, d_min=0.006))
        self.assertEqual(d["consistency"], "consistent")
        self.assertAlmostEqual(d["rf_implied_pct"], 3.0, places=2)

    def test_too_high_flags_reason(self):
        d = crosscheck.diagnose(dict(p10=9000, p50=12000, p90=15000),
                                dict(p10=80_000, p50=100_000, p90=120_000),
                                self.REF, dict(b=0.8, d_min=0.006))
        self.assertEqual(d["consistency"], "high")
        self.assertTrue(any(f["code"] == "RF_TOO_HIGH" for f in d["findings"]))

    def test_b_gt_1_no_dmin_is_critical(self):
        d = crosscheck.diagnose(dict(p10=2400, p50=3000, p90=3600),
                                dict(p10=80_000, p50=100_000, p90=120_000),
                                self.REF, dict(b=1.3, d_min=0.0))
        self.assertTrue(d["needs_human"])
        self.assertTrue(any(f["code"] == "B_GT_1_NO_DMIN" for f in d["findings"]))


class TestSEC(unittest.TestCase):
    def test_price_is_unweighted_12m_average(self):
        d = economics.deck("deck_2026_12")
        expected = sum(d["monthly_first_day_prices_usd_bbl"]) / 12
        self.assertAlmostEqual(economics.avg_12m_price("deck_2026_12"), expected, places=9)

    def test_higher_price_lowers_economic_limit(self):
        low = economics.economic_limit_rate("deck_2026_12")["q_econ"]
        high = economics.economic_limit_rate("deck_2025_12")["q_econ"]
        self.assertLess(high, low)      # deck_2025_12 油价更高 -> 经济极限更低

    def test_classification_branches(self):
        self.assertEqual(classify.classify({"status": "producing"}, 800, True)["category"], "PDP")
        self.assertEqual(classify.classify({"status": "producing"}, 800, False)["category"],
                         "NOT_PROVED")
        self.assertEqual(classify.classify({"status": "shut_in"}, 800, True)["category"], "PDNP")
        self.assertEqual(
            classify.classify({"status": "planned"}, 0, True, True, True)["category"], "PUD")
        self.assertEqual(
            classify.classify({"status": "planned"}, 0, True, False, True)["category"],
            "NOT_PROVED")

    def test_pud_cites_five_year_rule(self):
        c = classify.classify({"status": "planned"}, 0, True, True, True)
        self.assertIn("4-10(a)(31)", c["citation"])

    def test_checklist_citations_exist_in_corpus(self):
        """检查清单引用的条款号必须在知识库里真实存在，否则引用命中率就是假的。"""
        from src.agent.rag.retriever import Retriever
        valid = set(Retriever().citations())
        econ = dict(economics.economic_limit_rate("deck_2026_12"),
                    current_rate=5.0, t_econ_month=40)
        items = checklist.build(
            prod_days=800, economic=econ,
            classification=classify.classify({"status": "producing"}, 800, True),
            coverage=0.80, model_version="m", label_def_version="v1",
            data_source="SYNTHETIC", trace_id="t", has_pud=True,
            five_year_plan_confirmed=True, reliability_report="ok")
        cited = [i["citation"] for i in items if i["citation"]]
        self.assertTrue(cited)
        for c in cited:
            self.assertIn(c, valid, f"条款号 {c!r} 不在知识库中")

    def test_reconcile_balances(self):
        r = reconcile.build(dict(opening=100.0, production=-10.0, price_revision=-2.0,
                                 technical_revision=1.0, new_wells=5.0, category_change=0.0))
        self.assertTrue(r["balanced"])
        self.assertAlmostEqual(r["closing_reported"], 94.0, places=6)


class TestLabelingAgainstTruth(unittest.TestCase):
    """标签提取必须能把合成数据的真值还原出来 —— 这是整个仓库的地基测试。"""

    @classmethod
    def setUpClass(cls):
        cls.d = generate(50, seed=3, blocks=["A", "B"], layers=["Q1"])
        cls.lab = extract_all(cls.d["prod_daily"], cls.d["well_event"])
        cls.m = cls.lab.merge(cls.d["truth"], on="well_id", suffixes=("_lab", "_true"))

    def test_no_rejected_labels_on_clean_synthetic(self):
        self.assertLess((self.lab["label_quality"] == "rejected").mean(), 0.05)

    def test_peak_time_error(self):
        e = (self.m["t_peak_lab"] - self.m["t_peak_true"]).abs()
        self.assertLess(e.median(), 12.0)

    def test_peak_rate_and_pressure_error(self):
        self.assertLess((self.m["q_peak_lab"] - self.m["q_peak_true"]).abs().median(), 1.0)
        self.assertLess((self.m["p_peak_lab"] - self.m["p_peak_true"]).abs().median(), 1.0)

    def test_oil_break_error(self):
        self.assertLess((self.m["t_oil_break_lab"] - self.m["t_oil_break_true"]).abs().median(), 5.0)

    def test_quality_gate_catches_degraded_wells(self):
        rep = check_wells(self.d["prod_daily"], self.d["well_master"], self.d["geo_static"])
        self.assertGreater(rep["passed"].mean(), 0.85)   # 大多数应通过
        self.assertLess(rep["passed"].mean(), 1.0)       # 但注入的劣质井要被拦下


class TestDCAPrerequisite(unittest.TestCase):
    """递减分析的前提是已过峰。

    这条曾经被违反过：一口投产 100 天、仍在爬坡的新井被套上 DCA，
    外推出 8 万吨 EUR 和"经济极限时刻 600 个月"，SEC 模块照单全收给了 PDP。
    一眼假的结论比没有结论更危险，所以钉死在测试里。
    """

    @classmethod
    def setUpClass(cls):
        from src.api import services as S
        cls.S = S
        m = S._tables()["master"]
        new = m[m["first_prod_date"] > "2026-01-01"]
        cls.new_well = new.iloc[0]["well_code_anon"] if len(new) else None

    def test_dca_refuses_pre_peak_well(self):
        if not self.new_well:
            self.skipTest("数据集中没有新井")
        with self.assertRaises(self.S.KernelError):
            self.S.fit_dca(self.new_well)

    def test_sec_reports_unavailable_instead_of_fake_number(self):
        if not self.new_well:
            self.skipTest("数据集中没有新井")
        r = self.S.sec_screen(self.new_well)
        self.assertFalse(r["dca_available"])
        self.assertIsNone(r["proved_reserves_t"])
        self.assertIn("在现有经济条件下可经济开采", r["summary"]["needs_human"])

    def test_eur_horizon_cap_is_flagged(self):
        f = dca.DCAFit("modified_hyperbolic", qi=500.0, di=0.001, b=0.9, d_min=0.001 / 12)
        e = dca.eur(f, q_econ=0.5, horizon_years=50)
        self.assertTrue(e["t_econ_capped"])   # 50 年内没跌到经济极限 -> 必须标记


if __name__ == "__main__":
    unittest.main()
