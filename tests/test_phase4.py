"""第四阶段：线性流验证队列、NDIC 公开数据适配器（自造数据，不联网）、SEC 披露加分项。"""
import unittest

import numpy as np
import pandas as pd

from src.ingest import ndic_public as N
from src.synth import linear_flow_cohort as LF


class TestLinearFlowCohort(unittest.TestCase):
    def test_slab_solution_asymptotics(self):
        td = np.array([1e-4, 1e-3, 4e-3])
        self.assertAlmostEqual(np.polyfit(np.log(td), np.log(LF.slab_sum(td)), 1)[0], -0.5, places=3)   # 线性流
        td2 = np.array([1.0, 1.5, 2.0])
        rate = np.diff(np.log(LF.slab_sum(td2, 64))) / np.diff(td2)
        np.testing.assert_allclose(rate, -np.pi ** 2 / 4, rtol=1e-6)                                 # 边界控制流指数递减

    def test_cohort_truth_is_consistent(self):
        wells = LF.generate(n_wells=12, months=72, seed=3)
        self.assertEqual({w["block"] for w in wells}, set(LF.AREA_TAU_MEDIAN))
        for w in wells:
            self.assertGreater(w["eur_true"], 0)
            self.assertLessEqual(w["volumes"].sum(), w["eur_true"] * 1.1)          # 观测累产不超过真值 EUR（容许计量噪声）
            self.assertTrue(np.all(np.diff(w["months"]) > 0))
            self.assertGreater(w["tau_matrix_months"], w["tau_srv_months"])

    def test_not_generated_by_either_fitted_formula(self):
        """考题公平：同一口井的真值曲线既不是单一 Arps 双曲，也不是物理约束分段公式能精确还原的。"""
        from src.reserves import dca
        w = LF.generate(n_wells=3, months=72, seed=5)[0]
        shape = LF.monthly_shape((w["tau_srv_months"], w["tau_matrix_months"]), (1.0, w["matrix_weight"]), 240, w["t0_months"])
        q = w["q_first_t_per_d"] / shape[0] * shape
        t = np.arange(240) + 0.5
        f = dca.fit(t, q, model="hyperbolic", recency_halflife_months=None)
        self.assertGreater(np.abs(dca.rate(t, f) / q - 1.0).max(), 0.02)
        from src.reserves import physics_dca as PD
        g = PD.fit(t, q, q_econ=0.1, cum_to_now=0.0, halflife_months=1e6)
        self.assertGreater(np.abs(PD.rate(t - t.min(), g) / q - 1.0).max(), 0.02)


class TestNDICAdapter(unittest.TestCase):
    def _fixture(self):
        yms = [p.strftime("%Y-%m") for p in pd.period_range("2018-01", "2022-12", freq="M")]
        rows = []
        for ym in yms:                                               # 井 A：2018-01 已在产 → 看不到投产期，排除
            rows.append(dict(well=N.anon_key(33000000010000), pool="Bakken", ym=ym, oil_t=500.0, days=30.0))
        for k, ym in enumerate(yms[3:]):                              # 井 B：2018-04 投产，第 2 个月达峰
            q = 60.0 if k == 1 else 55.0 / np.sqrt(k + 1)
            rows.append(dict(well=N.anon_key(33000000020000), pool="Three Forks", ym=ym, oil_t=q * 30.0, days=30.0))
        for ym in yms[12:30]:                                         # 井 C：出油月不足 36 个，排除
            rows.append(dict(well=N.anon_key(33000000030000), pool="Bakken", ym=ym, oil_t=300.0, days=30.0))
        df = pd.DataFrame(rows)
        gap = "2020-06"
        return df[df["ym"] != gap], [m for m in yms if m != gap]

    def test_selection_peak_gap_and_anonymization(self):
        df, avail = self._fixture()
        wells = N.build_series(df, avail)
        self.assertEqual(len(wells), 1)
        w = wells[0]
        self.assertTrue(w["well_code"].startswith("ND-") and "33000000020000" not in w["well_code"])
        self.assertEqual(w["first_prod_ym"], "2018-04")
        self.assertAlmostEqual(w["rates"][0], 60.0, places=6)                       # 峰值月起算
        self.assertAlmostEqual(w["cum_before_peak_t"], 55.0 * 30.0, places=6)
        gap_offset = pd.Period("2020-06", "M").ordinal - pd.Period("2018-05", "M").ordinal + 0.5
        self.assertNotIn(gap_offset, set(w["months"]))                             # 缺报表的月份是缺测，不是零产量
        self.assertIn(gap_offset - 1, set(w["months"]))

    def test_unit_conversion(self):
        self.assertAlmostEqual(N.BBL_TO_T, 0.158987 * 0.82)
        self.assertEqual(N.anon_key(33053039010000), N.anon_key("33053039010000"))


if __name__ == "__main__":
    unittest.main()


class TestPudDisclosure(unittest.TestCase):
    """Item 1203 披露草稿：数字必须与 PUD 滚动一致，引用的条款必须真实存在于条文库。"""

    @classmethod
    def setUpClass(cls):
        from src.api import services as S
        cls.S = S
        u = S.list_units()
        cls.scope = u["company"]["name"]
        cls.dates = [str(d) for d in u["evaluation_dates"]]

    def test_numbers_come_from_tracking_and_citations_exist(self):
        from src.agent.rag.retriever import Retriever
        S = self.S
        d = S.unit_pud_disclosure(self.scope, from_as_of=self.dates[0], to_as_of=self.dates[-1])
        trk = S.unit_category_tracking(self.scope, from_as_of=self.dates[0], to_as_of=self.dates[-1])
        closing = next(r["value"] for r in trk["pud"]["table"] if r["key"] == "closing")
        self.assertEqual([x["item"] for x in d["sections"]], ["Item 1203(a)", "Item 1203(b)", "Item 1203(c)", "Item 1203(d)"])
        self.assertIn(f"{closing:,.0f} t", d["sections"][0]["text"])
        self.assertEqual(d["sections"][0]["facts"][0]["value"], closing)
        self.assertIn(f"{trk['pud']['n_converted']} 个井位", d["sections"][1]["text"])
        valid = set(Retriever().citations())
        for sec in d["sections"]:
            for c in sec["citations"]:
                self.assertIn(c, valid)

    def test_first_date_has_no_prior_period(self):
        with self.assertRaises(self.S.KernelError):
            self.S.unit_pud_disclosure(self.scope, from_as_of=self.dates[-1], to_as_of=self.dates[-1])


class TestStandardizedMeasureMath(unittest.TestCase):
    def test_exponential_profile_matches_reserves(self):
        from src.sec import finance
        v = finance.exponential_profile(12000.0, 300.0, horizon_months=600)
        self.assertAlmostEqual(v.sum(), 12000.0, delta=12000.0 * 0.005)
        self.assertEqual(len(finance.exponential_profile(0.0, 300.0)), 0)

    def test_identities_tax_and_discount(self):
        from src.sec import finance
        pd_vol = finance.exponential_profile(20000.0, 400.0)
        wells = [dict(start_month=13, volume_t=np.full(60, 150.0), capex_wan=500.0)]
        kw = dict(pd_volume_t=pd_vol, pd_opex_usd_per_t=120.0, pud_wells=wells, opex_per_well_month_usd=3000.0,
                  net_revenue_usd_per_t=450.0, fx_cny_per_usd=7.1, development_now_wan=30.0)
        sched = finance.cash_flow_schedule(**kw)
        self.assertAlmostEqual(sched["development_cost_wan"][12], 500.0)                 # 投资记在投产前一个月（第 2 年）
        self.assertAlmostEqual(sched["development_cost_wan"][0], 30.0)
        a = finance.standardized_measure(sched, tax_rate=0.0)
        b = finance.standardized_measure(sched, tax_rate=0.25)
        self.assertAlmostEqual(a["standardized_measure_wan"], a["pre_tax_discounted_wan"], places=6)
        self.assertLess(b["standardized_measure_wan"], a["standardized_measure_wan"])
        for r in (a, b):
            self.assertAlmostEqual(r["future_net_cash_flows_wan"],
                                   r["future_cash_inflows_wan"] - r["future_production_costs_wan"]
                                   - r["future_development_costs_wan"] - r["future_income_tax_wan"], places=6)
            self.assertAlmostEqual(r["standardized_measure_wan"], r["future_net_cash_flows_wan"] - r["discount_wan"], places=6)
            self.assertAlmostEqual(sum(y["discounted_wan"] for y in r["annual"]), r["standardized_measure_wan"], places=6)
        c = finance.standardized_measure(sched, tax_rate=0.25, discount_rate=0.2)
        self.assertLess(c["standardized_measure_wan"], b["standardized_measure_wan"])


class TestStandardizedMeasureService(unittest.TestCase):
    def test_summary_units_and_changes_close(self):
        from src.api import services as S
        u = S.list_units()
        d = S.unit_standardized_measure(u["company"]["name"], as_of=str(u["evaluation_dates"][-1]))
        sm = d["summary"]
        self.assertAlmostEqual(sum(x["standardized_measure_wan"] for x in d["units"]), sm["standardized_measure_wan"], delta=1.0)
        self.assertLessEqual(sm["standardized_measure_wan"], sm["future_net_cash_flows_wan"] + 1.0)
        rows = {r["key"]: r["value"] for r in d["changes"]["table"]}
        body = sum(v for k, v in rows.items() if k not in ("opening", "closing"))
        self.assertAlmostEqual(rows["opening"] + body, rows["closing"], delta=1.0)
        self.assertLess(rows["sales_net"], 0)
        self.assertGreater(rows["accretion"], 0)


class TestProbabilisticAggregation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.best = rng.lognormal(np.log(800.0), 0.6, 300)
        self.groups = np.array(["A", "B", "C"])[np.arange(300) % 3]
        self.bucket = np.full(300, 24)
        self.errors = {24: np.concatenate([rng.normal(0.0, 0.09, 180), rng.normal(-0.45, 0.15, 20)])}   # 带左尾

    def test_single_group_fully_correlated_equals_sum_of_well_quantiles(self):
        """只有一个分组且 ρ = 1：所有井取同一分位，汇总的 10% 分位 = 逐井 10% 分位之和（误差用平滑样本，避免分位函数过陡放大抽样噪声）。"""
        from src.sec import probabilistic as P
        smooth = {24: np.random.default_rng(5).normal(0.0, 0.1, 400)}
        r = P.aggregate(self.best, np.full(len(self.best), "A"), self.bucket, smooth, rho=1.0, n_sims=3000, seed=1)
        self.assertAlmostEqual(r["p10_t"], r["sum_well_p10_t"], delta=0.01 * r["sum_well_p10_t"])

    def test_rho_one_synchronizes_within_group_only(self):
        """ρ = 1 只让同一区块内的井同步；区块之间仍独立，多分组时汇总 10% 分位仍高于逐井分位之和。"""
        from src.sec import probabilistic as P
        r = P.aggregate(self.best, self.groups, self.bucket, self.errors, rho=1.0, n_sims=3000, seed=1)
        self.assertGreater(r["p10_t"], r["sum_well_p10_t"])

    def test_independence_diversifies_low_estimate(self):
        from src.sec import probabilistic as P
        r0 = P.aggregate(self.best, self.groups, self.bucket, self.errors, rho=0.0, n_sims=3000, seed=1)
        r1 = P.aggregate(self.best, self.groups, self.bucket, self.errors, rho=1.0, n_sims=3000, seed=1)
        self.assertGreater(r0["p10_t"], r1["p10_t"])
        self.assertGreater(r0["aggregation_gain_t"], 0)
        self.assertLessEqual(r0["p10_t"], r0["p50_t"])
        self.assertLessEqual(r0["p50_t"], r0["p90_t"])

    def test_icc_detects_group_effect(self):
        from src.sec import probabilistic as P
        rng = np.random.default_rng(2)
        g = np.repeat(np.arange(10), 30)
        v = rng.normal(0, 1, 10)[g] + rng.normal(0, 1, 300)
        self.assertGreater(P.icc_oneway(v, g)["icc"], 0.3)
        self.assertLess(P.icc_oneway(rng.normal(0, 1, 300), g)["icc"], 0.15)


class TestProbabilisticService(unittest.TestCase):
    def test_company_aggregation_is_consistent(self):
        from src.api import services as S
        u = S.list_units()
        p = S.unit_probabilistic_reserves(u["company"]["name"], as_of=str(u["evaluation_dates"][-1]), n_sims=1500)
        by_rho = {r["rho"]: r for r in p["results"]}
        self.assertIn(0.0, by_rho)
        self.assertIn(1.0, by_rho)
        self.assertGreaterEqual(by_rho[0.0]["p10_t"], by_rho[1.0]["p10_t"])
        self.assertEqual(sum(x["n_wells"] for x in p["units"]), p["n_wells"])
        for r in p["results"]:
            self.assertLessEqual(r["p10_t"], r["p50_t"])
            self.assertLessEqual(r["p50_t"], r["p90_t"])
