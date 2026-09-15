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
