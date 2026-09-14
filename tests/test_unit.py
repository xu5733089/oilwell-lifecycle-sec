"""SEC 单元"新-老-措"构成评估的内核测试。

合成数据的真值（扩边标记、措施增油）只在这里和评测里用 —— 业务代码从不读它。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import config, indicators as indicator_spec          # noqa: E402
from src.reserves import dca, workload as wl                          # noqa: E402
from src.sec import composition, economics, indicators, reconcile     # noqa: E402
from src.synth.well_generator import generate                         # noqa: E402

SU = config()["sec_unit"]


def _fixture():
    """两区块、单层系、每区块约 200 口井 —— 与正式数据的井网密度一致。"""
    if not hasattr(_fixture, "d"):
        d = generate(400, seed=11, blocks=["A", "B"], layers=["Q1"])
        d["monthly"] = wl.to_monthly(d["prod_daily"])
        _fixture.d = d
    return _fixture.d


def _eval(d, block: str, as_of: str, deck: str, scenario: str = "sec") -> dict:
    master = d["well_master"]
    unit = master[master["block"] == block]
    ids = set(unit["well_id"])
    nw = wl.identify_new_wells(master, as_of, SU["period_months"],
                               SU["new_well"]["radius_m"], SU["new_well"]["min_old_neighbors"])
    qe = economics.economic_limit_rate(deck, scenario)["q_econ"]
    return composition.evaluate_unit(
        monthly=d["monthly"][d["monthly"]["well_id"].isin(ids)], master=unit,
        events=d["well_event"], new_wells=nw, as_of=as_of, q_econ_well=qe,
        period_months=SU["period_months"], measure_cfg=SU["measure"],
        min_fit_months=SU["decline"]["min_fit_months"],
        type_curve_min_history=SU["type_curve"]["min_history_months"])


class TestSelectStart(unittest.TestCase):
    def test_rise_then_decline(self):
        rng = np.random.default_rng(0)
        t = np.arange(60, dtype=float)
        q = np.where(t < 18, 50 * np.exp(0.06 * t), 50 * np.exp(0.06 * 18) * np.exp(-0.03 * (t - 18)))
        q = q * rng.lognormal(0, 0.03, len(t))
        s = dca.select_start(t, q)
        self.assertLessEqual(abs(s["start_index"] - 18), 3, s)
        self.assertTrue(s["declining"])

    def test_step_change_from_large_workover(self):
        rng = np.random.default_rng(1)
        t = np.arange(70, dtype=float)
        q = 80 * np.exp(-0.02 * t) * np.where(t >= 40, 1.6, 1.0) * rng.lognormal(0, 0.03, len(t))
        s = dca.select_start(t, q)
        self.assertLessEqual(abs(s["start_index"] - 40), 3, s)

    def test_clean_decline_has_no_break(self):
        rng = np.random.default_rng(2)
        t = np.arange(48, dtype=float)
        q = 30 * np.exp(-0.025 * t) * rng.lognormal(0, 0.04, len(t))
        self.assertEqual(dca.select_start(t, q)["start_index"], 0)


class TestNewWellIdentification(unittest.TestCase):
    """提采新井判定是业内仍靠人工甄别的环节 —— 这里用合成真值量化自动识别的准确率。"""

    def test_accuracy_against_truth(self):
        d = _fixture()
        truth = d["truth"].set_index("well_id")["is_extension"]
        got, exp = [], []
        for as_of in ("2025-12-31", "2026-12-31"):
            nw = wl.identify_new_wells(d["well_master"], as_of, SU["period_months"],
                                       SU["new_well"]["radius_m"], SU["new_well"]["min_old_neighbors"])
            got += (nw["category"] == "extension").tolist()
            exp += truth.loc[nw["well_id"]].astype(bool).tolist()
        got, exp = np.array(got), np.array(exp)
        self.assertGreater(exp.sum(), 5)
        self.assertGreater((~exp).sum(), 5)
        self.assertGreaterEqual((got == exp).mean(), 0.90)

    def test_only_period_wells_are_new(self):
        d = _fixture()
        nw = wl.identify_new_wells(d["well_master"], "2026-12-31", 12, 1500, 3)
        self.assertTrue((nw["first_prod_date"] >= "2026-01-01").all())
        self.assertTrue((nw["first_prod_date"] <= "2026-12-31").all())


class TestMeasureEffect(unittest.TestCase):
    def test_realized_increment_against_truth(self):
        d = _fixture()
        et = d["event_truth"]
        et = et[et["realized_inc_t"] > 150]
        errs = []
        for r in et.itertuples(index=False):
            wm = d["monthly"][d["monthly"]["well_id"] == r.well_id]
            e = wl.measure_effect(wm, r.dt[:7], "2026-12", 1.0, **SU["measure"])
            if e["status"] in ("ok", "insufficient_post"):
                errs.append(abs(e["realized_inc_t"] - r.realized_inc_t) / r.realized_inc_t)
        self.assertGreaterEqual(len(errs), 20)
        self.assertLessEqual(float(np.median(errs)), 0.25)

    def test_short_history_is_refused_not_guessed(self):
        wm = pd.DataFrame(dict(ym=["2026-01", "2026-02", "2026-03", "2026-04"],
                               oil_t=[300.0, 290.0, 500.0, 480.0], days_on=[31, 28, 31, 30]))
        e = wl.measure_effect(wm, "2026-03", "2026-04", 1.0, **SU["measure"])
        self.assertEqual(e["status"], "insufficient_pre")
        self.assertIsNone(e["inc_eur_t"])

    def test_increment_remaining_bounds(self):
        """增储不为负；措施后总产量也低于经济极限时，有无措施都不经济，增储为零。

        注意增储不随经济极限单调（见 measure_increment_remaining 的说明），这里不测单调性。
        """
        e = dict(baseline_fit=dca.DCAFit("exponential", qi=10.0, di=0.03, b=0.0, d_min=0.0).to_dict(),
                 t_now_month=12.0, inc_now_t_per_d=3.0, d_inc_month=0.05)
        self.assertGreater(wl.measure_increment_remaining(e, 0.5), 0)
        self.assertGreater(wl.measure_increment_remaining(e, 5.0), 0)
        self.assertEqual(wl.measure_increment_remaining(e, 50.0), 0.0)


class TestComposition(unittest.TestCase):
    def test_monthly_components_sum_to_total(self):
        d = _fixture()
        comp = wl.monthly_composition(d["monthly"], d["well_master"], [], start_ym="2024-01",
                                      end_ym="2026-12", new_since_ym="2026-01",
                                      measure_since_ym="2026-01")
        s = comp["new_oil_t"] + comp["measure_inc_t"] + comp["old_base_t"]
        self.assertTrue(np.allclose(s, comp["total_oil_t"]))
        self.assertTrue((comp.loc[comp["ym"] < "2026-01", "new_oil_t"] == 0).all())

    def test_decline_rate_recovers_known_exponential(self):
        yms = [wl.idx_to_ym(wl.ym_to_idx("2024-01") + k) for k in range(24)]
        days = np.array([wl.days_in_month(y) for y in yms], float)
        base = 1000 * np.exp(-0.01 * np.arange(24)) * days
        comp = pd.DataFrame(dict(ym=yms, old_base_t=base, measure_inc_t=np.zeros(24)))
        r = wl.decline_rates(comp, "2024-01", "2025-12")
        self.assertAlmostEqual(r["natural_monthly_pct"], (1 - np.exp(-0.01)) * 100, places=6)
        self.assertAlmostEqual(r["natural_annual_pct"], (1 - np.exp(-0.12)) * 100, places=6)

    def test_unit_evaluation_structure(self):
        ev = _eval(_fixture(), "A", "2026-12-31", "deck_2026_12")
        keys = [c["key"] for c in ev["components"]]
        self.assertEqual(keys, ["old_base", "measure", "new_infill", "extension"])
        self.assertAlmostEqual(ev["total_t"], sum(c["reserves_t"] for c in ev["components"]), places=6)
        self.assertEqual(ev["old_base"]["status"], "ok")
        self.assertGreater(ev["total_t"], 0)
        self.assertTrue(all(c["reserves_t"] >= 0 for c in ev["components"]))

    def test_measure_on_new_well_is_not_counted_twice(self):
        """本期新井上的措施：效果已在新井储量里，不得再进措施增储（合成数据自然触发不到，这里注入一次）。"""
        d = _fixture()
        m = d["well_master"]
        cand = m[(m["block"] == "A") & (m["first_prod_date"] >= "2025-01-01")
                 & (m["first_prod_date"] <= "2025-02-28")]
        self.assertGreater(len(cand), 0)
        wid = cand.iloc[0]["well_id"]
        d2 = dict(d, well_event=pd.concat([d["well_event"], pd.DataFrame(
            [dict(well_id=wid, dt="2025-09-01", day_index=None, event_type="frac", note="注入")])],
            ignore_index=True))
        ev = _eval(d2, "A", "2025-12-31", "deck_2025_12")
        e = next(x for x in ev["measures"] if x["well_id"] == wid)
        self.assertTrue(e["on_new_well"])
        self.assertFalse(e["in_period"])
        counted = sum(1 for x in ev["measures"] if x["in_period"] and x["status"] == "ok")
        self.assertEqual(next(c for c in ev["components"] if c["key"] == "measure")["n_items"], counted)

    def test_lower_price_scenario_does_not_raise_reserves(self):
        d = _fixture()
        sec = _eval(d, "A", "2026-12-31", "deck_2026_12", "sec")
        imp = _eval(d, "A", "2026-12-31", "deck_2026_12", "impairment")
        self.assertLessEqual(imp["total_t"], sec["total_t"] + 1e-6)


class TestReconcileAndSensitivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = _fixture()
        cls.open = _eval(d, "B", "2025-12-31", "deck_2025_12")
        cls.close = _eval(d, "B", "2026-12-31", "deck_2026_12")
        cls.close_prev_price = _eval(d, "B", "2026-12-31", "deck_2025_12")

    def test_reconcile_closes(self):
        r = composition.reconcile_auto(self.open, self.close, self.close_prev_price)
        self.assertTrue(r["balanced"])
        self.assertAlmostEqual(r["closing_reported"], round(self.close["total_t"], 1), places=1)
        self.assertLess(next(x["value"] for x in r["table"] if x["key"] == "production"), 0)
        self.assertGreater(r["depletion_rate_pct"], 0)
        self.assertLess(r["depletion_rate_pct"], 100)

    def test_sensitivity_directions(self):
        deck = "deck_2026_12"
        price = economics.scenario_params(deck)["price_usd_bbl"]
        s = composition.sensitivity(
            self.close, lambda p, f: economics.economic_limit_rate(deck, price_usd_bbl=p, opex_factor=f)["q_econ"],
            price)
        c = {x["param"]: x["delta_t"] for x in s["curves"]}
        self.assertTrue(np.all(np.diff(c["price"]) >= -1e-6), c["price"])
        self.assertTrue(np.all(np.diff(c["opex"]) <= 1e-6), c["opex"])
        self.assertTrue(np.all(np.diff(c["decline"]) <= 1e-6), c["decline"])
        self.assertAlmostEqual(sum(w["weight"] for w in s["weights"]), 1.0, places=6)


class TestEconomicsScenarios(unittest.TestCase):
    def test_sec_scenario_matches_legacy_limit(self):
        legacy = economics.economic_limit_rate("deck_2026_12")["q_econ"]
        self.assertEqual(economics.economic_limit_rate("deck_2026_12", "sec")["q_econ"], legacy)

    def test_impairment_price_raises_economic_limit(self):
        sec = economics.economic_limit_rate("deck_2026_12", "sec")["q_econ"]
        imp = economics.economic_limit_rate("deck_2026_12", "impairment")["q_econ"]
        self.assertGreater(imp, sec)

    def test_deck_lookup_by_as_of(self):
        self.assertEqual(economics.deck_for("2026-12-31"), "deck_2026_12")
        with self.assertRaises(KeyError):
            economics.deck_for("2026-06-30")

    def test_reconcile_legacy_keys_still_balance(self):
        r = reconcile.build(dict(opening=100.0, production=-10.0, measure=3.0, extension=2.0))
        self.assertTrue(r["balanced"])
        self.assertAlmostEqual(r["closing_calculated"], 95.0)


class TestIndicators(unittest.TestCase):
    def test_scores_are_bounded_and_directional(self):
        """锚点取自配置：越过优值锚点得 100，越过差值锚点得 0（两个指标方向相反）。"""
        spec = indicator_spec()
        ind = spec["indicators"]

        def beyond(key: str, anchor: str) -> float:
            g, b = ind[key]["good"], ind[key]["bad"]
            step = abs(g - b)
            return (g + step if g > b else g - step) if anchor == "good" else (b - step if g > b else b + step)

        good = indicators.score(dict(natural_decline_pct=beyond("natural_decline_pct", "good"),
                                     opex_usd_per_t=beyond("opex_usd_per_t", "good")), spec)
        bad = indicators.score(dict(natural_decline_pct=beyond("natural_decline_pct", "bad"),
                                    opex_usd_per_t=beyond("opex_usd_per_t", "bad")), spec)
        g = {r["key"]: r["score"] for r in good["indicators"]}
        b = {r["key"]: r["score"] for r in bad["indicators"]}
        self.assertEqual(g["natural_decline_pct"], 100.0)
        self.assertEqual(b["natural_decline_pct"], 0.0)
        self.assertEqual(g["opex_usd_per_t"], 100.0)
        self.assertIsNone(g["water_cut_pct"])          # 没有值就不打分，不拿 0 分顶替

    def test_compute_derived_values(self):
        v = indicators.compute(production_t=12000.0, plan_production_t=12500.0,
                               natural_decline_pct=10.0, comprehensive_decline_pct=6.0,
                               water_cut_pct=70.0, water_cut_prev_pct=66.0, n_wells=50,
                               n_producing=45, new_oil_t=2000.0, plan_new_oil_t=2500.0,
                               n_measures_evaluated=10, n_measures_effective=8, pdp_t=96000.0,
                               producing_well_months=540.0, opex_per_well_month_usd=9000.0,
                               net_revenue_usd_per_t=360.0)
        self.assertAlmostEqual(v["open_well_rate_pct"], 90.0)
        self.assertAlmostEqual(v["water_cut_rise_pct"], 4.0)
        self.assertAlmostEqual(v["reserve_production_ratio"], 8.0)
        self.assertAlmostEqual(v["opex_usd_per_t"], 405.0)
        self.assertAlmostEqual(v["profit_usd_per_t"], -45.0)


if __name__ == "__main__":
    unittest.main()
