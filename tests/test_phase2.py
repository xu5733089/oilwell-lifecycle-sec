"""第二阶段：PDNP / PUD 类别跟踪、折耗与减值、指标锚点方案、数据导入。

前半部分是纯计算测试（合成夹具，不读库）；后半部分走真实库（需先 init + train）。
"""
from __future__ import annotations

import base64
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import config, indicators as indicator_spec          # noqa: E402
from src.reserves import workload as wl                              # noqa: E402
from src.sec import composition, economics, finance, indicators      # noqa: E402
from src.synth.well_generator import generate                        # noqa: E402

SU = config()["sec_unit"]


def _fixture():
    if not hasattr(_fixture, "d"):
        d = generate(400, seed=11, blocks=["A", "B"], layers=["Q1"], reactivation_frac=0.4)
        d["monthly"] = wl.to_monthly(d["prod_daily"])
        _fixture.d = d
    return _fixture.d


def _producing_xy(d, as_of):
    end = as_of[:7]
    lo = wl.idx_to_ym(wl.ym_to_idx(end) - 2)
    m = d["monthly"]
    ids = set(m[(m["ym"] >= lo) & (m["ym"] <= end) & (m["oil_t"] > 0)]["well_id"])
    wm = d["well_master"]
    return wm[wm["well_id"].isin(ids)][["x_off", "y_off"]].values.tolist()


def _eval(d, block, as_of, deck, locations=None):
    master = d["well_master"]
    unit = master[master["block"] == block]
    ids = set(unit["well_id"])
    nw = wl.identify_new_wells(master, as_of, SU["period_months"], SU["new_well"]["radius_m"],
                               SU["new_well"]["min_old_neighbors"])
    econ = economics.economic_limit_rate(deck, "sec")
    cfg = dict(pdnp=dict(max_shut_in_months=SU["pdnp"]["max_shut_in_months"], restart_min_t=0.0),
               pud=SU["pud"], locations=locations, producing_xy=_producing_xy(d, as_of),
               drilled_first_prod=dict(zip(master["well_id"], master["first_prod_date"])),
               net_revenue_usd_per_t=econ["net_revenue_usd_per_tonne"],
               opex_per_well_month_usd=econ["opex_per_well_month_usd"], fx_cny_per_usd=7.1, default_capex_wan=900)
    return composition.evaluate_unit(
        monthly=d["monthly"][d["monthly"]["well_id"].isin(ids)], master=unit, events=d["well_event"],
        new_wells=nw, as_of=as_of, q_econ_well=econ["q_econ"], period_months=SU["period_months"],
        measure_cfg=SU["measure"], min_fit_months=SU["decline"]["min_fit_months"],
        type_curve_min_history=SU["type_curve"]["min_history_months"], category_cfg=cfg)


class TestPDNP(unittest.TestCase):
    @staticmethod
    def _well(months_on: int, idle: int):
        start = wl.ym_to_idx("2022-01")
        rows = []
        for k in range(months_on + idle):
            ym = wl.idx_to_ym(start + k)
            q = 20.0 * math.exp(-0.04 * k) if k < months_on else 0.0
            rows.append(dict(ym=ym, oil_t=q * wl.days_in_month(ym)))
        return pd.DataFrame(rows), wl.idx_to_ym(start + months_on + idle - 1)

    def test_recently_shut_in_well_is_booked(self):
        wm, as_of_ym = self._well(36, 6)
        out = composition.old_wells_reserves({"W": wm}, ["W"], [], as_of_ym, 1.0,
                                             pdnp_cfg=dict(max_shut_in_months=24, restart_min_t=0.0))
        row = out["wells"][0]
        self.assertEqual(row["status"], "not_producing")
        self.assertEqual(row["pdnp_status"], "booked")
        self.assertGreater(row["pdnp_t"], 0)
        self.assertEqual(out["remaining_low_t"], 0.0)                 # 不计入 PDP
        self.assertAlmostEqual(out["pdnp_t"], row["pdnp_t"])

    def test_long_shut_in_and_uneconomic_are_excluded(self):
        wm, as_of_ym = self._well(36, 30)
        row = composition.old_wells_reserves({"W": wm}, ["W"], [], as_of_ym, 1.0,
                                             pdnp_cfg=dict(max_shut_in_months=24, restart_min_t=0.0))["wells"][0]
        self.assertEqual(row["pdnp_status"], "long_shut_in")
        wm, as_of_ym = self._well(36, 6)
        row = composition.old_wells_reserves({"W": wm}, ["W"], [], as_of_ym, 1.0,
                                             pdnp_cfg=dict(max_shut_in_months=24, restart_min_t=1e9))["wells"][0]
        self.assertEqual(row["pdnp_status"], "uneconomic")
        self.assertEqual(row["pdnp_t"], 0.0)


class TestPUDRules(unittest.TestCase):
    TC = dict(rate=[max(30.0 * math.exp(-0.05 * k), 0.1) for k in range(60)])

    def _run(self, loc, capex=900, xy=None, drilled=None):
        cfg = dict(radius_m=1000, min_producing_neighbors=3, horizon_years=5, certainty_factor=0.85)
        xy = [[0, 0], [300, 0], [0, 300]] if xy is None else xy
        base = dict(location_id="L", x_off=100, y_off=100, planned_drill_ym="2028-06", first_booked_as_of="2026-12-31",
                    drilled_well_id=None, status="planned", capex_wan=None)
        return composition.pud_reserves([dict(base, **loc)], np.array(xy), "2026-12-31", self.TC, 1.0, cfg=cfg,
                                        drilled_first_prod=drilled or {}, net_revenue_usd_per_t=360,
                                        opex_per_well_month_usd=9000, fx_cny_per_usd=7.1,
                                        default_capex_wan=capex)["locations"][0]

    def test_each_rule(self):
        self.assertEqual(self._run({})["status"], "booked")
        self.assertEqual(self._run({}, xy=[[0, 0]])["status"], "not_certain")
        self.assertEqual(self._run({"first_booked_as_of": "2021-12-31"})["status"], "expired_5yr")
        self.assertEqual(self._run({"planned_drill_ym": "2033-01"})["status"], "beyond_5yr")
        self.assertEqual(self._run({}, capex=1e9)["status"], "uneconomic")
        self.assertEqual(self._run({"first_booked_as_of": "2027-12-31"})["status"], "not_booked_yet")
        self.assertEqual(self._run({"drilled_well_id": "W1"}, drilled={"W1": "2026-05-01"})["status"], "drilled")
        self.assertEqual(self._run({"drilled_well_id": "W1"}, drilled={"W1": "2027-05-01"})["status"], "booked")
        self.assertEqual(self._run({"status": "cancelled"})["status"], "cancelled")

    def test_only_booked_locations_carry_reserves(self):
        self.assertGreater(self._run({})["reserves_t"], 0)
        self.assertEqual(self._run({}, xy=[[0, 0]])["reserves_t"], 0.0)


class TestCategoryReconcile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = _fixture()
        master = d["well_master"]
        olds = master[(master["block"] == "B") & (master["first_prod_date"] < "2025-01-01")]
        locs = []
        for k, r in enumerate(olds.head(10).itertuples(index=False)):
            locs.append(dict(location_id=f"L{k}", x_off=r.x_off + 500, y_off=r.y_off, planned_drill_ym="2027-06",
                             first_booked_as_of="2025-12-31" if k % 2 else "2026-12-31",
                             drilled_well_id=None, status="planned", capex_wan=None))
        locs.append(dict(location_id="OLD", x_off=olds.iloc[0].x_off, y_off=olds.iloc[0].y_off + 400,
                         planned_drill_ym="2026-06", first_booked_as_of="2021-12-31", drilled_well_id=None,
                         status="planned", capex_wan=None))
        cls.open = _eval(d, "B", "2025-12-31", "deck_2025_12", locs)
        cls.close = _eval(d, "B", "2026-12-31", "deck_2026_12", locs)
        cls.close_prev = _eval(d, "B", "2026-12-31", "deck_2025_12", locs)

    def test_reconcile_still_closes_with_category_change(self):
        r = composition.reconcile_auto(self.open, self.close, self.close_prev)
        self.assertTrue(r["balanced"])
        det = r["category_change_detail"]
        self.assertGreater(det["n_reactivated"] + det["n_shut_in"], 0)
        cc = next(x["value"] for x in r["table"] if x["key"] == "category_change")
        self.assertAlmostEqual(cc, round(det["reactivated_t"] + det["shut_in_t"], 1), places=0)

    def test_rollforward_tables_close(self):
        rf = composition.category_rollforward(self.open, self.close)
        for cat in ("pud", "pdnp"):
            rows = {x["key"]: x["value"] for x in rf[cat]["table"]}
            moves = sum(v for k, v in rows.items() if k not in ("opening", "closing"))
            self.assertAlmostEqual(rows["opening"] + moves, rows["closing"], places=6, msg=cat)
        self.assertIn("OLD", rf["pud"]["expired"])                    # 五年规则移出
        self.assertAlmostEqual(self.close["total_proved_t"],
                               self.close["total_t"] + self.close["pdnp"]["reserves_t"] + self.close["pud"]["reserves_t"])


class TestFinance(unittest.TestCase):
    def test_depletion(self):
        d = finance.depletion(1000.0, 200.0, 10.0, 90.0)
        self.assertAlmostEqual(d["depletion_rate"], 0.1)
        self.assertAlmostEqual(d["depletion_wan"], 120.0)
        self.assertAlmostEqual(d["carrying_after_depletion_wan"], 1080.0)

    def test_recoverable_matches_monthly_discounting(self):
        r = finance.recoverable_amount(12000.0, 300.0, 350.0, 120.0, 0.10, 7.1)
        dec = r["decline_month"]
        brute = sum(300.0 * math.exp(-dec * t) * 1.10 ** (-t / 12) for t in range(3000)) * 230.0 * 7.1 / 1e4
        self.assertAlmostEqual(r["recoverable_wan"], brute, delta=brute * 1e-6)
        undiscounted = sum(300.0 * math.exp(-dec * t) for t in range(5000))
        self.assertAlmostEqual(undiscounted, 12000.0, delta=1.0)     # 剖面累计 = 储量

    def test_negative_cash_and_impairment(self):
        self.assertEqual(finance.recoverable_amount(1000.0, 50.0, 100.0, 150.0, 0.1, 7.1)["recoverable_wan"], 0.0)
        i = finance.impairment(500.0, 420.0)
        self.assertTrue(i["impaired"])
        self.assertAlmostEqual(i["impairment_wan"], 80.0)
        self.assertAlmostEqual(i["closing_nbv_wan"], 420.0)
        self.assertFalse(finance.impairment(400.0, 420.0)["impaired"])


class TestIndicatorSpec(unittest.TestCase):
    def test_validation(self):
        ref = indicator_spec()
        ok = {k: dict(good=v["good"], bad=v["bad"], weight=v["weight"]) for k, v in ref["indicators"].items()}
        self.assertEqual(indicators.validate_spec(dict(indicators=ok), ref), [])
        bad = {k: dict(v) for k, v in ok.items()}
        bad["natural_decline_pct"].update(good=30, bad=30)
        bad["opex_usd_per_t"]["weight"] = -1
        bad.pop("water_cut_pct")
        errs = "；".join(indicators.validate_spec(dict(indicators=bad), ref))
        self.assertIn("不能相等", errs)
        self.assertIn("不能为负", errs)
        self.assertIn("缺少指标", errs)


# --------------------------------------------------------------------------- #
# 以下走真实库
from src.api import services as S            # noqa: E402
from src.ingest import imports               # noqa: E402

COMPANY = None


def _company():
    global COMPANY
    if COMPANY is None:
        COMPANY = S.list_units()["company"]["name"]
    return COMPANY


class TestCategoryServices(unittest.TestCase):
    def test_categories_sum_and_rules_present(self):
        c = S.unit_proved_categories(_company(), "2026-12-31")
        parts = sum(x["reserves_t"] for x in c["categories"])
        self.assertAlmostEqual(parts, c["total_proved_t"], delta=1.0)
        st = c["pud"]["n_by_status"]
        for k in ("booked", "expired_5yr", "not_certain", "beyond_5yr", "drilled"):
            self.assertIn(k, st, k)
        self.assertGreater(c["pdnp"]["n_booked"], 0)

    def test_tracking_closes_and_reconcile_has_category_change(self):
        t = S.unit_category_tracking(_company(), "2025-12-31", "2026-12-31")
        self.assertTrue(t["pud"]["balanced"])
        self.assertTrue(t["pdnp"]["balanced"])
        self.assertGreater(t["pud"]["n_converted"], 0)
        self.assertTrue(0 < t["pud"]["conversion_rate_pct"] <= 100)
        r = S.unit_reconcile(_company(), from_as_of="2025-12-31", to_as_of="2026-12-31")
        self.assertTrue(r["balanced"])
        cc = next(x["value"] for x in r["table"] if x["key"] == "category_change")
        self.assertNotEqual(cc, 0.0)

    def test_depletion_rolls_forward(self):
        unit = S.list_units()["plants"][0]["units"][0]["unit_id"]
        a = S.unit_depletion_impairment(unit, "2025-12-31")["units"][0]
        b = S.unit_depletion_impairment(unit, "2026-12-31")["units"][0]
        self.assertAlmostEqual(b["opening_nbv_wan"], a["closing_nbv_wan"], delta=0.2)
        self.assertAlmostEqual(b["closing_nbv_wan"], b["carrying_wan"] - b["impairment_wan"], delta=0.2)
        s = S.unit_depletion_impairment(_company(), "2026-12-31")["summary"]
        self.assertEqual(s["n_units"], 9)


class TestIndicatorProfiles(unittest.TestCase):
    def tearDown(self):
        # 无论断言成败都清掉本测试写入的自定义方案，免得残留在库里、出现在界面上
        for p in S.list_indicator_profiles()["profiles"]:
            if not p["is_builtin"] and p["name"] in ("测试方案", "坏方案"):
                S.delete_indicator_profile(p["profile_id"])

    def test_builtin_protected_custom_roundtrip(self):
        lst = S.list_indicator_profiles()
        builtin = next(p for p in lst["profiles"] if p["is_builtin"])
        with self.assertRaises(S.KernelError):
            S.save_indicator_profile(name="x", spec=builtin["spec"], profile_id=builtin["profile_id"])
        spec = {k: dict(v) for k, v in builtin["spec"]["indicators"].items()}
        base = S.unit_indicators(_company(), "2026-12-31", profile=builtin["profile_id"])
        value = next(x["value"] for x in base["periods"][0]["indicators"] if x["key"] == "natural_decline_pct")
        spec["natural_decline_pct"].update(good=0.0, bad=value * 2)       # 实际值落在两锚点正中 → 50 分
        bad = {k: dict(v) for k, v in spec.items()}
        bad["natural_decline_pct"]["bad"] = bad["natural_decline_pct"]["good"]           # 优值 = 差值：不合法
        with self.assertRaises(S.KernelError):
            S.save_indicator_profile(name="坏方案", spec=dict(indicators=bad))
        saved = S.save_indicator_profile(name="测试方案", spec=dict(indicators=spec))["profile"]
        try:
            a = S.unit_indicators(_company(), "2026-12-31", profile=builtin["profile_id"])
            b = S.unit_indicators(_company(), "2026-12-31", profile=saved["profile_id"])
            score = lambda r: next(x["score"] for x in r["periods"][0]["indicators"] if x["key"] == "natural_decline_pct")
            self.assertNotEqual(score(a), score(b))
            self.assertEqual(b["profile"]["name"], "测试方案")
        finally:
            S.delete_indicator_profile(saved["profile_id"])
        self.assertTrue(any(p["is_default"] for p in S.list_indicator_profiles()["profiles"]))


class TestImports(unittest.TestCase):
    @staticmethod
    def _b64(text: str) -> str:
        return base64.b64encode(text.encode("utf-8")).decode()

    def test_preview_flags_errors(self):
        csv = "单元号,年月,计划新井数,计划新井产油(t),计划措施井次,计划措施增油(t),计划老井产油(t)\n" \
              "NOPE,2030-13,1.5,-3,1,2,3\nSEC_GLA_Q1,2030-01,1,100,1,50,1000\nSEC_GLA_Q1,2030-01,1,100,1,50,1000\n"
        p = imports.preview("plan", "plan.csv", self._b64(csv))
        msgs = "；".join(e["message"] for e in p["errors"])
        for m in ("单元号不存在", "年月格式", "应为整数", "不能小于", "主键重复"):
            self.assertIn(m, msgs)
        self.assertFalse(p["can_commit"])
        with self.assertRaises(S.KernelError):
            imports.commit("plan", "plan.csv", self._b64(csv))

    def test_commit_and_undo_restores(self):
        key = ("SEC_GLA_Q1", "2031-01")
        before = S.db.read_df("SELECT COUNT(*) n FROM unit_plan_monthly WHERE unit_id=? AND ym=?", key)["n"].iloc[0]
        csv = "unit_id,ym,plan_new_wells,plan_new_oil_t,plan_measure_wells,plan_measure_inc_t,plan_old_oil_t\n" \
              "SEC_GLA_Q1,2031-01,2,300,1,80,2500\n"
        out = imports.commit("plan", "plan.csv", self._b64(csv))
        row = S.db.read_df("SELECT * FROM unit_plan_monthly WHERE unit_id=? AND ym=?", key)
        self.assertEqual(int(row["plan_new_wells"].iloc[0]), 2)
        self.assertEqual(row["data_source"].iloc[0], "IMPORT")
        h = imports.history()["batches"]
        self.assertTrue(next(b for b in h if b["batch_id"] == out["batch_id"])["can_undo"])
        imports.undo(out["batch_id"])
        after = S.db.read_df("SELECT COUNT(*) n FROM unit_plan_monthly WHERE unit_id=? AND ym=?", key)["n"].iloc[0]
        self.assertEqual(after, before)
        with self.assertRaises(S.KernelError):
            imports.undo(out["batch_id"])

    def test_template_has_chinese_headers(self):
        t = imports.template("asset_book").decode("utf-8-sig").splitlines()[0]
        self.assertIn("本期资本化投入", t)


if __name__ == "__main__":
    unittest.main()
