"""第三阶段：精确 TreeSHAP、序列模型、模型融合、带物理约束的递减模型、官方条文库。

这些测试不依赖训练好的模型包与数据库（条文库除外，它读仓库里的生成物），全部用小样本自造数据。
"""
import math
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.ensemble import blend, choose_weights, mix
from src.models.seq_model import SeqQuantileModel, _Net, pinball
from src.models.treeshap import TreeExplainer, brute_force_shap
from src.reserves import physics_dca as PD


class TestTreeSHAP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 5))
        X[rng.random(X.shape) < 0.06] = np.nan                   # 缺失值走训练时学到的方向
        y = np.nan_to_num(X[:, 0]) * 2 + np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) + rng.normal(size=300) * .1
        cls.X = X
        cls.m = HistGradientBoostingRegressor(max_iter=15, max_depth=4, random_state=0).fit(X, y)
        cls.ex = TreeExplainer(cls.m)

    def test_matches_brute_force_shapley(self):
        """与 Shapley 定义（穷举全部特征子集）逐位对拍。"""
        phi = self.ex.shap_values(self.X[:4])
        for i in range(4):
            bf, base = brute_force_shap(self.m, self.X[i])
            np.testing.assert_allclose(phi[i], bf, atol=1e-9)
            self.assertAlmostEqual(base, self.ex.expected_value, places=9)

    def test_additivity(self):
        phi = self.ex.shap_values(self.X[:50])
        np.testing.assert_allclose(self.ex.expected_value + phi.sum(1), self.m.predict(self.X[:50]), atol=1e-9)

    def test_unused_feature_gets_zero(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 3))
        m = HistGradientBoostingRegressor(max_iter=10, max_depth=3).fit(X, X[:, 0] * 3)
        phi = TreeExplainer(m).shap_values(X[:20])
        self.assertLess(np.abs(phi[:, 1:]).max(), 0.05 * np.abs(phi[:, 0]).max() + 1e-9)


class TestSequenceModel(unittest.TestCase):
    def test_backprop_matches_numeric_gradient(self):
        rng = np.random.default_rng(0)
        net = _Net(3, 4, 2, 5, 6, 3, (1, 2, 4), rng)
        S, X = rng.normal(size=(4, 3, 13)), rng.normal(size=(4, 4))
        Y, M = rng.normal(size=(4, 2)) * 2, np.ones((4, 2))
        M[0, 1] = 0

        def loss(S_=S):
            return pinball(net.forward(S_, X)[0], Y, M)[0]

        q, cache = net.forward(S, X)
        grads, dS = net.backward(pinball(q, Y, M)[1], cache, want_input=True)
        eps = 1e-6
        for k, p in net.p.items():
            idx = tuple(int(rng.integers(0, s)) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps
            a = loss()
            p[idx] = old - eps
            b = loss()
            p[idx] = old
            self.assertAlmostEqual((a - b) / (2 * eps), grads[k][idx], delta=1e-5 + 1e-4 * abs(grads[k][idx]), msg=k)
        idx = (1, 2, 5)
        S2 = S.copy(); S2[idx] += eps
        a = loss(S2); S2[idx] -= 2 * eps
        self.assertAlmostEqual((a - loss(S2)) / (2 * eps), dS[idx], delta=1e-5)

    def test_learns_signal_in_sequence_and_quantiles_ordered(self):
        """目标只由序列里的早期产量决定：模型必须学到，且 P10 ≤ P50 ≤ P90。"""
        rng = np.random.default_rng(3)
        n, T = 240, 30
        level = rng.uniform(1, 3, n)
        seq = np.zeros((n, 6, T))
        seq[:, 0, :] = level[:, None] + rng.normal(0, .05, (n, T))
        tab = pd.DataFrame({"noise": rng.normal(size=n)})
        Y = pd.DataFrame({"eur": np.expm1(level * 1.5 + rng.normal(0, .05, n))})
        m = SeqQuantileModel(["eur"], ["noise"], n_seeds=1, max_epochs=120, patience=20, hidden=8, dense=16)
        m.fit(seq[:200], tab.iloc[:200], Y.iloc[:200])
        pr = m.predict(seq[200:], tab.iloc[200:])["eur"]
        y = Y["eur"].to_numpy()[200:]
        self.assertTrue((pr["p10"] <= pr["p50"]).all() and (pr["p50"] <= pr["p90"]).all())
        corr = np.corrcoef(np.log1p(pr["p50"]), np.log1p(y))[0, 1]
        self.assertGreater(corr, 0.9)
        ig = m.temporal_attribution(seq[200], tab.iloc[[200]], "eur")
        self.assertEqual(ig["matrix"].shape, (6, T))
        self.assertLess(abs(ig["completeness_gap"]), 0.1 + 0.2 * abs(ig["delta_log"]))   # 积分梯度的完备性


class TestBlend(unittest.TestCase):
    def test_weight_selection_prefers_better_model(self):
        idx = pd.RangeIndex(100)
        y = np.linspace(10, 100, 100)
        good = pd.DataFrame({"p10": y - 5, "p50": y, "p90": y + 5}, index=idx)
        bad = pd.DataFrame({"p10": y * 0 + 20, "p50": y * 0 + 50, "p90": y * 0 + 80}, index=idx)
        sel = choose_weights(pd.DataFrame({"eur": y}), {"eur": bad}, {"eur": good}, ["eur"])
        self.assertEqual(sel["eur"]["weight"], 1.0)
        sel = choose_weights(pd.DataFrame({"eur": y}), {"eur": good}, {"eur": bad}, ["eur"])
        self.assertEqual(sel["eur"]["weight"], 0.0)

    def test_blend_keeps_quantiles_sorted(self):
        a = pd.DataFrame({"p10": [1.0], "p50": [2.0], "p90": [3.0]})
        b = pd.DataFrame({"p10": [5.0], "p50": [0.0], "p90": [1.0]})
        r = mix(a, b, 0.5).iloc[0]
        self.assertTrue(r["p10"] <= r["p50"] <= r["p90"])
        self.assertIs(blend({"x": a}, None, {"x": .5})["x"], a)


class TestPhysicsDCA(unittest.TestCase):
    def test_rate_continuous_and_bdf_b_bounded(self):
        f = PD.PhysicsFit(qi=20, di=0.3, t_elf=8.0, b_bdf=1.4, d_min=0.075 / 12)
        te = PD._elf_effective(f)
        left, right = PD.rate(np.array([te - 1e-6]), f)[0], PD.rate(np.array([te + 1e-6]), f)[0]
        self.assertAlmostEqual(left, right, delta=1e-4)
        d_l, d_r = PD.decline_rate(np.array([te - 1e-2]), f)[0], PD.decline_rate(np.array([te + 1e-2]), f)[0]
        self.assertAlmostEqual(d_l, d_r, delta=0.01)                         # 递减率在流态切换处连续
        late = PD.decline_rate(np.array([400.0]), f)[0]
        self.assertAlmostEqual(late, f.d_min, delta=2e-4)                     # 最终进入终端指数递减
        self.assertTrue(np.all(np.diff(PD.rate(np.linspace(0, 300, 600), f)) <= 1e-12))

    def test_fit_recovers_linear_flow_signature(self):
        truth = PD.PhysicsFit(qi=25, di=0.25, t_elf=14.0, b_bdf=0.6, d_min=0.075 / 12)
        t = np.arange(0.5, 48, 1.0)
        q = PD.rate(t, truth) * np.exp(np.random.default_rng(0).normal(0, .03, len(t)))
        f = PD.fit(t, q, q_econ=1.0, cum_to_now=0.0)
        self.assertLess(abs(f.t_elf - 14.0), 5.0)
        self.assertLessEqual(f.b_bdf, PD.B_BDF_MAX + 1e-9)
        self.assertGreater(f.r2, 0.95)

    def test_eur_cap_is_enforced(self):
        f0 = PD.PhysicsFit(qi=30, di=0.08, t_elf=2.0, b_bdf=0.9, d_min=0.075 / 12)
        t = np.arange(0.5, 10, 1.0)
        q = PD.rate(t, f0)
        free = PD.fit(t, q, q_econ=1.0, cum_to_now=5000.0)
        eur_free = 5000.0 + PD.remaining(free, 1.0, t.max())[0]
        cap = 0.6 * eur_free
        capped = PD.fit(t, q, q_econ=1.0, cum_to_now=5000.0, eur_cap_t=cap)
        eur_cap = 5000.0 + PD.remaining(capped, 1.0, t.max())[0]
        self.assertLess(eur_cap, eur_free)
        self.assertLess(eur_cap, cap * 1.08)
        self.assertTrue(capped.cap_binding)

    def test_flow_regime_diagnostic(self):
        t = np.arange(1, 60, 1.0)
        q = 30 / np.sqrt(t)
        d = PD.flow_regime(t, q)
        self.assertAlmostEqual(d["early_slope"], -0.5, delta=0.05)
        self.assertIn("线性流", d["regime_now"])

    def test_analog_prior_needs_enough_wells(self):
        fits = [PD.PhysicsFit(qi=10, di=0.1 + i * .01, t_elf=3 + i, b_bdf=0.7, d_min=0.006, n_points=40, converged=True)
                for i in range(6)]
        self.assertIsNone(PD.analog_prior(fits[:3]))
        pr = PD.analog_prior(fits)
        self.assertEqual(pr["n_analogs"], 6)
        self.assertGreaterEqual(pr["b_sd"], 0.12)


class TestOfficialCorpusParser(unittest.TestCase):
    def test_hierarchy_and_inline_markers(self):
        from src.agent.rag.build_corpus import parse_section, raw_dir
        sec = parse_section(raw_dir() / "210.4-10.xml", "Rule 4-10", "Regulation S-X")
        cites = [i["citation"] for i in sec["items"]]
        self.assertEqual(len(cites), len(set(cites)), "条款号不得重复")
        by = {i["citation"]: i for i in sec["items"]}
        self.assertIn("reasonable certainty", by["Rule 4-10(a)(24)"]["text"].lower())
        self.assertIn("90%", by["Rule 4-10(a)(24)"]["text"])
        self.assertEqual(by["Rule 4-10(a)(31)(ii)"]["parent"], "Rule 4-10(a)(31)")
        self.assertEqual(by["Rule 4-10(a)(17)"]["title"], "Possible reserves")


if __name__ == "__main__":
    unittest.main()


class TestDriftGuard(unittest.TestCase):
    def _frame(self, n, rng, shift=0.0, block="block_A"):
        X = pd.DataFrame({"a": rng.normal(shift, 1, n), "b": rng.normal(0, 1, n),
                          "block_A": 0.0, "block_B": 0.0})
        X[block] = 1.0
        return X

    def test_in_distribution_rarely_flagged_and_far_inputs_flagged(self):
        from src.models.drift import DriftGuard
        rng = np.random.default_rng(0)
        cols = ["a", "b", "block_A", "block_B"]
        tr = pd.concat([self._frame(200, rng), self._frame(200, rng, block="block_B")], ignore_index=True)
        cal = pd.concat([self._frame(50, rng), self._frame(50, rng, block="block_B")], ignore_index=True)
        g = DriftGuard().fit(tr, cols).calibrate(cal)
        same = pd.concat([self._frame(100, rng), self._frame(100, rng, block="block_B")], ignore_index=True)
        self.assertLess(g.assess(same)["flagged"].mean(), 0.08)
        far = self._frame(50, rng, shift=12.0)
        self.assertGreater(g.assess(far)["flagged"].mean(), 0.9)
        self.assertEqual(set(g.assess(far)["top_feature"]), {"a"})

    def test_unseen_category_is_flagged_even_when_numbers_look_normal(self):
        from src.models.drift import DriftGuard
        rng = np.random.default_rng(1)
        tr = self._frame(300, rng)                     # 训练只见过区块 A
        g = DriftGuard().fit(tr, ["a", "b", "block_A", "block_B"]).calibrate(self._frame(80, rng))
        r = g.assess(self._frame(40, rng, block="block_B"))
        self.assertTrue(r["novel_category"].all() and r["flagged"].all())


class TestCrossfitBlend(unittest.TestCase):
    def test_each_half_uses_weights_chosen_on_the_other_half(self):
        from src.models.ensemble import select_and_crossfit
        idx = pd.RangeIndex(40)
        y = np.linspace(10, 50, 40)
        good = pd.DataFrame({"p10": y - 2, "p50": y, "p90": y + 2}, index=idx)
        bad = pd.DataFrame({"p10": y * 0 + 10, "p50": y * 0 + 30, "p90": y * 0 + 50}, index=idx)
        # A 折（偶数行）序列模型准、B 折（奇数行）序列模型差：每折用对面选出的权重
        ps = good.copy()
        ps.iloc[1::2] = bad.iloc[1::2].to_numpy()
        pg = bad.copy()
        pg.iloc[1::2] = good.iloc[1::2].to_numpy()
        Y = pd.DataFrame({"eur": y}, index=idx)
        deployed, cross, folds = select_and_crossfit(Y, {"eur": pg}, {"eur": ps}, ["eur"], idx[::2], idx[1::2])
        self.assertEqual(folds["a"]["eur"], 0.0)       # A 折用 B 折选出的权重：B 折里序列模型差 → 0
        self.assertEqual(folds["b"]["eur"], 1.0)
        np.testing.assert_allclose(cross["eur"].loc[idx[::2], "p50"], pg.loc[idx[::2], "p50"])

    def test_flagged_rows_fall_back_to_gbdt(self):
        from src.models.ensemble import blend
        a = pd.DataFrame({"p10": [1.0, 1.0], "p50": [2.0, 2.0], "p90": [3.0, 3.0]})
        b = pd.DataFrame({"p10": [5.0, 5.0], "p50": [6.0, 6.0], "p90": [7.0, 7.0]})
        out = blend({"x": a}, {"x": b}, {"x": 1.0}, off=[True, False])["x"]
        self.assertEqual(out["p50"].tolist(), [2.0, 6.0])


class TestTreeSHAPAgainstShapPackage(unittest.TestCase):
    """第二条独立证据：与社区标准实现 shap.TreeExplainer 交叉比对（没装 shap 时跳过）。"""

    def test_matches_shap_package(self):
        try:
            import shap
        except Exception:
            self.skipTest("未安装 shap 包")
        rng = np.random.default_rng(7)
        X = rng.normal(size=(400, 6))
        y = X[:, 0] * 3 + X[:, 1] * X[:, 2] + np.sin(X[:, 3]) + rng.normal(size=400) * .1
        m = HistGradientBoostingRegressor(max_iter=40, max_depth=5, random_state=0).fit(X, y)
        ours = TreeExplainer(m)
        ref = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent")
        np.testing.assert_allclose(ours.shap_values(X[:30]), ref.shap_values(X[:30]), atol=1e-6)
        self.assertAlmostEqual(ours.expected_value, float(np.ravel(ref.expected_value)[0]), places=6)
