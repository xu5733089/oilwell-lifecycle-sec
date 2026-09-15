"""精确 TreeSHAP（路径依赖口径），直接读取 sklearn HistGradientBoosting 的树结构。

内网装不上 shap 包，这里自己实现，并用"穷举子集的 Shapley 定义"逐位对拍（tests/test_models_phase3.py）。

算法：一棵树的输出 = Σ_叶子 v · Π_{路径上的特征 j} [x 是否沿该特征的全部分裂走到这片叶子]。
特征 j "缺席"时按训练样本覆盖比例 z_j 走各分支（路径依赖口径），"在场"时取 o_j ∈ {0, 1}。
于是每片叶子对特征 k 的 Shapley 值
    φ_k += v · (o_k − z_k) · Σ_s c_s · s!(n−s−1)!/n!
其中 c_s 是 Π_{j≠k}(z_j + o_j·t) 展开后 t^s 的系数，n 为该叶子路径上不同特征的个数。
同一特征在路径上多次分裂时 z 相乘、条件逐条检查。复杂度 O(叶子数 · n²)，n 即树深量级。

可加性：expected_value + Σ_k φ_k = 模型原始输出，逐样本成立（测试校验）。
装了 numba 自动编译加速（单井毫秒级）；没装按纯 Python 运行，结果完全一致。
"""
from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Dict, List, Tuple

import numpy as np

try:                                        # pragma: no cover - 取决于环境
    from numba import njit
    HAS_NUMBA = True
except Exception:                           # pragma: no cover
    HAS_NUMBA = False


def _core(X, leaf_value, leaf_start, uf_feature, uf_z, uf_cond_start, cond_thr, cond_left, cond_mleft,
          W, n_features):
    N = X.shape[0]
    L = leaf_value.shape[0]
    maxn = W.shape[1]
    phi = np.zeros((N, n_features))
    o = np.zeros(maxn)
    coef = np.zeros(maxn + 1)
    for i in range(N):
        for leaf in range(L):
            a = leaf_start[leaf]
            n = leaf_start[leaf + 1] - a
            if n == 0:
                continue
            v = leaf_value[leaf]
            for k in range(n):
                u = a + k
                xv = X[i, uf_feature[u]]
                ok = 1.0
                for c in range(uf_cond_start[u], uf_cond_start[u + 1]):
                    if xv != xv:                        # NaN：按训练时的缺失值方向
                        go_left = cond_mleft[c] == 1
                    else:
                        go_left = xv <= cond_thr[c]
                    if go_left != (cond_left[c] == 1):
                        ok = 0.0
                        break
                o[k] = ok
            for k in range(n):
                for s in range(n + 1):
                    coef[s] = 0.0
                coef[0] = 1.0
                deg = 0
                for j in range(n):
                    if j == k:
                        continue
                    zj = uf_z[a + j]
                    oj = o[j]
                    for s in range(deg + 1, 0, -1):
                        coef[s] = coef[s] * zj + coef[s - 1] * oj
                    coef[0] = coef[0] * zj
                    deg += 1
                tot = 0.0
                for s in range(n):
                    tot += coef[s] * W[n, s]
                phi[i, uf_feature[a + k]] += v * (o[k] - uf_z[a + k]) * tot
    return phi


_core_fast = njit(cache=False)(_core) if HAS_NUMBA else _core


def _weights(maxn: int) -> np.ndarray:
    W = np.zeros((maxn + 1, max(maxn, 1)))
    for n in range(1, maxn + 1):
        for s in range(n):
            W[n, s] = factorial(s) * factorial(n - s - 1) / factorial(n)
    return W


def _trees(est):
    if not hasattr(est, "_predictors"):
        raise TypeError(f"TreeExplainer 只支持 HistGradientBoosting，收到 {type(est).__name__}")
    if getattr(est, "n_trees_per_iteration_", 1) != 1:
        raise TypeError("只支持单输出回归模型")
    return [preds[0] for preds in est._predictors]


def _flatten(est) -> Tuple[Dict[str, np.ndarray], float]:
    leaf_value: List[float] = []
    leaf_start = [0]
    uf_feature: List[int] = []
    uf_z: List[float] = []
    uf_cond_start = [0]
    cond_thr: List[float] = []
    cond_left: List[int] = []
    cond_mleft: List[int] = []
    expected = float(np.asarray(est._baseline_prediction).ravel()[0])
    for tree in _trees(est):
        nodes = tree.nodes
        if nodes["is_categorical"].any():
            raise TypeError("暂不支持类别型分裂")
        root = float(nodes[0]["count"]) or 1.0
        stack = [(0, {})]
        while stack:
            j, path = stack.pop()
            nd = nodes[j]
            if nd["is_leaf"]:
                v = float(nd["value"])
                expected += v * float(nd["count"]) / root
                for f, (z, conds) in path.items():
                    uf_feature.append(int(f))
                    uf_z.append(z)
                    for thr, left, ml in conds:
                        cond_thr.append(thr)
                        cond_left.append(left)
                        cond_mleft.append(ml)
                    uf_cond_start.append(len(cond_thr))
                leaf_value.append(v)
                leaf_start.append(len(uf_feature))
                continue
            f = int(nd["feature_idx"])
            thr, ml, cnt = float(nd["num_threshold"]), int(nd["missing_go_to_left"]), float(nd["count"]) or 1.0
            for child, left in ((int(nd["left"]), 1), (int(nd["right"]), 0)):
                ratio = float(nodes[child]["count"]) / cnt
                p2 = {k: (z, list(c)) for k, (z, c) in path.items()}
                z0, c0 = p2.get(f, (1.0, []))
                p2[f] = (z0 * ratio, c0 + [(thr, left, ml)])
                stack.append((child, p2))
    arr = dict(leaf_value=np.asarray(leaf_value, float), leaf_start=np.asarray(leaf_start, np.int64),
               uf_feature=np.asarray(uf_feature, np.int64), uf_z=np.asarray(uf_z, float),
               uf_cond_start=np.asarray(uf_cond_start, np.int64), cond_thr=np.asarray(cond_thr, float),
               cond_left=np.asarray(cond_left, np.int64), cond_mleft=np.asarray(cond_mleft, np.int64))
    return arr, expected


class TreeExplainer:
    """用法：ex = TreeExplainer(est); phi = ex.shap_values(X)；ex.expected_value + phi.sum(1) == est.predict(X)。"""

    def __init__(self, est):
        self.n_features = int(est.n_features_in_)
        self._arr, self.expected_value = _flatten(est)
        n_per_leaf = np.diff(self._arr["leaf_start"])
        self.max_path_features = int(n_per_leaf.max()) if len(n_per_leaf) else 0
        self._W = _weights(max(self.max_path_features, 1))

    def shap_values(self, X) -> np.ndarray:
        X = np.ascontiguousarray(np.asarray(X, dtype=float))
        if X.ndim == 1:
            X = X[None, :]
        a = self._arr
        return _core_fast(X, a["leaf_value"], a["leaf_start"], a["uf_feature"], a["uf_z"], a["uf_cond_start"],
                          a["cond_thr"], a["cond_left"], a["cond_mleft"], self._W, self.n_features)


# --------------------------------------------------------------------------- #
# 对拍用：按 Shapley 定义穷举子集（只适合特征很少的小模型）
def _cond_expectation(nodes, x: np.ndarray, present: set, j: int = 0) -> float:
    nd = nodes[j]
    if nd["is_leaf"]:
        return float(nd["value"])
    f = int(nd["feature_idx"])
    left, right = int(nd["left"]), int(nd["right"])
    if f in present:
        xv = x[f]
        go_left = bool(nd["missing_go_to_left"]) if np.isnan(xv) else xv <= nd["num_threshold"]
        return _cond_expectation(nodes, x, present, left if go_left else right)
    cnt = float(nd["count"]) or 1.0
    return (float(nodes[left]["count"]) / cnt * _cond_expectation(nodes, x, present, left)
            + float(nodes[right]["count"]) / cnt * _cond_expectation(nodes, x, present, right))


def brute_force_shap(est, x: np.ndarray) -> Tuple[np.ndarray, float]:
    x = np.asarray(x, float)
    trees = _trees(est)
    F = int(est.n_features_in_)
    base = float(np.asarray(est._baseline_prediction).ravel()[0])

    def value(S: set) -> float:
        return base + sum(_cond_expectation(t.nodes, x, S) for t in trees)

    phi = np.zeros(F)
    for k in range(F):
        others = [f for f in range(F) if f != k]
        for s in range(F):
            w = factorial(s) * factorial(F - s - 1) / factorial(F)
            for S in combinations(others, s):
                S = set(S)
                phi[k] += w * (value(S | {k}) - value(S))
    return phi, value(set())
