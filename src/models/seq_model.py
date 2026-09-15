"""早期序列模型：膨胀卷积 TCN + 分位数输出，纯 numpy 实现（前向、反向传播、Adam 全部手写）。

为什么要有它：梯度提升吃的是"前 90 天的统计量"，爬坡形态、停井扰动、压力与产量的先后关系
这些时序细节在统计量里被压扁了。序列模型直接读逐日曲线，两者的误差来源不同，融合后更稳。

为什么不用 torch：内网离线环境装不上；样本只有几百口井、序列 90 步，
三层卷积的 numpy 实现训练一次十几秒，完全够用，而且每一步都能在答辩时讲清楚。

结构：
    逐日序列 (N, C, T) ─ Conv(k=5, d=1) ─ ReLU ─ Conv(d=2)+残差 ─ Conv(d=4)+残差
                        └ 均值池化 ⊕ 最大池化 ⊕ 静态/完井特征 ─ Dense ─ ReLU ─ Dense
    每个目标输出 (mid, g_lo, g_hi)：P50 = mid，P10 = mid − softplus(g_lo)，P90 = mid + softplus(g_hi) —— 分位数永不交叉。
    目标取 log1p 后标准化，分位数损失（pinball），Adam + L2，按验证集早停，多个随机种子集成。
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
CHANNELS = ("log_oil", "on", "whp_rel", "water_cut", "log_gor", "log_cum")


def _conv_fwd(x: np.ndarray, W: np.ndarray, b: np.ndarray, d: int, k: int):
    N, C, T = x.shape
    total = d * (k - 1)
    left = total // 2
    xp = np.pad(x, ((0, 0), (0, 0), (left, total - left)))
    cols = np.stack([xp[:, :, m * d:m * d + T] for m in range(k)], axis=3)      # (N, C, T, k)
    cols = cols.transpose(0, 2, 1, 3).reshape(N, T, C * k)
    out = cols @ W.T + b                                                          # (N, T, H)
    return out.transpose(0, 2, 1), (cols, xp.shape, left)


def _conv_bwd(dout: np.ndarray, cache, W: np.ndarray, d: int, k: int):
    cols, xshape, left = cache
    N, C, _ = xshape
    T = dout.shape[2]
    g = dout.transpose(0, 2, 1)                                                  # (N, T, H)
    dW = np.einsum("nth,ntc->hc", g, cols, optimize=True)
    db = g.sum(axis=(0, 1))
    dcols = (g @ W).reshape(N, T, C, k).transpose(0, 2, 1, 3)
    dxp = np.zeros(xshape)
    for m in range(k):
        dxp[:, :, m * d:m * d + T] += dcols[..., m]
    return dxp[:, :, left:left + T], dW, db


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class _Net:
    def __init__(self, C: int, P: int, n_out: int, hidden: int, dense: int, k: int,
                 dilations: Sequence[int], rng: np.random.Generator):
        self.k, self.dil, self.n_out = k, tuple(dilations), n_out
        self.p: Dict[str, np.ndarray] = {}
        cin = C
        for i, _ in enumerate(self.dil):
            fan = cin * k
            self.p[f"W{i}"] = rng.normal(0.0, math.sqrt(2.0 / fan), (hidden, fan))
            self.p[f"b{i}"] = np.zeros(hidden)
            cin = hidden
        zin = 2 * hidden + P
        self.p["D1"] = rng.normal(0.0, math.sqrt(2.0 / zin), (dense, zin))
        self.p["c1"] = np.zeros(dense)
        self.p["D2"] = rng.normal(0.0, math.sqrt(1.0 / dense), (n_out * 3, dense))
        self.p["c2"] = np.zeros(n_out * 3)
        self.p["c2"][2::3] = 0.5                       # 初始区间宽度 softplus(0.5)≈1 个标准差
        self.p["c2"][1::3] = 0.5

    def forward(self, seq: np.ndarray, tab: np.ndarray):
        cache: Dict = {}
        h = seq
        for i, d in enumerate(self.dil):
            pre, cc = _conv_fwd(h, self.p[f"W{i}"], self.p[f"b{i}"], d, self.k)
            act = np.maximum(pre, 0.0)
            cache[i] = (cc, pre)
            h = act if i == 0 else h + act           # 首层通道数变化，不加残差
        T = h.shape[2]
        am = h.argmax(axis=2)
        z = np.concatenate([h.mean(axis=2), h.max(axis=2), tab], axis=1)
        a_pre = z @ self.p["D1"].T + self.p["c1"]
        a = np.maximum(a_pre, 0.0)
        out = (a @ self.p["D2"].T + self.p["c2"]).reshape(-1, self.n_out, 3)
        mid, gl, gh = out[..., 0], out[..., 1], out[..., 2]
        q = np.stack([mid - np.logaddexp(0.0, gl), mid, mid + np.logaddexp(0.0, gh)], axis=2)
        cache.update(h=h, am=am, z=z, a_pre=a_pre, a=a, gl=gl, gh=gh, T=T)
        return q, cache

    def backward(self, dq: np.ndarray, cache: Dict, want_input: bool = False):
        g: Dict[str, np.ndarray] = {}
        dmid = dq[..., 0] + dq[..., 1] + dq[..., 2]
        dgl = -dq[..., 0] * _sigmoid(cache["gl"])
        dgh = dq[..., 2] * _sigmoid(cache["gh"])
        dout = np.stack([dmid, dgl, dgh], axis=2).reshape(dq.shape[0], -1)
        g["D2"], g["c2"] = dout.T @ cache["a"], dout.sum(0)
        da_pre = (dout @ self.p["D2"]) * (cache["a_pre"] > 0)
        g["D1"], g["c1"] = da_pre.T @ cache["z"], da_pre.sum(0)
        dz = da_pre @ self.p["D1"]
        h = cache["h"]
        N, H, T = h.shape
        dh = np.repeat((dz[:, :H] / T)[:, :, None], T, axis=2)
        dh[np.arange(N)[:, None], np.arange(H)[None, :], cache["am"]] += dz[:, H:2 * H]
        for i in reversed(range(len(self.dil))):
            cc, pre = cache[i]
            dx, g[f"W{i}"], g[f"b{i}"] = _conv_bwd(dh * (pre > 0), cc, self.p[f"W{i}"], self.dil[i], self.k)
            dh = dx if i == 0 else dh + dx           # 残差支路的梯度直接相加
        return g, (dh if want_input else None)


def pinball(q: np.ndarray, y: np.ndarray, mask: np.ndarray):
    """分位数损失及其对 q 的梯度。q (N, n, 3)，y/mask (N, n)。"""
    taus = np.asarray(QUANTILES)
    e = y[..., None] - q
    m = mask[..., None]
    denom = max(float(mask.sum()) * len(taus), 1.0)
    loss = float((np.where(e >= 0, taus * e, (taus - 1.0) * e) * m).sum() / denom)
    dq = np.where(e >= 0, -taus, 1.0 - taus) * m / denom
    return loss, dq


class _Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr: float, wd: float):
        self.lr, self.wd, self.t = lr, wd, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, p in params.items():
            gk = grads[k] + (self.wd * p if k[0] in "WD" else 0.0)       # 只对权重做 L2
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)


@dataclass
class SeqQuantileModel:
    targets: List[str]
    tab_cols: List[str]
    hidden: int = 16
    dense: int = 32
    kernel: int = 5
    dilations: Sequence[int] = (1, 2, 4)
    n_seeds: int = 2
    max_epochs: int = 200
    patience: int = 25
    lr: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    val_frac: float = 0.15
    seed: int = 42
    nets: List[_Net] = field(default_factory=list)
    norm: Dict[str, np.ndarray] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)

    # ---------------- 数据准备 ----------------
    def _seq(self, seq: np.ndarray) -> np.ndarray:
        return (np.asarray(seq, float) - self.norm["seq_mu"][None, :, None]) / self.norm["seq_sd"][None, :, None]

    def _tab(self, tab: pd.DataFrame) -> np.ndarray:
        X = tab.reindex(columns=self.tab_cols).astype(float).to_numpy()
        X = (X - self.norm["tab_mu"]) / self.norm["tab_sd"]
        return np.nan_to_num(X, nan=0.0)

    def _y(self, Y: pd.DataFrame):
        raw = Y.reindex(columns=self.targets).astype(float).to_numpy()
        mask = np.isfinite(raw).astype(float)
        L = np.log1p(np.maximum(np.nan_to_num(raw, nan=0.0), 0.0))
        return L, mask

    # ---------------- 训练 ----------------
    def fit(self, seq: np.ndarray, tab: pd.DataFrame, Y: pd.DataFrame) -> "SeqQuantileModel":
        seq = np.asarray(seq, float)
        sd = seq.std(axis=(0, 2))
        self.norm["seq_mu"], self.norm["seq_sd"] = seq.mean(axis=(0, 2)), np.where(sd < 1e-6, 1.0, sd)
        T = tab.reindex(columns=self.tab_cols).astype(float).to_numpy()
        tmu, tsd = np.nanmean(T, axis=0), np.nanstd(T, axis=0)
        self.norm["tab_mu"] = np.nan_to_num(tmu, nan=0.0)
        self.norm["tab_sd"] = np.where(~np.isfinite(tsd) | (tsd < 1e-9), 1.0, tsd)
        L, mask = self._y(Y)
        ymu = np.array([L[mask[:, j] > 0, j].mean() if mask[:, j].any() else 0.0 for j in range(L.shape[1])])
        ysd = np.array([L[mask[:, j] > 0, j].std() if mask[:, j].sum() > 1 else 1.0 for j in range(L.shape[1])])
        self.norm["y_mu"], self.norm["y_sd"] = ymu, np.where(ysd < 1e-6, 1.0, ysd)
        Ys = (L - ymu) / self.norm["y_sd"] * mask

        S, X = self._seq(seq), self._tab(tab)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(S))
        n_val = max(int(len(S) * self.val_frac), 1)
        va, tr = order[:n_val], order[n_val:]
        self.nets, self.history = [], []
        for s in range(self.n_seeds):
            net = _Net(S.shape[1], X.shape[1], len(self.targets), self.hidden, self.dense, self.kernel,
                       self.dilations, np.random.default_rng([self.seed, s]))
            opt = _Adam(net.p, self.lr, self.weight_decay)
            best, best_p, bad = float("inf"), None, 0
            r = np.random.default_rng([self.seed, s, 7])
            for epoch in range(self.max_epochs):
                idx = r.permutation(tr)
                for i in range(0, len(idx), self.batch_size):
                    bi = idx[i:i + self.batch_size]
                    q, cache = net.forward(S[bi], X[bi])
                    _, dq = pinball(q, Ys[bi], mask[bi])
                    grads, _ = net.backward(dq, cache)
                    opt.step(net.p, grads)
                qv, _ = net.forward(S[va], X[va])
                lv, _ = pinball(qv, Ys[va], mask[va])
                if lv < best - 1e-5:
                    best, best_p, bad = lv, copy.deepcopy(net.p), 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            net.p = best_p if best_p is not None else net.p
            self.nets.append(net)
            self.history.append(dict(seed=s, epochs=epoch + 1, val_pinball=round(best, 5)))
        return self

    # ---------------- 预测 ----------------
    def _q_std(self, S: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.mean([net.forward(S, X)[0] for net in self.nets], axis=0)

    def predict(self, seq: np.ndarray, tab: pd.DataFrame, index=None) -> Dict[str, pd.DataFrame]:
        q = self._q_std(self._seq(seq), self._tab(tab))
        L = q * self.norm["y_sd"][None, :, None] + self.norm["y_mu"][None, :, None]
        vals = np.sort(np.maximum(np.expm1(L), 0.0), axis=2)
        idx = index if index is not None else tab.index
        return {t: pd.DataFrame(vals[:, j, :], index=idx, columns=["p10", "p50", "p90"])
                for j, t in enumerate(self.targets)}

    # ---------------- 时间归因 ----------------
    def temporal_attribution(self, seq_one: np.ndarray, tab_row: pd.DataFrame, target: str,
                             steps: int = 32) -> Dict:
        """积分梯度（Integrated Gradients）：基线 = 训练集平均日曲线（标准化空间的 0）。

        返回 log 空间的贡献：Σ 贡献 ≈ log1p(预测) − log1p(基线预测)，逐日与逐通道拆开。
        """
        S = self._seq(np.asarray(seq_one, float)[None])
        X = self._tab(tab_row)
        j = self.targets.index(target)
        base = np.zeros_like(S)
        total = np.zeros_like(S)
        for a in (np.arange(steps) + 0.5) / steps:
            xa = base + a * (S - base)
            for net in self.nets:
                q, cache = net.forward(xa, X)
                dq = np.zeros_like(q)
                dq[:, j, 1] = 1.0
                _, dS = net.backward(dq, cache, want_input=True)
                total += dS
        ig = (S - base) * total / (steps * len(self.nets)) * self.norm["y_sd"][j]
        f_x = float(self._q_std(S, X)[0, j, 1]) * self.norm["y_sd"][j]
        f_b = float(self._q_std(base, X)[0, j, 1]) * self.norm["y_sd"][j]
        return dict(matrix=ig[0], per_day=ig[0].sum(axis=0), per_channel=dict(zip(CHANNELS, ig[0].sum(axis=1))),
                    delta_log=f_x - f_b, completeness_gap=float(ig.sum() - (f_x - f_b)))


def sequence_tensor(prod: pd.DataFrame, well_ids: Sequence[str], obs_days: int) -> np.ndarray:
    """逐日序列张量 (N, C, T)：只用投产后前 obs_days 天，井号顺序与 well_ids 一致。"""
    out = np.zeros((len(well_ids), len(CHANNELS), obs_days))
    p = prod[prod["day_index"] <= obs_days]
    groups = {w: g for w, g in p.groupby("well_id")}
    for n, wid in enumerate(well_ids):
        g = groups.get(wid)
        if g is None or g.empty:
            continue
        g = g.sort_values("day_index")
        idx = g["day_index"].to_numpy(int) - 1
        on = (g["hours_on"].to_numpy(float) > 0).astype(float)
        oil = np.where(on > 0, np.nan_to_num(g["oil_t"].to_numpy(float)), 0.0)
        whp = pd.Series(g["whp_mpa"].to_numpy(float)).ffill().bfill().to_numpy()
        valid = whp[np.isfinite(whp)]
        p_ref = float(np.median(valid[:5])) if len(valid) else 1.0
        wc = pd.Series(g["water_cut"].to_numpy(float)).ffill().fillna(0.0).to_numpy() / 100.0
        gor = pd.Series(g["gor"].to_numpy(float)).ffill().fillna(0.0).to_numpy()
        out[n, 0, idx] = np.log1p(oil)
        out[n, 1, idx] = on
        out[n, 2, idx] = np.nan_to_num(whp / max(p_ref, 1e-6), nan=0.0) * on
        out[n, 3, idx] = wc
        out[n, 4, idx] = np.log1p(np.maximum(gor, 0.0)) / 5.0
        out[n, 5, idx] = np.log1p(np.cumsum(oil)) / 10.0
        out[n, 5] = np.maximum.accumulate(out[n, 5])             # 缺记录的日子累产沿用前值
    return out
