"""算法内核服务层（方案 §2.1 的 L2）。

**全部数值的唯一产地。** 智能体、HTTP 接口、CLI、报告生成都只能调这里，
任何一层都不许自己算数。每个返回体都带 model_version / label_def_version /
data_source / trace_id 四个追溯字段。
"""
from __future__ import annotations

import functools
import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import db, trace
from ..config import config, indicators as indicator_spec, path, price_decks
from ..features.build import align, build, encode, feature_columns
from ..models import registry
from ..models.analog_retrieval import _resample_curve
from ..models.attribution import SHAP_METHOD, local_shap
from ..models.ensemble import blend
from ..models.seq_model import CHANNELS, sequence_tensor
from ..reserves import crosscheck, dca, physics_dca, volumetric, workload
from ..sec import checklist as sec_checklist
from ..sec import classify as sec_classify
from ..sec import composition as sec_composition
from ..sec import economics
from ..sec import finance
from ..sec import indicators as sec_indicators
from ..sec import reconcile as sec_reconcile

DEFAULT_PRICE_DECK = "deck_2026_12"
MIN_POST_PEAK_DAYS = 180   # 递减分析所需的最短峰后历史
N_BOOTSTRAP = 80          # 自助法次数：80 次足够定 P10/P90，再多只是烧时间

_CACHED: List = []


def cached_service(fn):
    """服务结果缓存。

    内核是**确定性的纯函数**（同一份库数据 + 同一组参数 -> 同一个结果），
    所以缓存是安全的。但每次返回都换一个新的 trace_id：
    缓存的是计算，不是这一次调用的身份 —— 审计留痕必须逐次唯一。

    库数据变了（重新 init / train）调用 reset_cache() 清空。
    """
    store = {}

    @functools.wraps(fn)
    def wrapper(*args, trace_id=None, **kwargs):
        key = json.dumps([fn.__name__, args, sorted(kwargs.items())],
                         default=str, ensure_ascii=False)
        if key not in store:
            store[key] = fn(*args, trace_id=trace_id, **kwargs)
        out = dict(store[key])
        out["trace_id"] = trace_id or trace.new_trace_id()
        return out

    wrapper.cache_clear = store.clear
    _CACHED.append(wrapper)
    return wrapper


class KernelError(RuntimeError):
    """内核可预期的失败（井不存在、历史不足等）。智能体须如实转述，不得臆造。"""


@functools.lru_cache(maxsize=1)
def _bundle():
    return registry.load()


def reset_cache() -> None:
    _bundle.cache_clear()
    _tables.cache_clear()
    _rf_reference.cache_clear()
    _well_agg.cache_clear()
    for f in (_units, _monthly, _events, _codes, _new_wells, _new_well_model_estimates, _evaluate,
              _eval_fingerprint, _locations, _producing_xy, _first_prod_map, _asset_book,
              _unit_depletion_chain, _indicator_periods, _physics_library, _retriever):
        f.cache_clear()
    _FIT_CACHE.clear()
    for w in _CACHED:
        w.cache_clear()


@functools.lru_cache(maxsize=1)
def _tables():
    # 标签表按 (well_id, label_def_version) 存多版口径；这里只取当前模型训练所用的版本，
    # 否则一口井会出现多行，井表重复、标签取到旧口径
    labels = db.read_df("SELECT * FROM lifecycle_label")
    if not labels.empty:
        ver = _bundle()["meta"].get("label_def_version")
        if ver not in set(labels["label_def_version"]):
            ver = sorted(labels["label_def_version"].unique())[-1]
        labels = labels[labels["label_def_version"] == ver].reset_index(drop=True)
    return dict(master=db.read_df("SELECT * FROM well_master"),
                static=db.read_df("SELECT * FROM geo_static"),
                labels=labels)


def _resolve(well_code: str) -> pd.Series:
    t = _tables()["master"]
    hit = t[(t["well_id"] == well_code) | (t["well_code_anon"] == well_code)]
    if hit.empty:
        raise KernelError(f"井 {well_code!r} 不存在于井表中")
    return hit.iloc[0]


def _master_df(well_id: str) -> pd.DataFrame:
    """取单井 master 时必须走切片，不能用 Series.to_frame().T ——
    后者会把所有列变成 object dtype，下游数值运算全炸。"""
    t = _tables()["master"]
    return t[t["well_id"] == well_id].reset_index(drop=True)


def _prod(well_id: str) -> pd.DataFrame:
    p = db.read_df("SELECT * FROM prod_daily WHERE well_id = ? ORDER BY day_index", (well_id,))
    if p.empty:
        raise KernelError(f"井 {well_id} 无生产数据")
    return p


def _env(trace_id: Optional[str] = None) -> Dict[str, str]:
    b = _bundle()
    return trace.envelope(data_source=b["meta"].get("data_source", "SYNTHETIC"),
                          model_version=b["meta"]["model_version"], trace_id=trace_id)


# --------------------------------------------------------------------------- #
@cached_service
def query_well(well_code: str, fields: Optional[List[str]] = None,
               trace_id: Optional[str] = None) -> Dict:
    w = _resolve(well_code)
    st = _tables()["static"]
    st = st[st["well_id"] == w["well_id"]]
    p = _prod(w["well_id"])
    on = p[p["hours_on"] > 0]
    window = 14          # 近期均值窗口。作为字段返回，文本里出现的每个数字都要有出处
    info = dict(
        rate_window_days=window,
        well_code=w["well_code_anon"], well_id=w["well_id"], block=w["block"],
        layer=w["layer"], well_type=w["well_type"], status=w["status"],
        first_prod_date=w["first_prod_date"], lateral_length=_r(w["lateral_length"]),
        stage_count=int(w["stage_count"] or 0), tvd=_r(w["tvd"]),
        prod_days=int(len(on)), history_days=int(p["day_index"].max()),
        cum_oil_t=_r(p["oil_t"].sum()),
        current_rate_t_per_d=_r(on["oil_t"].tail(window).mean()),
        latest_whp_mpa=_r(on["whp_mpa"].tail(window).mean()),
        water_cut_pct=_r(on["water_cut"].tail(window).mean()),
    )
    if not st.empty:
        s = st.iloc[0]
        info["static"] = {k: _r(s[k]) for k in
                          ["porosity_pct", "so_pct", "net_pay_m", "perm_md", "toc_pct",
                           "sweet_spot_idx", "pressure_coef"]}
    if fields:
        info = {k: v for k, v in info.items() if k in set(fields) | {"well_code", "well_id"}}
    return dict(**_env(trace_id), **info)


def _features_for(well_code: str, obs_days: Optional[int] = None):
    b = _bundle()
    obs_days = obs_days or b["obs_days"]
    w = _resolve(well_code)
    p = _prod(w["well_id"])
    if p["day_index"].max() < obs_days:
        raise KernelError(f"井 {w['well_code_anon']} 仅有 {int(p['day_index'].max())} 天历史，"
                          f"不足观测窗 {obs_days} 天")
    t = _tables()
    st = t["static"][t["static"]["well_id"] == w["well_id"]]
    X = encode(build(p, _master_df(w["well_id"]), st, obs_days,
                     ref_labels=b["train_labels"], spatial_master=t["master"]))
    return w, p, align(X, b["feature_cols"]), obs_days


def _family_predictions(b: Dict, X: pd.DataFrame, p: pd.DataFrame, well_id: str, obs_days: int):
    """线上预测 = 梯度提升 与 序列模型 按目标加权融合（权重训练时在校准集上选出）。
    输入超出序列模型训练分布（漂移守卫判定）时该井序列权重置 0；旧版模型包没有序列模型时退回纯梯度提升。"""
    pg = b["model"].predict(X)
    sm = b.get("seq_model")
    if sm is None:
        return pg, None
    ps = sm.predict(sequence_tensor(p, [well_id], obs_days), X, index=X.index)
    guard = b.get("drift_guard")
    drift = None
    if guard is not None:
        d = guard.assess(X).iloc[0]
        drift = dict(flagged=bool(d["flagged"]), score=_r(d["score"], 3), threshold=_r(guard.threshold, 3),
                     top_feature=d["top_feature"], novel_category=bool(d["novel_category"]))
    off = np.array([bool(drift and drift["flagged"])])
    return blend(pg, ps, b.get("blend_weights", {}), off=off), dict(gbdt=pg, seq=ps, drift=drift)


@cached_service
def predict_lifecycle(well_code: str, obs_days: Optional[int] = None,
                      targets: Optional[List[str]] = None,
                      trace_id: Optional[str] = None) -> Dict:
    b = _bundle()
    w, p, X, obs_days = _features_for(well_code, obs_days)
    preds, fam = _family_predictions(b, X, p, w["well_id"], obs_days)
    drifted = bool(fam and fam.get("drift") and fam["drift"]["flagged"])
    # 漂移井的预测已退回梯度提升，区间同样用梯度提升自己的保形修正量
    conf = b.get("conformal_family", {}).get("gbdt", b["conformal"]) if drifted else b["conformal"]
    out: Dict[str, Dict] = {}
    units = dict(t_oil_break="d", t_peak="d", q_peak="t/d", p_peak="MPa", eur="t")
    want = set(targets or preds.keys())
    cum = float(p["oil_t"].sum())
    for tgt, dfp in preds.items():
        if tgt not in want:
            continue
        lo, hi = conf.apply(tgt, dfp["p10"].to_numpy(float), dfp["p90"].to_numpy(float))
        q = [float(lo[0]), float(dfp["p50"].iloc[0]), float(hi[0])]
        # 物理下界：EUR 不可能低于已累产，时间/产量/压力不可能为负。
        # 保形区间是对称加宽的，老井上会把下界推到负数 —— 一眼假的数不许出门。
        floor = cum if tgt == "eur" else 0.0
        floored = min(q) < floor
        q = [max(v, floor) for v in q]
        out[tgt] = dict(p10=_r(q[0]), p50=_r(q[1]), p90=_r(q[2]), unit=units.get(tgt, ""))
        if floored:
            out[tgt]["floored_at"] = _r(floor)

    families = None
    if fam is not None:
        families = {}
        cf = b.get("conformal_family", {})
        for tgt in out:
            row = dict(seq_weight=0.0 if drifted else _r(b.get("blend_weights", {}).get(tgt, 0.0), 2),
                       blend=[out[tgt]["p10"], out[tgt]["p50"], out[tgt]["p90"]])
            for name in ("gbdt", "seq"):
                dfp = fam[name][tgt]
                lo, hi = (cf[name].apply(tgt, dfp["p10"].to_numpy(float), dfp["p90"].to_numpy(float))
                          if name in cf else (dfp["p10"].to_numpy(float), dfp["p90"].to_numpy(float)))
                floor = cum if tgt == "eur" else 0.0
                row[name] = [_r(max(float(lo[0]), floor)), _r(max(float(dfp["p50"].iloc[0]), floor)),
                             _r(max(float(hi[0]), floor))]
            families[tgt] = row

    curve = None
    if {"q_peak", "t_peak", "eur"} <= set(out):
        curve = dca.synthesize_curve(out["q_peak"]["p50"], out["t_peak"]["p50"],
                                     out["eur"]["p50"])

    top = []
    if "t_peak" in b["model"].models:
        top = local_shap(b["model"], X.iloc[0], "t_peak", 0.5, top_k=6)["contributions"]

    return dict(**_env(trace_id), well_code=w["well_code_anon"], obs_days=obs_days,
                cum_to_date_t=_r(cum), results=out, curve=curve, model_families=families,
                input_drift=fam.get("drift") if fam else None,
                explain=dict(attribution_target="t_peak", top_features=top,
                             method=SHAP_METHOD + "；解释对象为梯度提升 P50 分量",
                             calibration=dict(alpha=conf.alpha,
                                              delta={k: _r(v) for k, v in conf.delta.items()},
                                              test_coverage=b.get("coverage", {}))),
                note="P10/P50/P90 为分位数本身（p10 为数值小者）；储量口径下 P90 为低估计。")


@cached_service
def explain_lifecycle(well_code: str, target: str = "eur", trace_id: Optional[str] = None) -> Dict:
    """单井可解释性明细（界面用）：各目标的 TreeSHAP 瀑布 + 序列模型的逐日积分梯度热力条。"""
    b = _bundle()
    w, p, X, obs_days = _features_for(well_code)
    shap = {t: local_shap(b["model"], X.iloc[0], t, 0.5, top_k=8) for t in b["model"].models}
    temporal = None
    sm = b.get("seq_model")
    guard = b.get("drift_guard")
    drifted = bool(guard is not None and guard.assess(X)["flagged"].iloc[0])
    if sm is not None and target in sm.targets:
        seq = sequence_tensor(p, [w["well_id"]], obs_days)
        ig = sm.temporal_attribution(seq[0], X.iloc[[0]], target)
        early = p[p["day_index"] <= obs_days].sort_values("day_index")
        temporal = dict(
            target=target, obs_days=obs_days, drift_flagged=drifted,
            seq_weight=0.0 if drifted else _r(b.get("blend_weights", {}).get(target, 0.0), 2),
            channels=list(CHANNELS),
            matrix=[[round(float(v), 5) for v in row] for row in ig["matrix"]],
            per_day=[round(float(v), 5) for v in ig["per_day"]],
            per_channel={k: round(float(v), 5) for k, v in ig["per_channel"].items()},
            delta_log=round(float(ig["delta_log"]), 5), completeness_gap=round(float(ig["completeness_gap"]), 5),
            daily_oil=[[int(d), _r(o, 2)] for d, o in zip(early["day_index"], early["oil_t"])],
            method="积分梯度（Integrated Gradients，32 步，基线 = 训练集平均日曲线）；贡献为对数空间，"
                   "正值表示该日该通道把预测推高")
    return dict(**_env(trace_id), well_code=w["well_code_anon"], shap=shap, temporal=temporal)


@cached_service
def model_global_shap(target: str = "eur", trace_id: Optional[str] = None) -> Dict:
    b = _bundle()
    g = b.get("global_shap") or {}
    if target not in g:
        raise KernelError(f"模型包中没有目标 {target!r} 的全局 SHAP，可选：{'、'.join(g) or '无（请重新训练）'}")
    return dict(**_env(trace_id), target=target, targets=list(g), **g[target],
                blend_weight_seq=_r(b.get("blend_weights", {}).get(target, 0.0), 2))


@cached_service
def find_analog_wells(well_code: str, top_k: int = 5,
                      trace_id: Optional[str] = None) -> Dict:
    b = _bundle()
    w = _resolve(well_code)
    p = _prod(w["well_id"])
    t = _tables()
    st = t["static"][t["static"]["well_id"] == w["well_id"]]
    X = align(encode(build(p, _master_df(w["well_id"]), st, b["obs_days"],
                           ref_labels=b["train_labels"], spatial_master=t["master"])),
              b["feature_cols"])
    curve = _resample_curve(p, b["obs_days"])
    hits = b["analog"].query(X.iloc[0], curve, top_k=top_k, exclude=w["well_id"])
    code = dict(zip(t["master"]["well_id"], t["master"]["well_code_anon"]))
    for h in hits:
        h["well_code"] = code.get(h["well_id"], h["well_id"])
    return dict(**_env(trace_id), well_code=w["well_code_anon"], analogs=hits,
                method="早期曲线 DTW + 静态/完井特征距离 的混合相似度")


def _post_peak(w: pd.Series, p: pd.DataFrame, on: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    # 递减分析的前提是**已过峰**。把上升段当递减段拟合会外推出荒谬的 EUR
    # （见 scripts/_patch_dca_guard.py 的由来）。没过峰就明确拒绝，不给数。
    lab = _tables()["labels"]
    lab = lab[lab["well_id"] == w["well_id"]]
    if lab.empty or pd.isna(lab["t_peak"].iloc[0]):
        raise KernelError(f"井 {w['well_code_anon']} 尚未确定达峰点，递减分析不适用。"
                          "可先看全生命周期预测给出的 EUR 区间。")
    if lab["label_quality"].iloc[0] != "ok":
        raise KernelError(
            f"井 {w['well_code_anon']} 的达峰标签质量为 "
            f"{lab['label_quality'].iloc[0]}（{lab['reject_reason'].iloc[0] or '峰后下降未确认'}），"
            "递减分析不适用 —— 该井很可能尚未真正达峰。")
    t_peak = float(lab["t_peak"].iloc[0])
    post = on[on["day_index"] >= t_peak]
    post_days = int(post["day_index"].max() - t_peak) if len(post) else 0
    if len(post) < 60 or post_days < MIN_POST_PEAK_DAYS:
        raise KernelError(
            f"井 {w['well_code_anon']} 峰后历史仅 {post_days} 天（有效 {len(post)} 点），"
            f"不足递减分析所需的 {MIN_POST_PEAK_DAYS} 天。储量结论需等历史积累或改用类比法。")

    return t_peak, post


@cached_service
def fit_dca(well_code: str, model: str = "auto", d_min_year: float = 0.075,
            price_deck_id: str = DEFAULT_PRICE_DECK,
            trace_id: Optional[str] = None) -> Dict:
    w = _resolve(well_code)
    p = _prod(w["well_id"])
    on = p[(p["hours_on"] > 0) & (p["oil_t"] > 0)]
    if len(on) < 90:
        raise KernelError(f"井 {w['well_code_anon']} 有效生产不足 90 天，无法可靠拟合递减曲线")

    t_peak, post = _post_peak(w, p, on)
    tm = post["day_index"].to_numpy(float) / 30.4
    q = post["oil_t"].to_numpy(float)
    f = dca.fit_best(tm, q, d_min_year=d_min_year) if model == "auto" else \
        dca.fit(tm, q, model=model, d_min_year=d_min_year)

    econ = economics.economic_limit_rate(price_deck_id)
    cum = float(p["oil_t"].sum())
    band = dca.eur_from_history(tm, q, f, econ["q_econ"], cum_to_date=cum,
                                n_boot=N_BOOTSTRAP)

    return dict(**_env(trace_id), well_code=w["well_code_anon"],
                fit=dict(model=f.model, qi=_r(f.qi), di_per_month=_r(f.di), b=_r(f.b),
                         d_min_per_month=_r(f.d_min), r2=_r(f.r2), rmse=_r(f.rmse),
                         n_points=f.n_points),
                eur=dict(p10=_r(band["p10"]), p50=_r(band["p50"]), p90=_r(band["p90"]),
                         low_estimate=_r(band["low_estimate"]),
                         high_estimate=_r(band["high_estimate"]),
                         unit="t", n_bootstrap=band["n_boot"],
                         convention=dca.CONVENTION),
                cum_to_date_t=_r(cum),
                remaining_t=dict(p50=_r(band["remaining_p50"]), unit="t",
                                 note="EUR = 已累产（确定） + 剩余可采（不确定）"),
                economics=dict(q_econ_t_per_d=_r(econ["q_econ"]),
                               t_econ_month=_r(band["t_econ_month"]),
                               t_econ_capped=band["t_econ_capped"],
                               t_econ_note=("产量在 50 年评估期内始终高于经济极限，"
                                            "该时刻为积分上限而非真实经济极限时刻"
                                            if band["t_econ_capped"] else None),
                               t_now_month=_r(band["t_now_month"]),
                               avg_12m_price_usd_bbl=econ["avg_12m_price_usd_bbl"],
                               price_deck_id=price_deck_id),
                guard="b>1 时强制要求终端递减率，否则 Arps 积分不收敛（EUR 发散）")


@functools.lru_cache(maxsize=1)
def _physics_library() -> Dict[str, Dict[str, "physics_dca.PhysicsFit"]]:
    """各区块成熟井（峰后开井日均 ≥ 24 个月）的物理约束拟合 —— 类比先验的来源。一次算好全区缓存。"""
    t = _tables()
    ok = (t["labels"][t["labels"]["label_quality"] == "ok"][["well_id", "t_peak"]]
          .merge(t["master"][["well_id", "block"]], on="well_id"))
    prod = db.read_df("SELECT well_id, day_index, oil_t, hours_on FROM prod_daily")
    groups = dict(tuple(prod.groupby("well_id")))
    q_econ = economics.economic_limit_rate(DEFAULT_PRICE_DECK)["q_econ"]
    lib: Dict[str, Dict] = {}
    for _, r in ok.iterrows():
        g = groups.get(r["well_id"])
        if g is None:
            continue
        post = g[g["day_index"] >= r["t_peak"]]
        mo, ra = physics_dca.monthly_series(post["day_index"].to_numpy() - r["t_peak"] + 1,
                                            post["oil_t"].to_numpy(), post["hours_on"].to_numpy())
        if len(mo) >= 24:
            lib.setdefault(r["block"], {})[r["well_id"]] = physics_dca.fit(mo - mo[0], ra, q_econ=q_econ, cum_to_now=0.0)
    return lib


@cached_service
def dca_physics(well_code: str, price_deck_id: str = DEFAULT_PRICE_DECK,
                trace_id: Optional[str] = None) -> Dict:
    """带物理约束的递减复核 + 流态诊断（界面"递减诊断"用）。官方储量口径仍是 fit_dca 的经验 Arps。"""
    pcfg = config().get("physics_dca", {})
    w = _resolve(well_code)
    p = _prod(w["well_id"])
    on = p[(p["hours_on"] > 0) & (p["oil_t"] > 0)]
    if len(on) < 90:
        raise KernelError(f"井 {w['well_code_anon']} 有效生产不足 90 天，无法做递减诊断")
    t_peak, post = _post_peak(w, p, on)
    econ = economics.economic_limit_rate(price_deck_id)
    q_econ = float(econ["q_econ"])
    cum = float(p["oil_t"].sum())

    # 经验 Arps：与 fit_dca 同一份数据、同一个择优规则
    tm = post["day_index"].to_numpy(float) / 30.4
    fa = dca.fit_best(tm, post["oil_t"].to_numpy(float), d_min_year=0.075)
    t0 = float(tm.min())

    # 物理约束：月度开井日均口径，自峰值起算
    mo, ra = physics_dca.monthly_series(post["day_index"].to_numpy() - t_peak + 1,
                                        post["oil_t"].to_numpy(), post["hours_on"].to_numpy())
    if len(mo) < 6:
        raise KernelError(f"井 {w['well_code_anon']} 峰后有效月份不足 6 个，物理约束递减不适用")
    tt = mo - mo[0]
    t_now = float(tt.max())
    lib = _physics_library().get(w["block"], {})
    prior = physics_dca.analog_prior([f for k, f in lib.items() if k != w["well_id"]],
                                     int(pcfg.get("min_analogs", 5)))
    cap = None
    try:
        rf = _rf_reference(w["block"])
        ooip = estimate_reserves_volumetric(well_code)["ooip_t"]["p50"]
        cap = float(ooip) * float(rf["high_pct"]) / 100.0 if ooip else None
    except KernelError:
        cap = None
    f = physics_dca.fit(tt, ra, q_econ=q_econ, cum_to_now=cum, eur_cap_t=cap, prior=prior)
    rem, t_econ, capped = physics_dca.remaining(f, q_econ, t_now)
    band = physics_dca.bootstrap_remaining(tt, ra, f, q_econ=q_econ, cum_to_now=cum, prior=prior,
                                           n_boot=int(pcfg.get("n_bootstrap", 30)))
    arps_rem = dca.eur(fa, q_econ, t_start_month=float(tm.max() - t0))["remaining"]

    # 诊断曲线：横轴统一为"投产后月份"
    mo_all, ra_all = physics_dca.monthly_series(p["day_index"].to_numpy(), p["oil_t"].to_numpy(),
                                                p["hours_on"].to_numpy())
    peak_m = t_peak / 30.4
    flow = physics_dca.flow_regime(mo_all, ra_all, t_peak_month=peak_m)
    live = p[p["hours_on"] > 0]
    mk = ((live["day_index"] - 1) // 30.4).astype(int)
    agg = live.groupby(mk).agg(q=("oil_t", "mean"), whp=("whp_mpa", "mean"), n=("oil_t", "size"))
    cum_m = p.groupby(((p["day_index"] - 1) // 30.4).astype(int))["oil_t"].sum().cumsum()
    agg = agg[agg["n"] >= 8].join(cum_m.rename("cum"), how="left")
    rnp = physics_dca.rnp_diagnostic(agg["cum"].to_numpy(float), agg["q"].to_numpy(float),
                                     agg["whp"].to_numpy(float))

    horizon = float(min(max(t_econ, tt.max()) + 12.0, 360.0))
    grid = np.arange(0.0, horizon + 1e-9, 1.0)
    arps_q = dca.rate(grid, fa)
    phys_q = physics_dca.rate(grid, f)
    x_arps = grid + t0                       # Arps 的时间原点是峰后首个有效日
    x_phys = grid + peak_m + mo[0]

    def curve(xs, qs):
        keep = qs >= q_econ * 0.5
        return [[round(float(a), 2), round(float(b), 3)] for a, b in zip(xs[keep], qs[keep])]

    return dict(**_env(trace_id), well_code=w["well_code_anon"],
                method="线性流（q∝t^-1/2，等效 b=2）→ 边界控制流（b≤1）→ 终端递减；同区块成熟井参数作先验（最大后验），"
                       "EUR 不超过容积法 OOIP P50 × 区块采收率 90% 分位",
                physics=dict(qi=_r(f.qi), di_per_month=_r(f.di, 4), t_elf_month=_r(f.t_elf, 2), b_bdf=_r(f.b_bdf),
                             d_min_per_month=_r(f.d_min, 5), r2=_r(f.r2), rmse=_r(f.rmse), n_points=f.n_points,
                             prior_weight=f.prior_weight, converged=f.converged),
                prior=dict(source=(f"同区块 {prior['n_analogs']} 口成熟井" if prior else "宽先验（同区块成熟井不足）"),
                           **({k: _r(v, 4) for k, v in prior.items() if k != "n_analogs"} if prior else {})),
                eur=dict(p10=_r(cum + band["p10"]), p50=_r(cum + band["p50"]), p90=_r(cum + band["p90"]),
                         low_estimate=_r(cum + band["low_estimate"]), unit="t", n_bootstrap=band["n_boot"]),
                eur_point_t=_r(cum + rem), cum_to_date_t=_r(cum),
                eur_cap_t=_r(cap), cap_binding=bool(f.cap_binding),
                arps=dict(model=fa.model, b=_r(fa.b), di_per_month=_r(fa.di, 4), eur_point_t=_r(cum + arps_rem)),
                eur_vs_arps_pct=_r((rem - arps_rem) / max(cum + arps_rem, 1e-9) * 100, 1),
                economics=dict(q_econ_t_per_d=_r(q_econ), t_econ_month_after_peak=_r(t_econ, 1),
                               t_econ_capped=capped, price_deck_id=price_deck_id),
                flow_regime=flow, rnp=rnp,
                chart=dict(observed=[[round(float(a), 2), round(float(b), 3)] for a, b in zip(mo_all, ra_all)],
                           peak_month=_r(peak_m, 2), now_month=_r(float(mo_all.max()) if len(mo_all) else None, 2),
                           arps=curve(x_arps, arps_q), physics=curve(x_phys, phys_q),
                           t_elf_month=_r(peak_m + mo[0] + physics_dca._elf_effective(f), 2)))


@cached_service
def estimate_reserves_volumetric(well_code: str, mc_samples: int = 10000,
                                 trace_id: Optional[str] = None) -> Dict:
    w = _resolve(well_code)
    t = _tables()
    st = t["static"][t["static"]["well_id"] == w["well_id"]]
    if st.empty:
        raise KernelError(f"井 {w['well_code_anon']} 缺少静态地质参数，无法做容积法")
    res = volumetric.well_ooip(w, st.iloc[0], n=mc_samples)
    return dict(**_env(trace_id), well_code=w["well_code_anon"],
                ooip_t={k: (_r(v) if isinstance(v, (int, float)) else v)
                        for k, v in res["ooip"].items()},
                inputs={k: _r(v) if isinstance(v, float) else v
                        for k, v in res["inputs"].items()},
                method="容积法 + 蒙特卡洛；P90 为低估计、P10 为高估计")


@cached_service
def cross_check_reserves(well_code: str, trace_id: Optional[str] = None) -> Dict:
    w = _resolve(well_code)
    d = fit_dca(well_code, trace_id=trace_id)
    v = estimate_reserves_volumetric(well_code, trace_id=trace_id)
    ref = _rf_reference(w["block"])
    diag = crosscheck.diagnose(
        eur={k: d["eur"][k] for k in ("p10", "p50", "p90")},
        ooip={k: v["ooip_t"][k] for k in ("p10", "p50", "p90")},
        rf_reference={k: ref[k] for k in ("p10", "p50", "p90", "low_pct", "high_pct")},
        dca_params=dict(b=d["fit"]["b"], d_min=d["fit"]["d_min_per_month"]),
        lateral_length=float(w["lateral_length"] or 0))
    diag["rf_reference_source"] = dict(n=ref["n"], source=ref["source"], block=w["block"])
    return dict(**_env(trace_id), well_code=w["well_code_anon"], **diag)


@functools.lru_cache(maxsize=32)
def _rf_reference(block: str) -> Dict:
    t = _tables()
    m = t["master"][t["master"]["block"] == block]
    lab = t["labels"].merge(m[["well_id"]], on="well_id")
    st = t["static"]
    rows = m.merge(st, on="well_id", how="left").merge(lab[["well_id", "eur"]], on="well_id")
    rows = rows.dropna(subset=["eur", "net_pay_m"])
    if rows.empty:
        return dict(p90=1.0, p50=3.0, p10=8.0, n=0, source="default")
    oo = np.array([volumetric.well_ooip(r, r, n=1500, seed=1)["ooip"]["p50"]
                   for _, r in rows.iterrows()])
    return crosscheck.block_rf_reference(rows["eur"].to_numpy(float), oo)


@cached_service
def sec_screen(well_code: str, as_of: str = "2026-12-31",
               price_deck_id: str = DEFAULT_PRICE_DECK,
               in_five_year_plan: Optional[bool] = None,
               offset_continuity_evidence: Optional[bool] = None,
               trace_id: Optional[str] = None) -> Dict:
    b = _bundle()
    w = _resolve(well_code)
    p = _prod(w["well_id"])
    on = p[p["hours_on"] > 0]
    try:
        d = fit_dca(well_code, price_deck_id=price_deck_id, trace_id=trace_id)
        dca_error = None
    except KernelError as exc:
        d, dca_error = None, str(exc)

    econ_limit = economics.economic_limit_rate(price_deck_id)
    econ = d["economics"] if d else dict(
        q_econ_t_per_d=_r(econ_limit["q_econ"]), t_econ_month=None, t_econ_capped=None,
        avg_12m_price_usd_bbl=econ_limit["avg_12m_price_usd_bbl"],
        price_deck_id=price_deck_id)
    current_rate = float(on["oil_t"].tail(14).mean())
    is_econ = current_rate > econ["q_econ_t_per_d"]

    cls = sec_classify.classify(dict(status=w["status"]), prod_days=int(len(on)),
                                economic=is_econ, in_five_year_plan=in_five_year_plan,
                                offset_continuity_evidence=offset_continuity_evidence)

    cum = float(p["oil_t"].sum())
    # 已证实储量取低估计（p10 分位），对应准则「概率法下 >=90% 把握」的口径。
    # 递减分析不可用时不给数 —— 宁可空着让人补，也不给一个看起来很像样的假数。
    proved = None
    if d is not None and cls["category"] in ("PDP", "PDNP"):
        proved = max(d["eur"]["low_estimate"] - cum, 0.0)
    elif cls["category"] not in ("PDP", "PDNP"):
        proved = 0.0
    env = _env(trace_id)
    cov = b.get("coverage", {}).get("eur")
    items = sec_checklist.build(
        prod_days=int(len(on)),
        economic=dict(q_econ=econ["q_econ_t_per_d"], current_rate=current_rate,
                      t_econ_month=econ["t_econ_month"], t_econ_capped=econ.get("t_econ_capped"),
                      price_deck_id=price_deck_id, as_of=as_of,
                      avg_12m_price_usd_bbl=econ["avg_12m_price_usd_bbl"]),
        dca_error=dca_error,
        classification=cls, coverage=cov, model_version=env["model_version"],
        label_def_version=env["label_def_version"], data_source=env["data_source"],
        trace_id=env["trace_id"], has_pud=(cls["category"] == "PUD"),
        five_year_plan_confirmed=in_five_year_plan,
        reliability_report=(f"回测集 EUR 区间覆盖率 {cov:.2f}（名义 0.80）" if cov else None))

    return dict(**env, well_code=w["well_code_anon"], as_of=as_of,
                category=cls["category"], category_rationale=cls["rationale"],
                proved_reserves_t=_r(proved),
                basis=("low_estimate（EUR 的 10% 分位，即行业口径的 P90 低估计）"
                       if proved is not None else "不可得：递减分析不适用，见 dca_error"),
                dca_available=d is not None, dca_error=dca_error,
                cum_to_date_t=_r(cum),
                economics=econ, checklist=items,
                summary=sec_checklist.summarize(items))


@functools.lru_cache(maxsize=1)
def _well_agg():
    """每口井的累产与历史长度。**一次分组查询**，别对 prod_daily 做多次全表扫描 ——
    百万行量级下，一条 SELECT SUM(...) 全表扫描可能比分组查询慢一个数量级。"""
    return db.read_df("SELECT well_id, SUM(oil_t) cum, MAX(day_index) hist "
                      "FROM prod_daily GROUP BY well_id")


@cached_service
def list_wells(limit: int = 400, trace_id: Optional[str] = None) -> Dict:
    """井列表，供前端选井与画井位图。坐标已偏移，不可反推真实井位。"""
    m = _tables()["master"]
    lab = _tables()["labels"]
    cum = _well_agg()
    df = (m.merge(cum, on="well_id", how="left")
           .merge(lab[["well_id", "t_peak", "q_peak", "eur", "label_quality"]],
                  on="well_id", how="left"))
    df = df.sort_values("first_prod_date", ascending=False).head(limit)
    wells = [dict(well_code=r["well_code_anon"], block=r["block"], layer=r["layer"],
                  well_type=r["well_type"], status=r["status"],
                  first_prod_date=r["first_prod_date"],
                  x=_r(r["x_off"], 1), y=_r(r["y_off"], 1),
                  history_days=int(r["hist"] or 0), cum_oil_t=_r(r["cum"]),
                  t_peak=_r(r["t_peak"]), q_peak=_r(r["q_peak"]), eur=_r(r["eur"]),
                  is_new=bool((r["hist"] or 0) < 420))
             for _, r in df.iterrows()]
    return dict(**_env(trace_id), n=len(wells), wells=wells)


@cached_service
def well_curve(well_code: str, agg: str = "month", trace_id: Optional[str] = None) -> Dict:
    """单井实测产量/压力序列。按月聚合，日度噪声太大不适合画在概率带上。"""
    w = _resolve(well_code)
    p = _prod(w["well_id"])
    p = p[p["hours_on"] > 0].copy()
    p["m"] = ((p["day_index"] - 1) // 30) + 1
    g = p.groupby("m").agg(oil_t=("oil_t", "mean"), whp=("whp_mpa", "mean"),
                           wc=("water_cut", "mean"), days=("oil_t", "size")).reset_index()
    g = g[g["days"] >= 8]
    lab = _tables()["labels"]
    lab = lab[lab["well_id"] == w["well_id"]]
    return dict(**_env(trace_id), well_code=w["well_code_anon"],
                month=[int(x) for x in g["m"]],
                oil_t_per_d=[_r(x) for x in g["oil_t"]],
                whp_mpa=[_r(x) for x in g["whp"]],
                water_cut_pct=[_r(x) for x in g["wc"]],
                actual=dict(
                    t_peak=_r(lab["t_peak"].iloc[0]) if len(lab) else None,
                    q_peak=_r(lab["q_peak"].iloc[0]) if len(lab) else None,
                    p_peak=_r(lab["p_peak"].iloc[0]) if len(lab) else None,
                    eur=_r(lab["eur"].iloc[0]) if len(lab) else None,
                    label_quality=lab["label_quality"].iloc[0] if len(lab) else None))


@cached_service
def overview(trace_id: Optional[str] = None) -> Dict:
    """总览看板：数据规模、质量门禁、模型指标。数字全部来自训练时落盘的 meta。"""
    b = _bundle()
    meta = b["meta"]
    m = _tables()["master"]
    agg = _well_agg()
    lab = _tables()["labels"]
    return dict(**_env(trace_id),
                data=dict(n_wells=int(len(m)), n_wells_with_prod=int(len(agg)),
                          cum_oil_t=_r(agg["cum"].sum()),
                          n_labeled=int((lab["label_quality"] == "ok").sum()),
                          n_new_wells=int(sum(1 for _ in m[m["first_prod_date"] > "2026-01-01"].index)),
                          blocks=sorted(m["block"].unique().tolist())),
                quality=meta.get("quality", {}),
                model=dict(model_version=meta["model_version"], backend=meta["backend"],
                           split_method=meta["split_method"], n_train=meta["n_train"],
                           n_calib=meta["n_calib"], n_test=meta["n_test"],
                           label_def_version=meta["label_def_version"]),
                report=meta.get("report", {}))


def eval_summary(trace_id: Optional[str] = None) -> Dict:
    """读取最近一次评测产物。没跑过就如实说没跑过，不编数字。"""
    import json as _json
    out: Dict = dict(**_env(trace_id))
    for key, fname in (("agent", "eval_agent.json"), ("algo", "eval_algo.json"), ("dca", "eval_dca.json")):
        f = path("artifacts_dir") / fname
        out[key] = _json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    meta = _bundle()["meta"]
    out["model"] = dict(model_version=meta["model_version"], split_method=meta.get("split_method"),
                        failure_cases=meta.get("failure_cases"), drift=(meta.get("seq") or {}).get("drift"),
                        families={t: dict(families=r.get("families"), seq_weight=r.get("seq_weight"))
                                  for t, r in (meta.get("report") or {}).items()})
    out["hint"] = "为空表示尚未运行：python -m src.cli eval-agent / eval-algo / eval-dca"
    return out


def reserves_reconcile(values: Dict[str, float], trace_id: Optional[str] = None) -> Dict:
    return dict(**_env(trace_id), **sec_reconcile.build(values))


def _r(v, nd: int = 3):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
            return None
        return round(float(v), nd)
    except Exception:
        return v


@cached_service
def production_timeline(trace_id: Optional[str] = None) -> Dict:
    """全油田生产动态回放：逐月、逐井的日历日均产油（t/d），以及逐月的油田合计指标。

    井位图上的点大小由前端按 rate 缩放（纯视觉映射）；页面上显示的全部数字取自这里的合计数组。"""
    mon = _monthly()
    months = sorted(mon["ym"].unique())
    pos = {ym: i for i, ym in enumerate(months)}
    days_in = np.array([pd.Period(ym, "M").days_in_month for ym in months], float)
    n = len(months)
    field = np.zeros(n)
    producing = np.zeros(n, int)
    wells = []
    for wid, g in mon.groupby("well_id"):
        ii = g["ym"].map(pos).to_numpy(int)
        rate = g["oil_t"].to_numpy(float) / days_in[ii]
        field[ii] += np.nan_to_num(rate)
        producing[ii] += (g["days_on"].to_numpy() > 0).astype(int)
        s0, s1 = int(ii.min()), int(ii.max())
        arr = np.full(s1 - s0 + 1, np.nan)
        arr[ii - s0] = rate
        wells.append(dict(well_code=_code(wid), start=s0,
                          rate=[None if not np.isfinite(v) else round(float(v), 2) for v in arr]))
    m = _tables()["master"]
    fp = m["first_prod_date"].astype(str).str[:7].value_counts()
    new = np.array([int(fp.get(ym, 0)) for ym in months])
    vol = mon.groupby("ym")["oil_t"].sum().reindex(months).fillna(0.0).to_numpy()
    return dict(**_env(trace_id), months=months,
                field_rate_t_per_d=[round(float(v), 1) for v in field],
                producing_wells=producing.tolist(), new_wells=new.tolist(),
                cum_oil_t=[round(float(v), 0) for v in np.cumsum(vol)],
                max_well_rate=round(float(np.nanmax([max([r for r in w["rate"] if r is not None] or [0]) for w in wells])), 2),
                wells=wells, note="日均产油按日历天数折算（当月产油 ÷ 当月天数）")


@functools.lru_cache(maxsize=1)
def _retriever():
    from ..agent.rag.retriever import Retriever
    return Retriever()


def standards_toc(trace_id: Optional[str] = None) -> Dict:
    r = _retriever()
    toc = r.toc()
    return dict(**_env(trace_id), n=len(toc), clauses=toc,
                edition=next((c.meta.get("source") for c in r.chunks if c.meta.get("source")), None),
                note="英文为 eCFR 官方原文；中文标题与要点为平台整理的非官方说明，合规判断以原文为准。")


def standards_clause(citation: str, trace_id: Optional[str] = None) -> Dict:
    c = _retriever().get(citation)
    if c is None:
        raise KernelError(f"条款 {citation!r} 不在条文库中")
    return dict(**_env(trace_id), clause=c)


def standards_search(query: str, top_k: int = 8, trace_id: Optional[str] = None) -> Dict:
    return dict(**_env(trace_id), query=query, results=_retriever().search(query, top_k=int(top_k)))


# =========================================================================== #
# SEC 单元"新-老-措"构成评估（docs/dev-plan-sec-unit.md）
#
# 评估对象 scope 可以是 SEC 单元号、采油厂号或名称、公司号或名称。
# 采油厂与公司的结果是下属各单元结果相加：各单元已证实储量取低值，相加是保守口径。
# =========================================================================== #
SCENARIOS = ("sec", "assessment", "impairment")
CATEGORY_CN = {"infill": "提采新井", "extension": "扩边井"}
BASIS_CN = {"dynamic": "动态法", "analog": "类比法", "model_low": "模型法低估计", None: "不可得"}
STATUS_CN = {"ok": "可评", "insufficient_pre": "措施前历史不足", "insufficient_post": "观察期不足"}
_FIT_CACHE: Dict = {}      # 逐井递减拟合只依赖数据与基准日，多个价格情景共用


def _su() -> Dict:
    return config()["sec_unit"]


@functools.lru_cache(maxsize=1)
def _units() -> Tuple[pd.DataFrame, pd.DataFrame]:
    su = db.read_df("SELECT * FROM sec_unit ORDER BY plant_id, unit_id")
    uw = db.read_df("SELECT * FROM unit_well")
    if su.empty:
        raise KernelError("评估单元表为空，请先运行 python -m src.cli init")
    return su, uw


@functools.lru_cache(maxsize=1)
def _monthly() -> pd.DataFrame:
    """月度生产表：一次分组查询，全部单元级计算共用。"""
    return db.read_df(
        "SELECT well_id, substr(dt, 1, 7) AS ym, SUM(oil_t) AS oil_t, SUM(water_m3) AS water_m3, "
        "SUM(CASE WHEN hours_on > 0 THEN 1 ELSE 0 END) AS days_on, COUNT(*) AS days "
        "FROM prod_daily GROUP BY well_id, substr(dt, 1, 7)")


@functools.lru_cache(maxsize=1)
def _events() -> pd.DataFrame:
    return db.read_df("SELECT well_id, dt, event_type FROM well_event")


@functools.lru_cache(maxsize=1)
def _codes() -> Dict[str, str]:
    m = _tables()["master"]
    return dict(zip(m["well_id"], m["well_code_anon"]))


def _code(well_id: str) -> str:
    return _codes().get(well_id, well_id)


def _wells_of(unit_ids) -> List[str]:
    _, uw = _units()
    return uw[uw["unit_id"].isin(list(unit_ids))]["well_id"].tolist()


def _scope(scope: str) -> Dict:
    su, _ = _units()
    key = str(scope or "").strip()
    if key in set(su["unit_id"]):
        hit = su[su["unit_id"] == key]
        level, name = "unit", hit["unit_name"].iloc[0]
    elif key in set(su["plant_id"]) | set(su["plant_name"]):
        hit = su[(su["plant_id"] == key) | (su["plant_name"] == key)]
        level, name = "plant", hit["plant_name"].iloc[0]
    elif key in set(su["company_id"]) | set(su["company_name"]):
        hit, level, name = su, "company", su["company_name"].iloc[0]
    else:
        raise KernelError(
            f"评估对象 {key!r} 不存在。可选 SEC 单元：{'、'.join(su['unit_id'])}；"
            f"采油厂：{'、'.join(dict.fromkeys(su['plant_name']))}；公司：{su['company_name'].iloc[0]}")
    return dict(level=level, id=key, name=name, unit_ids=hit["unit_id"].tolist())


def _scope_out(sc: Dict) -> Dict:
    return dict(level=sc["level"], id=sc["id"], name=sc["name"], n_units=len(sc["unit_ids"]))


def _as_of(as_of: str) -> Tuple[str, str]:
    dates = [str(d) for d in _su()["evaluation_dates"]]
    if str(as_of) not in dates:
        raise KernelError(f"评估基准日 {as_of} 不在可选范围内：{'、'.join(dates)}")
    try:
        return str(as_of), economics.deck_for(str(as_of))
    except KeyError as exc:
        raise KernelError(str(exc.args[0])) from None


def _scenario(deck: str, scenario: str) -> Dict:
    try:
        return economics.scenario_params(deck, scenario)
    except KeyError as exc:
        raise KernelError(str(exc.args[0])) from None


def _years_back_ym(as_of: str, years: int) -> str:
    return workload.idx_to_ym(workload.ym_to_idx(str(as_of)[:7]) - int(years) * 12 + 1)


def _int_list(v, default) -> List[int]:
    if v in (None, "", []):
        return [int(x) for x in default]
    if isinstance(v, (int, float)):
        return [int(v)]
    if isinstance(v, str):
        return [int(x) for x in v.replace("，", ",").split(",") if x.strip()]
    return [int(x) for x in v]


@functools.lru_cache(maxsize=8)
def _new_wells(as_of: str) -> pd.DataFrame:
    cfg = _su()
    return workload.identify_new_wells(_tables()["master"], as_of, cfg["period_months"],
                                       cfg["new_well"]["radius_m"],
                                       cfg["new_well"]["min_old_neighbors"])


@functools.lru_cache(maxsize=8)
def _new_well_model_estimates(as_of: str) -> Dict[str, Dict]:
    """新井"模型法"：全生命周期模型对本期新井的 EUR 分位数（保形校准后），批量预测。

    只喂基准日之前的数据；观测窗不够的井不给模型值（交给类比法），不拿短数据硬算。
    """
    b = _bundle()
    if "eur" not in b["model"].models:
        return {}
    ids = _new_wells(as_of)["well_id"].tolist()
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    prod = db.read_df(f"SELECT * FROM prod_daily WHERE well_id IN ({marks}) AND dt <= ? "
                      "ORDER BY well_id, day_index", (*ids, as_of))
    hist = prod.groupby("well_id")["day_index"].max()
    ok_ids = hist[hist >= b["obs_days"]].index.tolist()
    if not ok_ids:
        return {}
    t = _tables()
    X = encode(build(prod[prod["well_id"].isin(ok_ids)],
                     t["master"][t["master"]["well_id"].isin(ok_ids)].reset_index(drop=True),
                     t["static"], b["obs_days"], ref_labels=b["train_labels"],
                     spatial_master=t["master"]))
    wids = X["well_id"].tolist()
    p = b["model"].predict(align(X, b["feature_cols"]))["eur"]
    lo, hi = b["conformal"].apply("eur", p["p10"].to_numpy(float), p["p90"].to_numpy(float))
    cum = prod.groupby("well_id")["oil_t"].sum()
    out = {}
    for i, wid in enumerate(wids):
        c = float(cum.get(wid, 0.0))
        out[wid] = dict(p10=max(float(lo[i]), c), p50=max(float(p["p50"].iloc[i]), c),
                        p90=max(float(hi[i]), c))
    return out


_EVAL_MEMO: Dict[Tuple[str, str, str, str], Dict] = {}
_EVAL_LOCKS: Dict[Tuple[str, str, str, str], threading.Lock] = {}
_EVAL_GUARD = threading.Lock()
# 评估算法源码也进快照指纹：改了算法，旧快照自动作废，不会读到按旧口径算的结果
SNAPSHOT_SOURCES = ("sec/composition.py", "reserves/workload.py", "reserves/dca.py", "sec/economics.py")


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"评估结果含不可序列化的类型 {type(o).__name__}")


@functools.lru_cache(maxsize=1)
def _snapshot_ready() -> bool:
    """老库没有快照表时补建（schema.sql 全部是 IF NOT EXISTS，重复执行无副作用）。"""
    try:
        db.init_schema()
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _eval_fingerprint() -> str:
    """快照指纹：数据、模型、评估口径、价格册、评估算法源码任一变化，指纹就变。"""
    root = Path(__file__).resolve().parents[1]
    su, uw = _units()
    parts = [
        _bundle()["meta"].get("model_version"),
        json.dumps(_su(), sort_keys=True, ensure_ascii=False, default=str),
        json.dumps(price_decks(), sort_keys=True, ensure_ascii=False, default=str),
        db.read_df("SELECT COUNT(*) AS n, MAX(dt) AS dt, ROUND(SUM(oil_t), 3) AS oil, "
                   "ROUND(SUM(water_m3), 3) AS water FROM prod_daily").to_json(),
        db.read_df("SELECT COUNT(*) AS n, MAX(dt) AS dt FROM well_event").to_json(),
        su.to_json(), uw.sort_values(list(uw.columns)).to_json(),
        _tables()["master"].sort_values("well_id").to_json(),
        json.dumps(_locations(), sort_keys=True, ensure_ascii=False, default=str),
    ] + [(root / f).read_text(encoding="utf-8") for f in SNAPSHOT_SOURCES]
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


def _snapshot_get(unit_id: str, as_of: str, deck: str, scenario: str) -> Optional[Dict]:
    if not _snapshot_ready():
        return None
    try:
        df = db.read_df("SELECT payload_json FROM sec_eval_snapshot WHERE unit_id = ? AND as_of = ? "
                        "AND price_deck_id = ? AND scenario = ? AND fingerprint = ?",
                        (unit_id, as_of, deck, scenario, _eval_fingerprint()))
    except Exception:
        return None
    return json.loads(df["payload_json"].iloc[0]) if len(df) else None


def _snapshot_put(unit_id: str, as_of: str, deck: str, scenario: str, ev: Dict) -> None:
    if not _snapshot_ready():
        return
    try:
        with db.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sec_eval_snapshot (unit_id, as_of, price_deck_id, scenario, "
                "fingerprint, payload_json, model_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (unit_id, as_of, deck, scenario, _eval_fingerprint(),
                 json.dumps(ev, ensure_ascii=False, default=_json_default),
                 _bundle()["meta"].get("model_version"), trace.now_iso()))
    except Exception:
        pass                  # 快照只为加速；写不进去（库被锁等）不影响本次结果


def _evaluate(unit_id: str, as_of: str, deck: str, scenario: str) -> Dict:
    """单元评估（结果原样缓存，调用方不得修改返回的 dict）。

    三级取数：进程内缓存 → 库里指纹一致的快照 → 逐井重算并写快照。
    同一个评估同一时刻只算一次（服务启动预热与页面请求可能并发）。
    """
    key = (unit_id, as_of, deck, scenario)
    hit = _EVAL_MEMO.get(key)
    if hit is not None:
        return hit
    with _EVAL_GUARD:
        lock = _EVAL_LOCKS.setdefault(key, threading.Lock())
    with lock:
        if key not in _EVAL_MEMO:
            ev = _snapshot_get(*key)
            if ev is None:
                ev = _compute_evaluation(*key)
                _snapshot_put(*key, ev)
            _EVAL_MEMO[key] = ev
    return _EVAL_MEMO[key]


def _evaluate_cache_clear() -> None:
    with _EVAL_GUARD:
        _EVAL_MEMO.clear()
        _EVAL_LOCKS.clear()


_evaluate.cache_clear = _evaluate_cache_clear


def _compute_evaluation(unit_id: str, as_of: str, deck: str, scenario: str) -> Dict:
    """逐井重算一个单元的评估。

    deck 与 as_of 分开传：价格修订要用"期末的数据 + 期初的价格册"重算一次。
    返回值统一过一遍 JSON 往返 —— 与从快照读回的结果逐位一致，首次计算和重启后读快照不会差一个字节。
    """
    su, _ = _units()
    u = su[su["unit_id"] == unit_id].iloc[0]
    ids = set(_wells_of([unit_id]))
    master = _tables()["master"]
    mon = _monthly()
    cfg = _su()
    econ = economics.economic_limit_rate(deck, scenario, opex_factor=float(u["opex_factor"]))
    qe = econ["q_econ"]
    ev = sec_composition.evaluate_unit(
        monthly=mon[mon["well_id"].isin(ids)], master=master[master["well_id"].isin(ids)],
        events=_events(), new_wells=_new_wells(as_of), as_of=as_of, q_econ_well=qe,
        period_months=cfg["period_months"], measure_cfg=cfg["measure"],
        min_fit_months=cfg["decline"]["min_fit_months"],
        type_curve_min_history=cfg["type_curve"]["min_history_months"],
        model_estimates=_new_well_model_estimates(as_of), fit_cache=_FIT_CACHE,
        category_cfg=_category_cfg(unit_id, as_of, econ))
    return json.loads(json.dumps(ev, ensure_ascii=False, default=_json_default))


def _records(df: pd.DataFrame) -> List[Dict]:
    return df.astype(object).where(df.notna(), None).to_dict("records")


@functools.lru_cache(maxsize=1)
def _locations() -> List[Dict]:
    """部署井位表（老库没有这张表时补建后为空）。"""
    _snapshot_ready()
    try:
        return _records(db.read_df("SELECT * FROM unit_location ORDER BY location_id"))
    except Exception:
        return []


@functools.lru_cache(maxsize=8)
def _producing_xy(as_of: str) -> List[List[float]]:
    """基准日在产井（近 3 个月有产量）的坐标 —— PUD 连续性判定的依据，不限单元、不限层系。"""
    end = str(as_of)[:7]
    lo = workload.idx_to_ym(workload.ym_to_idx(end) - 2)
    mon = _monthly()
    ids = set(mon[(mon["ym"] >= lo) & (mon["ym"] <= end) & (mon["oil_t"] > 0)]["well_id"])
    m = _tables()["master"]
    m = m[m["well_id"].isin(ids)]
    return m[["x_off", "y_off"]].astype(float).values.tolist()


@functools.lru_cache(maxsize=1)
def _first_prod_map() -> Dict[str, str]:
    m = _tables()["master"]
    return dict(zip(m["well_id"], m["first_prod_date"].astype(str)))


def _category_cfg(unit_id: str, as_of: str, econ: Dict) -> Dict:
    cfg = _su()
    fin = cfg["finance"]
    netrev = float(econ["net_revenue_usd_per_tonne"])
    return dict(
        pdnp=dict(max_shut_in_months=cfg["pdnp"]["max_shut_in_months"],
                  # 复产费用折成吨数：剩余可采的净收入须覆盖复产作业费
                  restart_min_t=float(fin["restart_cost_wan"]) * 1e4 / float(fin["fx_cny_per_usd"]) / max(netrev, 1e-9)),
        pud=cfg["pud"], locations=[r for r in _locations() if r["unit_id"] == unit_id],
        producing_xy=_producing_xy(str(as_of)), drilled_first_prod=_first_prod_map(),
        net_revenue_usd_per_t=netrev, opex_per_well_month_usd=float(econ["opex_per_well_month_usd"]),
        fx_cny_per_usd=float(fin["fx_cny_per_usd"]), default_capex_wan=float(fin["pud_capex_wan"]))


def _composition_frame(unit_ids, as_of: str, deck: str, since_ym: str, start_ym: str) -> pd.DataFrame:
    """若干单元逐月"新-老-措"产量构成之和。since_ym 之后投产的井算新井、实施的措施单列。"""
    master = _tables()["master"]
    mon = _monthly()
    frames = []
    for u in unit_ids:
        ids = set(_wells_of([u]))
        frames.append(workload.monthly_composition(
            mon[mon["well_id"].isin(ids)], master[master["well_id"].isin(ids)],
            _evaluate(u, as_of, deck, "sec")["measures"], start_ym=start_ym,
            end_ym=str(as_of)[:7], new_since_ym=since_ym, measure_since_ym=since_ym))
    return pd.concat(frames).groupby("ym", as_index=False).sum(numeric_only=True)


@cached_service
def list_units(trace_id: Optional[str] = None) -> Dict:
    """评估单元层级：公司 → 采油厂 → SEC 单元，附井数、可选基准日与价格情景。"""
    su, uw = _units()
    n = uw.groupby("unit_id").size()
    plants = [dict(plant_id=pid, plant_name=g["plant_name"].iloc[0],
                   units=[dict(unit_id=r.unit_id, unit_name=r.unit_name, area_type=r.area_type,
                               opex_factor=_r(r.opex_factor), n_wells=int(n.get(r.unit_id, 0)))
                          for r in g.itertuples(index=False)])
              for pid, g in su.groupby("plant_id", sort=False)]
    dates = [str(d) for d in _su()["evaluation_dates"]]
    deck = economics.deck_for(dates[-1])
    return dict(**_env(trace_id), company=dict(id=su["company_id"].iloc[0],
                                               name=su["company_name"].iloc[0]),
                plants=plants, n_units=int(len(su)), evaluation_dates=dates,
                period_months=_su()["period_months"],
                scenarios=[dict(key=s, label=economics.scenario_params(deck, s)["label"])
                           for s in SCENARIOS],
                measure_types=_su()["measure_types"])


@cached_service
def unit_sec_composition(scope: str, as_of: str = "2026-12-31", scenario: str = "sec",
                         trace_id: Optional[str] = None) -> Dict:
    """SEC 单元"新-老-措"PDP 构成：老井基础 / 措施增储 / 提采新井 / 扩边井。"""
    sc = _scope(scope)
    as_of, deck = _as_of(as_of)
    sp = _scenario(deck, scenario)
    evs = [_evaluate(u, as_of, deck, scenario) for u in sc["unit_ids"]]
    su, _ = _units()
    names = su.set_index("unit_id")["unit_name"]

    comps = []
    for i, c0 in enumerate(evs[0]["components"]):
        cs = [ev["components"][i] for ev in evs]
        comps.append(dict(key=c0["key"], name=c0["name"], basis=c0["basis"],
                          reserves_t=float(sum(c["reserves_t"] for c in cs)),
                          n_items=int(sum(c["n_items"] for c in cs))))
    total = sum(c["reserves_t"] for c in comps)
    period_meas = [e for ev in evs for e in ev["measures"] if e["in_period"]]
    ms = workload.summarize_measures(period_meas, _su()["measure_types"])["overall"]
    olds = [ev["old_base"] for ev in evs]
    nws = [w for ev in evs for w in ev["new_wells"]]
    econ = economics.economic_limit_rate(deck, scenario)
    basis_counts: Dict[str, int] = {}
    for w in nws:
        k = BASIS_CN.get(w["basis"], w["basis"])
        basis_counts[k] = basis_counts.get(k, 0) + 1
    return dict(
        **_env(trace_id), scope=_scope_out(sc), as_of=as_of, price_deck_id=deck,
        scenario=scenario, scenario_label=sp["label"], category="PDP",
        period_start_ym=evs[0]["period_start_ym"], period_end_ym=evs[0]["period_end_ym"],
        total_t=_r(total, 1),
        components=[dict(key=c["key"], name=c["name"], reserves_t=_r(c["reserves_t"], 1),
                         share_pct=_r(100 * c["reserves_t"] / total, 1) if total else None,
                         n_items=c["n_items"], basis=c["basis"]) for c in comps],
        units=[dict(unit_id=u, unit_name=names[u], total_t=_r(ev["total_t"], 1),
                    components={c["key"]: _r(c["reserves_t"], 1) for c in ev["components"]})
               for u, ev in zip(sc["unit_ids"], evs)],
        old_wells=dict(n_evaluated=sum(o["n_evaluated"] for o in olds),
                       n_short_history=sum(o["n_short_history"] for o in olds),
                       n_not_producing=sum(o["n_not_producing"] for o in olds),
                       n_insufficient=sum(o["n_insufficient"] for o in olds),
                       n_measure_baseline=sum(o["n_measure_baseline"] for o in olds),
                       best_estimate_t=_r(sum(o["remaining_best_t"] for o in olds), 1)),
        measures=dict(n_in_period=len(period_meas), n_evaluated=ms["n_evaluated"],
                      n_effective=ms["n_effective"], inc_eur_total_t=_r(ms["inc_eur_total_t"], 1)),
        new_wells=dict(n_infill=sum(1 for w in nws if w["category"] == "infill"),
                       n_extension=sum(1 for w in nws if w["category"] == "extension"),
                       basis_counts=basis_counts),
        production_in_period_t=_r(sum(ev["production_in_period_t"] for ev in evs), 1),
        economics=dict(price_usd_bbl=econ["price_usd_bbl"], q_econ_t_per_d=_r(econ["q_econ"]),
                       opex_per_well_month_usd=econ["opex_per_well_month_usd"],
                       note="单井经济极限为基准成本口径，各单元另乘自身成本系数"),
        proved=dict(pdp_t=_r(total, 1),
                    pdnp_t=_r(sum((ev.get("pdnp") or {}).get("reserves_t", 0.0) for ev in evs), 1),
                    pud_t=_r(sum((ev.get("pud") or {}).get("reserves_t", 0.0) for ev in evs), 1),
                    total_t=_r(sum(ev.get("total_proved_t", ev["total_t"]) for ev in evs), 1)),
        aggregation_note=None if sc["level"] == "unit" else
        "采油厂/公司结果为下属单元相加；各单元已证实储量取低值，相加为保守口径")


@cached_service
def unit_production_composition(scope: str, as_of: str = "2026-12-31",
                                trace_id: Optional[str] = None) -> Dict:
    """逐月"新-老-措"产量构成（近 3 年）与本期新井 / 措施 / 老井的计划与实际对标。"""
    sc = _scope(scope)
    as_of, deck = _as_of(as_of)
    start_ym, end_ym = workload.period_bounds(as_of, _su()["period_months"])
    df = _composition_frame(sc["unit_ids"], as_of, deck, start_ym, _years_back_ym(as_of, 3))

    ids = set(_wells_of(sc["unit_ids"]))
    master = _tables()["master"]
    fp = master[master["well_id"].isin(ids)]["first_prod_date"].str[:7]
    ev = _events()
    ev = ev[ev["well_id"].isin(ids)]["dt"].str[:7]
    plan = db.read_df("SELECT * FROM unit_plan_monthly WHERE ym >= ? AND ym <= ?", (start_ym, end_ym))
    plan = plan[plan["unit_id"].isin(sc["unit_ids"])].groupby("ym").sum(numeric_only=True)
    act = df.set_index("ym")

    rows = []
    for i in range(workload.ym_to_idx(start_ym), workload.ym_to_idx(end_ym) + 1):
        ym = workload.idx_to_ym(i)
        a = act.loc[ym] if ym in act.index else None
        p = plan.loc[ym] if ym in plan.index else None
        rows.append(dict(
            ym=ym,
            plan_new_wells=None if p is None else int(p["plan_new_wells"]),
            actual_new_wells=int((fp == ym).sum()),
            plan_new_oil_t=None if p is None else _r(p["plan_new_oil_t"], 1),
            actual_new_oil_t=None if a is None else _r(a["new_oil_t"], 1),
            plan_measure_wells=None if p is None else int(p["plan_measure_wells"]),
            actual_measure_wells=int((ev == ym).sum()),
            plan_measure_inc_t=None if p is None else _r(p["plan_measure_inc_t"], 1),
            actual_measure_inc_t=None if a is None else _r(a["measure_inc_t"], 1),
            plan_old_oil_t=None if p is None else _r(p["plan_old_oil_t"], 1),
            actual_old_oil_t=None if a is None else _r(a["old_base_t"], 1)))

    def total(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r[key] is not None]
        return float(sum(vals)) if vals else None

    def pair(plan_key: str, actual_key: str) -> Dict:
        p, a = total(plan_key), total(actual_key)
        return dict(plan=_r(p, 1), actual=_r(a, 1),
                    deviation_pct=_r(100 * (a - p) / p, 1) if p and a is not None else None)

    return dict(**_env(trace_id), scope=_scope_out(sc), as_of=as_of,
                period_start_ym=start_ym, period_end_ym=end_ym,
                months=[dict(ym=r.ym, total_oil_t=_r(r.total_oil_t, 1), new_oil_t=_r(r.new_oil_t, 1),
                             measure_inc_t=_r(r.measure_inc_t, 1), old_base_t=_r(r.old_base_t, 1),
                             n_producing=int(r.n_producing)) for r in df.itertuples(index=False)],
                plan_vs_actual=rows, plan_available=bool(len(plan)),
                period_totals=dict(new_wells=pair("plan_new_wells", "actual_new_wells"),
                                   new_oil=pair("plan_new_oil_t", "actual_new_oil_t"),
                                   measure_wells=pair("plan_measure_wells", "actual_measure_wells"),
                                   measure_inc=pair("plan_measure_inc_t", "actual_measure_inc_t"),
                                   old_oil=pair("plan_old_oil_t", "actual_old_oil_t")),
                note="新井 = 本期投产井的全部产量；措施增油 = 本期措施的实际产量减措施前递减基线；"
                     "老井 = 总产量减前两者")


@cached_service
def unit_base_decline(scope: str, as_of: str = "2026-12-31", exclude_years=None,
                      trace_id: Optional[str] = None) -> Dict:
    """老井基础递减：扣除近 N 年投产新井与近 N 年措施增油后的自然递减率 / 综合递减率。"""
    sc = _scope(scope)
    as_of, deck = _as_of(as_of)
    out = []
    for n in _int_list(exclude_years, _su()["decline"]["exclude_years"]):
        if n < 1:
            raise KernelError(f"扣除年限须为正整数，收到 {n}")
        since = _years_back_ym(as_of, n)
        df = _composition_frame(sc["unit_ids"], as_of, deck, since, since)
        r = workload.decline_rates(df, since, as_of[:7])
        out.append(dict(
            exclude_years=n, window_start_ym=since, window_end_ym=as_of[:7], n_months=r["n_months"],
            natural_monthly_pct=_r(r["natural_monthly_pct"]),
            natural_annual_pct=_r(r["natural_annual_pct"], 2),
            comprehensive_monthly_pct=_r(r["comprehensive_monthly_pct"]),
            comprehensive_annual_pct=_r(r["comprehensive_annual_pct"], 2),
            natural_fit_r2=_r(r["natural_r2"]),
            series=[dict(ym=x.ym, old_base_t=_r(x.old_base_t, 1), measure_inc_t=_r(x.measure_inc_t, 1),
                         new_oil_t=_r(x.new_oil_t, 1)) for x in df.itertuples(index=False)]))
    return dict(**_env(trace_id), scope=_scope_out(sc), as_of=as_of, results=out,
                definition="自然递减率扣新井、扣措施增油；综合递减率只扣新井。"
                           "均为逐月日历日产的指数拟合，年递减率 = 1 − (1 − 月递减率)^12")


@cached_service
def unit_new_wells(scope: str, as_of: str = "2026-12-31", trace_id: Optional[str] = None) -> Dict:
    """本期新井自动分类（提采新井 / 扩边井）及其储量与取值方法。"""
    sc = _scope(scope)
    as_of, deck = _as_of(as_of)
    cfg = _su()["new_well"]
    nearest = _new_wells(as_of).set_index("well_id")["nearest_old_m"]
    rows = []
    for u in sc["unit_ids"]:
        for w in _evaluate(u, as_of, deck, "sec")["new_wells"]:
            rows.append(dict(well_code=_code(w["well_id"]), unit_id=u,
                             category=CATEGORY_CN[w["category"]], category_code=w["category"],
                             first_prod_ym=w["first_prod_ym"], months_on=w["months_on"],
                             n_old_neighbors=w["n_old_neighbors"],
                             nearest_old_m=_r(nearest.get(w["well_id"]), 0),
                             cum_t=_r(w["cum_t"], 1), rate_now_t_per_d=_r(w["rate_now_t_per_d"]),
                             reserves_t=_r(w["reserves_t"], 1),
                             basis=BASIS_CN.get(w["basis"], w["basis"])))
    rows.sort(key=lambda r: (r["category_code"], -(r["reserves_t"] or 0)))
    return dict(**_env(trace_id), scope=_scope_out(sc), as_of=as_of,
                rule=dict(radius_m=cfg["radius_m"], min_old_neighbors=cfg["min_old_neighbors"],
                          text=f"半径 {cfg['radius_m']} m 内评估期前已投产的老井不少于 "
                               f"{cfg['min_old_neighbors']} 口判为提采新井，否则判为扩边井"),
                n_infill=sum(1 for r in rows if r["category_code"] == "infill"),
                n_extension=sum(1 for r in rows if r["category_code"] == "extension"),
                wells=rows)


@cached_service
def unit_measure_effects(scope: str, as_of: str = "2026-12-31", event_type: Optional[str] = None,
                         trace_id: Optional[str] = None) -> Dict:
    """本期措施效果：措施前后单井日产、日产增幅、已实现增油、增加可采储量，按类型汇总。"""
    sc = _scope(scope)
    as_of, deck = _as_of(as_of)
    names = _su()["measure_types"]
    if event_type and event_type not in names:
        code = {v: k for k, v in names.items()}.get(event_type)
        if code is None:
            raise KernelError(f"未知措施类型 {event_type!r}，可选：{'、'.join(names.values())}")
        event_type = code
    eff = [e for u in sc["unit_ids"] for e in _evaluate(u, as_of, deck, "sec")["measures"]
           if e["in_period"] and (not event_type or e["event_type"] == event_type)]
    summ = workload.summarize_measures(eff, names)

    def rnd(d: Dict) -> Dict:
        return {k: (_r(v, 2) if isinstance(v, float) else v) for k, v in d.items()}

    return dict(**_env(trace_id), scope=_scope_out(sc), as_of=as_of,
                event_type=event_type, by_type=[rnd(x) for x in summ["by_type"]],
                overall=rnd(summ["overall"]),
                measures=[dict(well_code=_code(e["well_id"]), event_type=e["event_type"],
                               event_name=names.get(e["event_type"], e["event_type"]),
                               event_ym=e["event_ym"], status=e["status"],
                               status_cn=STATUS_CN.get(e["status"], e["status"]), reason=e["reason"],
                               pre_rate_t_per_d=_r(e.get("pre_rate_t_per_d")),
                               post_rate_t_per_d=_r(e.get("post_rate_t_per_d")),
                               rate_gain_t_per_d=_r(e.get("rate_gain_t_per_d")),
                               realized_inc_t=_r(e.get("realized_inc_t"), 1),
                               inc_remaining_t=_r(e.get("inc_remaining_t"), 1),
                               inc_eur_t=_r(e.get("inc_eur_t"), 1), effective=e.get("effective"))
                          for e in sorted(eff, key=lambda e: e["event_ym"])],
                definition="基线 = 措施前最多 24 个月的递减外推；增加可采储量 = 已实现增油 + 增加的剩余可采")


def _reconcile_values(unit_id: str, fa: str, fdeck: str, ta: str, tdeck: str, scenario: str) -> Dict:
    r = sec_composition.reconcile_auto(_evaluate(unit_id, fa, fdeck, scenario),
                                       _evaluate(unit_id, ta, tdeck, scenario),
                                       _evaluate(unit_id, ta, fdeck, scenario))
    vals = {row["key"]: row["value"] for row in r["table"]}
    vals.pop("closing", None)
    vals["closing"] = r["closing_reported"]
    return vals, r["category_change_detail"]


@cached_service
def unit_reconcile(scope: str, from_as_of: str = "2025-12-31", to_as_of: str = "2026-12-31",
                   scenario: str = "sec", trace_id: Optional[str] = None) -> Dict:
    """期初 → 期末储量对账：产量消耗、价格/成本修订、新井、措施、扩边、技术修订，并给产量法折耗率。"""
    sc = _scope(scope)
    fa, fdeck = _as_of(from_as_of)
    ta, tdeck = _as_of(to_as_of)
    if fa >= ta:
        raise KernelError(f"期初基准日 {fa} 须早于期末基准日 {ta}")
    _scenario(tdeck, scenario)
    per_unit = {u: _reconcile_values(u, fa, fdeck, ta, tdeck, scenario) for u in sc["unit_ids"]}
    detail = dict(n_reactivated=0, reactivated_t=0.0, n_shut_in=0, shut_in_t=0.0, reactivated=[], shut_in=[])
    for u, (_, det) in per_unit.items():
        for k in ("n_reactivated", "reactivated_t", "n_shut_in", "shut_in_t"):
            detail[k] += det[k]
        for k in ("reactivated", "shut_in"):
            detail[k] += [dict(well_code=_code(w), unit_id=u, value_t=_r(v, 1)) for w, v in det[k]]
    for k in ("reactivated", "shut_in"):
        detail[k] = sorted(detail[k], key=lambda x: -abs(x["value_t"] or 0))[:10]
    pdnp_close = float(sum((_evaluate(u, ta, tdeck, scenario).get("pdnp") or {}).get("reserves_t", 0.0)
                           for u in sc["unit_ids"]))

    # 历史评估成果管理：上期结果已入库就用入库值做期初（差额进技术修订），否则按上期数据截面重算
    stored = db.read_df("SELECT unit_id, reserves_t FROM sec_eval_record WHERE as_of = ? "
                        "AND scenario = ? AND component = 'total'", (fa, scenario))
    stored = stored[stored["unit_id"].isin(sc["unit_ids"])].set_index("unit_id")["reserves_t"]
    opening_source = "入库记录" if len(stored) == len(sc["unit_ids"]) else "按上期数据截面重算"
    keys = [k for k, _, _ in sec_reconcile.ROWS]
    values = {k: 0.0 for k in keys}
    closing = 0.0
    for u, (v, _) in per_unit.items():
        if opening_source == "入库记录":
            v = dict(v, opening=float(stored[u]))
            v["technical_revision"] = v["closing"] - sum(v[k] for k in keys if k != "technical_revision")
        for k in keys:
            values[k] += float(v.get(k, 0.0))
        closing += float(v["closing"])
    values["closing"] = closing
    tab = sec_reconcile.build(values)
    prod = -values["production"]
    return dict(**_env(trace_id), scope=_scope_out(sc), from_as_of=fa, to_as_of=ta,
                scenario=scenario, price_deck_open=fdeck, price_deck_close=tdeck,
                opening_source=opening_source,
                table=[dict(r, value=_r(r["value"], 1)) for r in tab["table"]],
                closing_calculated=tab["closing_calculated"], closing_reported=tab["closing_reported"],
                difference=tab["difference"], balanced=tab["balanced"], unit="t",
                depletion_rate_pct=_r(100 * prod / (closing + pdnp_close + prod), 2)
                if closing + pdnp_close + prod > 0 else None,
                pdnp_closing_t=_r(pdnp_close, 1),
                category_change_detail=dict(detail, reactivated_t=_r(detail["reactivated_t"], 1),
                                            shut_in_t=_r(detail["shut_in_t"], 1)),
                depletion_note="产量法折耗率 = 本期产量 /（期末证实已开发储量 PDP + PDNP + 本期产量）；"
                               "折耗额与减值见\"折耗与减值\"",
                note="技术修订为轧差项（与 20-F 的 revisions of previous estimates 口径一致）；"
                     "类别调整 = 停产井复产转入 − 在产井停井转出，PUD 钻井转化按提采新井计入")


@cached_service
def unit_sensitivity(scope: str, as_of: str = "2026-12-31", scenario: str = "sec",
                     trace_id: Optional[str] = None) -> Dict:
    """油价、成本、产量、递减率对 PDP 的影响曲线，及各单元的参数敏感权重。"""
    sc = _scope(scope)
    as_of, deck = _as_of(as_of)
    sp = _scenario(deck, scenario)
    su, _ = _units()
    info = su.set_index("unit_id")
    per = {}
    for u in sc["unit_ids"]:
        f = float(info.loc[u, "opex_factor"])
        per[u] = sec_composition.sensitivity(
            _evaluate(u, as_of, deck, scenario),
            lambda p, k, f=f: economics.economic_limit_rate(deck, scenario, opex_factor=f * k,
                                                            price_usd_bbl=p)["q_econ"],
            sp["price_usd_bbl"])
    first = next(iter(per.values()))
    curves = []
    for i, c0 in enumerate(first["curves"]):
        xs = [s["curves"][i]["x"] for s in per.values()]
        ds = [s["curves"][i]["delta_t"] for s in per.values()]
        x = np.sum(xs, axis=0) if c0["param"] == "rate" else np.asarray(c0["x"])
        curves.append(dict(param=c0["param"], name=c0["name"], unit=c0["unit"],
                           x=[_r(v, 3) for v in x], delta_t=[_r(v, 1) for v in np.sum(ds, axis=0)]))
    swing = {w["param"]: sum(next(x["swing_t"] for x in s["weights"] if x["param"] == w["param"])
                             for s in per.values()) for w in first["weights"]}
    tot = sum(swing.values()) or 1.0
    return dict(**_env(trace_id), scope=_scope_out(sc), as_of=as_of, scenario=scenario,
                scenario_label=sp["label"], base_price_usd_bbl=_r(sp["price_usd_bbl"]),
                base_best_estimate_t=_r(sum(s["base_best_estimate_t"] for s in per.values()), 1),
                curves=curves,
                weights=[dict(param=w["param"], name=w["name"], swing_t=_r(swing[w["param"]], 1),
                              weight=_r(swing[w["param"]] / tot, 4),
                              weight_pct=_r(100 * swing[w["param"]] / tot, 1))
                         for w in first["weights"]],
                units=[dict(unit_id=u, unit_name=info.loc[u, "unit_name"],
                            weights=[dict(param=w["param"], name=w["name"], weight=_r(w["weight"], 4),
                                          weight_pct=_r(100 * w["weight"], 1))
                                     for w in s["weights"]],
                            top_param=max(s["weights"], key=lambda w: w["weight"])["name"])
                       for u, s in per.items()],
                perturbation="油价、成本、老井月产 ±10%，递减率 ±1 个百分点",
                note=first["note"])


@cached_service
def unit_change_attribution(scope: str, from_as_of: str = "2025-12-31",
                            to_as_of: str = "2026-12-31", scenario: str = "sec",
                            trace_id: Optional[str] = None) -> Dict:
    """单元 PDP 为什么变了：对账行项按影响大小排序，每项落到具体的井、措施或价格参数。"""
    sc = _scope(scope)
    fa, fdeck = _as_of(from_as_of)
    ta, tdeck = _as_of(to_as_of)
    if fa >= ta:
        raise KernelError(f"期初基准日 {fa} 须早于期末基准日 {ta}")
    su, _ = _units()
    info = su.set_index("unit_id")
    master = _tables()["master"]
    start_ym, _ = workload.period_bounds(ta, _su()["period_months"])
    mon = _monthly()
    merged: Dict[str, Dict] = {}
    opening = closing = 0.0
    for u in sc["unit_ids"]:
        f = float(info.loc[u, "opex_factor"])
        close = _evaluate(u, ta, tdeck, scenario)
        recon = sec_composition.reconcile_auto(_evaluate(u, fa, fdeck, scenario), close,
                                               _evaluate(u, ta, fdeck, scenario))
        ids = set(_wells_of([u]))
        old_ids = master[master["well_id"].isin(ids) & (master["first_prod_date"].str[:7] < start_ym)]["well_id"]
        tech = workload.well_change_evidence(mon[mon["well_id"].isin(ids)], old_ids, fa[:7], ta[:7],
                                             oil_density=_su()["oil_density_t_per_m3"])
        att = sec_composition.change_attribution(
            recon, close, tech, economics.economic_limit_rate(fdeck, scenario, opex_factor=f),
            economics.economic_limit_rate(tdeck, scenario, opex_factor=f))
        opening += att["opening_t"]
        closing += att["closing_t"]
        for d in att["drivers"]:
            m = merged.setdefault(d["key"], dict(key=d["key"], item=d["item"], value_t=0.0,
                                                 evidence_kind=d["evidence_kind"], evidence=[]))
            m["value_t"] += d["value_t"]
            m["evidence"] += [dict(e, unit_id=u) for e in d["evidence"]]

    sort_key = {"wells": lambda e: -(e.get("reserves_t") or 0),
                "measures": lambda e: -abs(e.get("inc_eur_t") or 0),
                "wells_rate_change": lambda e: e.get("rate_change_t_per_d") or 0,
                "category": lambda e: -abs(e.get("reserves_t") or 0)}
    drivers = []
    for d in merged.values():
        if d["evidence_kind"] == "economics":
            econ_o = economics.economic_limit_rate(fdeck, scenario)
            econ_c = economics.economic_limit_rate(tdeck, scenario)
            ev = [dict(price_open_usd_bbl=econ_o["price_usd_bbl"], price_close_usd_bbl=econ_c["price_usd_bbl"],
                       q_econ_open_t_per_d=_r(econ_o["q_econ"]), q_econ_close_t_per_d=_r(econ_c["q_econ"]))]
        else:
            ev = sorted(d["evidence"], key=sort_key.get(d["evidence_kind"], lambda e: 0))[:3]
            ev = [{k: (_code(v) if k == "well_id" else _r(v, 2) if isinstance(v, float) else v)
                   for k, v in e.items()} for e in ev]
            for e in ev:
                if "well_id" in e:
                    e["well_code"] = e.pop("well_id")
                if "basis" in e:
                    e["basis"] = BASIS_CN.get(e["basis"], e["basis"])
                if "event_type" in e:
                    e["event_name"] = _su()["measure_types"].get(e["event_type"], e["event_type"])
        drivers.append(dict(d, value_t=_r(d["value_t"], 1), evidence=ev))
    denom = sum(abs(d["value_t"]) for d in drivers) or 1.0
    for d in drivers:
        d["share_pct"] = _r(100 * abs(d["value_t"]) / denom, 1)
    drivers.sort(key=lambda d: -abs(d["value_t"]))
    return dict(**_env(trace_id), scope=_scope_out(sc), from_as_of=fa, to_as_of=ta,
                scenario=scenario, opening_t=_r(opening, 1), closing_t=_r(closing, 1),
                change_t=_r(closing - opening, 1), drivers=drivers,
                note="占比为各行项绝对值占全部行项绝对值之和；技术修订列出两期间日产降幅最大的老井")


@functools.lru_cache(maxsize=64)
def _indicator_periods(scope: str, dates: Tuple[str, ...]) -> Tuple[Dict, List[Dict]]:
    """各期指标原始值（不含评分）。评分锚点可随方案切换，值只算一次。"""
    sc = _scope(scope)
    cfg = _su()
    su, _ = _units()
    info = su.set_index("unit_id")
    ids = set(_wells_of(sc["unit_ids"]))
    master = _tables()["master"]
    master = master[master["well_id"].isin(ids)]
    uw = _units()[1].set_index("well_id")["unit_id"]
    mon_all = _monthly()
    mon_all = mon_all[mon_all["well_id"].isin(ids)]
    periods = []
    for a in dates:
        a, deck = _as_of(a.strip())
        start_ym, end_ym = workload.period_bounds(a, cfg["period_months"])
        comp = _composition_frame(sc["unit_ids"], a, deck, start_ym, start_ym)
        dec = workload.decline_rates(comp, start_ym, end_ym)
        per = mon_all[(mon_all["ym"] >= start_ym) & (mon_all["ym"] <= end_ym)]
        production = float(per["oil_t"].sum())
        plan = db.read_df("SELECT * FROM unit_plan_monthly WHERE ym >= ? AND ym <= ?", (start_ym, end_ym))
        plan = plan[plan["unit_id"].isin(sc["unit_ids"])]

        def water_cut(end: str) -> Optional[float]:
            lo = workload.idx_to_ym(workload.ym_to_idx(end) - 2)
            s_ = mon_all[(mon_all["ym"] >= lo) & (mon_all["ym"] <= end)]
            liquid = s_["water_m3"].sum() + s_["oil_t"].sum() / cfg["oil_density_t_per_m3"]
            return float(100 * s_["water_m3"].sum() / liquid) if liquid > 0 else None

        effects = [e for u in sc["unit_ids"] for e in _evaluate(u, a, deck, "sec")["measures"]
                   if e["in_period"] and e["status"] == "ok"]
        producing = per[per["days_on"] > 0]
        well_months = float(len(producing))
        opex_total = float(sum(
            n * economics.economic_limit_rate(deck, "sec", opex_factor=float(info.loc[u, "opex_factor"]))[
                "opex_per_well_month_usd"]
            for u, n in producing["well_id"].map(uw).value_counts().items()))
        values = sec_indicators.compute(
            production_t=production,
            plan_production_t=float(plan[["plan_new_oil_t", "plan_measure_inc_t", "plan_old_oil_t"]]
                                    .sum().sum()) if len(plan) else None,
            natural_decline_pct=dec["natural_annual_pct"],
            comprehensive_decline_pct=dec["comprehensive_annual_pct"],
            water_cut_pct=water_cut(end_ym),
            water_cut_prev_pct=water_cut(workload.idx_to_ym(workload.ym_to_idx(end_ym) - 12)),
            n_wells=int((master["first_prod_date"] <= a).sum()),
            n_producing=int(((mon_all["ym"] == end_ym) & (mon_all["days_on"] > 0)).sum()),
            new_oil_t=float(comp["new_oil_t"].sum()),
            plan_new_oil_t=float(plan["plan_new_oil_t"].sum()) if len(plan) else None,
            n_measures_evaluated=len(effects),
            n_measures_effective=sum(1 for e in effects if e["effective"]),
            pdp_t=float(sum(_evaluate(u, a, deck, "sec")["total_t"] for u in sc["unit_ids"])),
            producing_well_months=well_months,
            opex_per_well_month_usd=(opex_total / well_months) if well_months else 0.0,
            net_revenue_usd_per_t=economics.economic_limit_rate(deck)["net_revenue_usd_per_tonne"])
        periods.append(dict(as_of=a, period_start_ym=start_ym, period_end_ym=end_ym, values=values))
    return _scope_out(sc), periods


def _dates_arg(as_of_list) -> Tuple[str, ...]:
    dates = as_of_list.replace("，", ",").split(",") if isinstance(as_of_list, str) else list(as_of_list)
    return tuple(str(d).strip() for d in dates if str(d).strip())


def _score_periods(periods: List[Dict], spec: Dict) -> List[Dict]:
    out = []
    for p in periods:
        scored = sec_indicators.score(p["values"], spec)
        out.append(dict(as_of=p["as_of"], period_start_ym=p["period_start_ym"], period_end_ym=p["period_end_ym"],
                        indicators=[dict(r, value=_r(r["value"], 2), score=_r(r["score"], 1)) for r in scored["indicators"]],
                        groups=[dict(g, score=_r(g["score"], 1)) for g in scored["groups"]]))
    return out


SCORING_TEXT = ("得分 = (值 − 差值锚点) / (优值锚点 − 差值锚点) × 100，截断到 0~100；分组综合分为组内加权平均。")


def unit_indicators(scope: str, as_of_list="2025-12-31,2026-12-31", profile: Optional[str] = None,
                    trace_id: Optional[str] = None) -> Dict:
    """开发与经营指标及评分（运行监控），多个基准日并列对比；profile 为锚点方案 id，默认取默认方案。"""
    prof = _profile(profile)
    scope_out, periods = _indicator_periods(str(scope), _dates_arg(as_of_list))
    return dict(**_env(trace_id), scope=scope_out, periods=_score_periods(periods, prof["spec"]),
                profile=dict(profile_id=prof["profile_id"], name=prof["name"], is_builtin=prof["is_builtin"]),
                scoring=SCORING_TEXT + f"锚点方案：{prof['name']}",
                indicator_def_version=prof["spec"].get("version"))


def preview_indicator_scores(scope: str, spec: Dict, as_of_list="2025-12-31,2026-12-31",
                             trace_id: Optional[str] = None) -> Dict:
    """未保存锚点的实时预览：校验通过才打分，不写库。"""
    errors = sec_indicators.validate_spec(spec, _reference_spec())
    if errors:
        raise KernelError("锚点方案不合法：" + "；".join(errors))
    scope_out, periods = _indicator_periods(str(scope), _dates_arg(as_of_list))
    return dict(**_env(trace_id), scope=scope_out,
                periods=_score_periods(periods, sec_indicators.normalize_spec(spec, _reference_spec())),
                scoring=SCORING_TEXT + "（预览，未保存）")


# ---- 锚点方案管理（配置写入：走 HTTP 并记审计，不注册为智能体工具）----
def _reference_spec() -> Dict:
    spec = indicator_spec()
    return dict(version=spec.get("version"), indicators=spec["indicators"], groups=spec.get("groups", {}))


def _builtin_profiles() -> Dict[str, Dict]:
    spec, ref = indicator_spec(), _reference_spec()
    base = spec.get("base_profile") or dict(id="builtin", name="内置口径", description="")
    out = {base["id"]: dict(name=base["name"], description=base.get("description", ""), spec=ref)}
    for pid, pr in (spec.get("profiles") or {}).items():
        ind = {k: dict(v, **(pr.get("anchors") or {}).get(k, {})) for k, v in ref["indicators"].items()}
        out[pid] = dict(name=pr["name"], description=pr.get("description", ""), spec=dict(ref, indicators=ind))
    return out


def _ensure_profiles() -> None:
    _snapshot_ready()
    builtins = _builtin_profiles()
    have = db.read_df("SELECT profile_id, is_default FROM indicator_profile")
    with db.connect() as conn:
        for pid, pr in builtins.items():
            if pid not in set(have["profile_id"]):
                conn.execute("INSERT INTO indicator_profile (profile_id, name, description, spec_json, is_builtin, "
                             "is_default, updated_at) VALUES (?, ?, ?, ?, 1, 0, ?)",
                             (pid, pr["name"], pr["description"], json.dumps(pr["spec"], ensure_ascii=False), trace.now_iso()))
        if not int(have["is_default"].sum() if len(have) else 0):
            conn.execute("UPDATE indicator_profile SET is_default = CASE WHEN profile_id = ? THEN 1 ELSE 0 END",
                         (next(iter(builtins)),))


def _profile_row(r: Dict) -> Dict:
    return dict(profile_id=r["profile_id"], name=r["name"], description=r["description"] or "",
                is_builtin=bool(r["is_builtin"]), is_default=bool(r["is_default"]), updated_at=r["updated_at"],
                spec=json.loads(r["spec_json"]))


def _profile(profile_id: Optional[str] = None) -> Dict:
    _ensure_profiles()
    df = db.read_df("SELECT * FROM indicator_profile")
    hit = df[df["profile_id"] == profile_id] if profile_id else df[df["is_default"] == 1]
    if hit.empty:
        raise KernelError(f"锚点方案 {profile_id!r} 不存在，可选：{'、'.join(df['profile_id'])}")
    return _profile_row(_records(hit)[0])


def list_indicator_profiles(trace_id: Optional[str] = None) -> Dict:
    _ensure_profiles()
    rows = [_profile_row(r) for r in _records(db.read_df("SELECT * FROM indicator_profile ORDER BY is_builtin DESC, name"))]
    return dict(**_env(trace_id), profiles=rows, reference=_reference_spec(), scoring=SCORING_TEXT)


def save_indicator_profile(name: str, spec: Dict, profile_id: Optional[str] = None, description: str = "",
                           trace_id: Optional[str] = None) -> Dict:
    _ensure_profiles()
    name = str(name or "").strip()
    if not name:
        raise KernelError("请填写方案名称")
    if profile_id:
        cur = _profile(profile_id)
        if cur["is_builtin"]:
            raise KernelError("内置方案不可修改，请另存为新方案")
    errors = sec_indicators.validate_spec(spec, _reference_spec())
    if errors:
        raise KernelError("锚点方案不合法：" + "；".join(errors))
    norm = sec_indicators.normalize_spec(spec, _reference_spec())
    pid = profile_id or "custom_" + trace.new_trace_id("p")[2:12]
    tid = trace_id or trace.new_trace_id("cfg")
    with db.connect() as conn:
        conn.execute("INSERT INTO indicator_profile (profile_id, name, description, spec_json, is_builtin, is_default, updated_at) "
                     "VALUES (?, ?, ?, ?, 0, 0, ?) ON CONFLICT(profile_id) DO UPDATE SET name = excluded.name, "
                     "description = excluded.description, spec_json = excluded.spec_json, updated_at = excluded.updated_at",
                     (pid, name, str(description or ""), json.dumps(norm, ensure_ascii=False), trace.now_iso()))
    trace.audit(tid, "user", "indicator_profile:save", dict(profile_id=pid, name=name))
    return dict(**_env(tid), profile=_profile(pid))


def delete_indicator_profile(profile_id: str, trace_id: Optional[str] = None) -> Dict:
    cur = _profile(profile_id)
    if cur["is_builtin"]:
        raise KernelError("内置方案不可删除")
    tid = trace_id or trace.new_trace_id("cfg")
    with db.connect() as conn:
        conn.execute("DELETE FROM indicator_profile WHERE profile_id = ?", (profile_id,))
    _ensure_profiles()                                 # 删掉的是默认方案时，默认回到内置方案
    trace.audit(tid, "user", "indicator_profile:delete", dict(profile_id=profile_id))
    return dict(**_env(tid), deleted=profile_id)


def set_default_indicator_profile(profile_id: str, trace_id: Optional[str] = None) -> Dict:
    _profile(profile_id)
    tid = trace_id or trace.new_trace_id("cfg")
    with db.connect() as conn:
        conn.execute("UPDATE indicator_profile SET is_default = CASE WHEN profile_id = ? THEN 1 ELSE 0 END", (profile_id,))
    trace.audit(tid, "user", "indicator_profile:default", dict(profile_id=profile_id))
    return dict(**_env(tid), profile=_profile(profile_id))


# ---- 证实储量类别：PDP / PDNP / PUD ----
PUD_STATUS_CN = sec_composition.PUD_STATUS_CN
PDNP_STATUS_CN = {"booked": "计入 PDNP", "long_shut_in": "长停井，不计入", "uneconomic": "不足以覆盖复产费用，不计入",
                  "no_fit": "停产前资料不足，不计入", "no_history": "无生产记录"}


@cached_service
def unit_proved_categories(scope: str, as_of: str = "2026-12-31", scenario: str = "sec",
                           trace_id: Optional[str] = None) -> Dict:
    """证实储量类别：已开发已生产（PDP）、已开发未生产（PDNP，停产井）、未开发（PUD，部署井位），含逐井判定依据。"""
    sc = _scope(scope)
    as_of, deck = _as_of(as_of)
    sp = _scenario(deck, scenario)
    su, _ = _units()
    names = su.set_index("unit_id")["unit_name"]
    cfg = _su()
    evs = {u: _evaluate(u, as_of, deck, scenario) for u in sc["unit_ids"]}
    pdp = sum(ev["total_t"] for ev in evs.values())
    pdnp = sum(ev["pdnp"]["reserves_t"] for ev in evs.values())
    pud = sum(ev["pud"]["reserves_t"] for ev in evs.values())
    total = pdp + pdnp + pud
    n_pdp = sum(ev["old_base"]["n_evaluated"] + len(ev["new_wells"]) for ev in evs.values())
    n_pdnp = sum(ev["pdnp"]["n_booked"] for ev in evs.values())
    n_pud = sum(ev["pud"]["n_booked"] for ev in evs.values())
    share = lambda v: _r(100 * v / total, 1) if total else None
    wells, locs, pdnp_counts, pud_counts, warnings = [], [], {}, {}, []
    as_of_ym = as_of[:7]
    for u, ev in evs.items():
        for w in ev["pdnp"]["wells"]:
            pdnp_counts[w["pdnp_status"]] = pdnp_counts.get(w["pdnp_status"], 0) + 1
            wells.append(dict(well_code=_code(w["well_id"]), unit_id=u, last_prod_ym=w["last_prod_ym"],
                              shut_in_months=w["shut_in_months"], status=w["pdnp_status"],
                              status_cn=PDNP_STATUS_CN.get(w["pdnp_status"], w["pdnp_status"]),
                              reserves_t=_r(w["reserves_t"], 1), estimate_t=_r(w["estimate_t"], 1),
                              rate_at_shut_t_per_d=_r(w["rate_at_shut_t_per_d"], 2)))
        for l in ev["pud"]["locations"]:
            pud_counts[l["status"]] = pud_counts.get(l["status"], 0) + 1
            locs.append(dict(location_id=l["location_id"], unit_id=u, x_off=_r(l["x_off"], 1), y_off=_r(l["y_off"], 1),
                             planned_drill_ym=l["planned_drill_ym"], first_booked_as_of=l["first_booked_as_of"],
                             deadline_ym=l["deadline_ym"], years_to_deadline=_r(l["years_to_deadline"], 1),
                             n_producing_neighbors=l["n_producing_neighbors"], capex_wan=_r(l["capex_wan"], 1),
                             cash_wan=_r(l["cash_wan"], 1), estimate_t=_r(l["estimate_t"], 1),
                             reserves_t=_r(l["reserves_t"], 1), status=l["status"], status_cn=l["status_cn"],
                             new_booking=l["new_booking"],
                             drilled_well_code=_code(l["drilled_well_id"]) if l["drilled_well_id"] else None))
            if l["status"] == "booked" and l["years_to_deadline"] <= 1:
                warnings.append(dict(kind="pud_deadline", location_id=l["location_id"], unit_id=u,
                                     text=f"{l['location_id']}（{u}）距五年期限不足 1 年（期限 {l['deadline_ym']}），计划钻井 {l['planned_drill_ym']}"))
            elif l["status"] == "booked" and l["planned_drill_ym"] < as_of_ym:
                warnings.append(dict(kind="pud_overdue", location_id=l["location_id"], unit_id=u,
                                     text=f"{l['location_id']}（{u}）计划钻井 {l['planned_drill_ym']} 已过期，尚未钻井"))
    wells.sort(key=lambda w: (w["status"] != "booked", -(w["reserves_t"] or 0)))
    order = ["booked", "not_booked_yet", "expired_5yr", "beyond_5yr", "not_certain", "uneconomic", "no_type_curve",
             "drilled", "cancelled"]
    locs.sort(key=lambda l: (order.index(l["status"]) if l["status"] in order else 99, l["location_id"]))
    pc, pd_ = cfg["pud"], cfg["pdnp"]
    return dict(
        **_env(trace_id), scope=_scope_out(sc), as_of=as_of, scenario=scenario, scenario_label=sp["label"],
        price_deck_id=deck, total_proved_t=_r(total, 1),
        categories=[dict(key="PDP", name="已开发已生产（PDP）", reserves_t=_r(pdp, 1), share_pct=share(pdp), n_items=n_pdp,
                         basis="新-老-措构成合计"),
                    dict(key="PDNP", name="已开发未生产（PDNP）", reserves_t=_r(pdnp, 1), share_pct=share(pdnp), n_items=n_pdnp,
                         basis="停产井按停产前递减续算，扣除长停与不经济"),
                    dict(key="PUD", name="未开发（PUD）", reserves_t=_r(pud, 1), share_pct=share(pud), n_items=n_pud,
                         basis="部署井位按类型曲线折减取值，须满足连续性、五年规则与经济性")],
        units=[dict(unit_id=u, unit_name=names[u], PDP=_r(ev["total_t"], 1), PDNP=_r(ev["pdnp"]["reserves_t"], 1),
                    PUD=_r(ev["pud"]["reserves_t"], 1), total_t=_r(ev["total_proved_t"], 1)) for u, ev in evs.items()],
        pdnp=dict(n_booked=n_pdnp, reserves_t=_r(pdnp, 1), n_by_status=pdnp_counts, wells=wells,
                  rule=f"近 3 个月无产量的老井；停产不超过 {pd_['max_shut_in_months']} 个月，"
                       f"停产前数据递减续算的剩余可采须覆盖复产作业费"),
        pud=dict(n_booked=n_pud, reserves_t=_r(pud, 1), n_by_status=pud_counts, locations=locs,
                 rule=f"半径 {pc['radius_m']} m 内当期在产井不少于 {pc['min_producing_neighbors']} 口；"
                      f"首次入账后 {pc['horizon_years']} 年内钻井；类型曲线 × {pc['certainty_factor']} 截到经济极限，"
                      f"净现金流须覆盖钻完井投资"),
        warnings=warnings,
        note="PDNP 与 PUD 按单元各自的经济极限取值；采油厂/公司为下属单元相加")


@cached_service
def unit_category_tracking(scope: str, from_as_of: str = "2025-12-31", to_as_of: str = "2026-12-31",
                           scenario: str = "sec", trace_id: Optional[str] = None) -> Dict:
    """证实储量类别滚动：PUD 转化率与五年规则移出、PDNP 复产与新增停产，及 PDP 对账里的类别调整。"""
    sc = _scope(scope)
    fa, fdeck = _as_of(from_as_of)
    ta, tdeck = _as_of(to_as_of)
    if fa >= ta:
        raise KernelError(f"期初基准日 {fa} 须早于期末基准日 {ta}")
    _scenario(tdeck, scenario)
    agg: Dict[str, Dict] = {}
    lists: Dict[str, List] = {k: [] for k in ("converted", "expired", "removed", "new", "reactivated", "pdnp_removed", "pdnp_new")}
    for u in sc["unit_ids"]:
        rf = sec_composition.category_rollforward(_evaluate(u, fa, fdeck, scenario), _evaluate(u, ta, tdeck, scenario))
        for cat in ("pud", "pdnp"):
            if cat not in rf:
                continue
            a = agg.setdefault(cat, dict(table={}, counts={}))
            for row in rf[cat]["table"]:
                t = a["table"].setdefault(row["key"], dict(key=row["key"], item=row["item"], value=0.0))
                t["value"] += row["value"]
            for k, v in rf[cat].items():
                if k.startswith("n_"):
                    a["counts"][k] = a["counts"].get(k, 0) + v
        if "pud" in rf:
            for k in ("converted", "expired", "removed", "new"):
                lists[k] += [dict(location_id=x, unit_id=u) for x in rf["pud"][k]]
        if "pdnp" in rf:
            for k, dst in (("reactivated", "reactivated"), ("removed", "pdnp_removed"), ("new", "pdnp_new")):
                lists[dst] += [dict(well_code=_code(w), unit_id=u) for w in rf["pdnp"][k]]

    def finish(cat: str) -> Optional[Dict]:
        if cat not in agg:
            return None
        rows = [dict(r, value=_r(r["value"], 1)) for r in agg[cat]["table"].values()]
        vals = {r["key"]: r["value"] for r in rows}
        diff = (vals["closing"] or 0) - (vals["opening"] or 0) - sum((r["value"] or 0) for r in rows if r["key"] not in ("opening", "closing"))
        return dict(table=rows, balanced=abs(diff) <= 1.0, difference=_r(diff, 1), **agg[cat]["counts"])

    pud, pdnp = finish("pud"), finish("pdnp")
    if pud:
        pud["conversion_rate_pct"] = _r(100 * pud["n_converted"] / pud["n_opening"], 1) if pud["n_opening"] else None
        pud["years_to_convert_at_pace"] = _r(pud["n_closing"] / pud["n_converted"], 1) if pud["n_converted"] else None
        pud.update(converted=lists["converted"], expired=lists["expired"], removed=lists["removed"], new=lists["new"])
    if pdnp:
        pdnp.update(reactivated=lists["reactivated"], removed=lists["pdnp_removed"], new=lists["pdnp_new"])
    recon = unit_reconcile(scope, from_as_of=fa, to_as_of=ta, scenario=scenario)
    return dict(**_env(trace_id), scope=_scope_out(sc), from_as_of=fa, to_as_of=ta, scenario=scenario,
                pud=pud, pdnp=pdnp, pdp_category_change=recon["category_change_detail"],
                note="PUD 转化率 = 期初已入账井位中本期钻井转为已开发的比例；按目前节奏消化期末 PUD 所需年数 = 期末入账井位数 / 本期转化数。"
                     "五年规则：井位须在首次入账后五年内钻井，逾期未钻即移出")


# ---- 折耗与减值 ----
@functools.lru_cache(maxsize=1)
def _asset_book() -> Dict[Tuple[str, str], Dict]:
    _snapshot_ready()
    try:
        df = db.read_df("SELECT * FROM unit_asset_book")
    except Exception:
        return {}
    return {(r["unit_id"], str(r["as_of"])): r for r in _records(df)}


@functools.lru_cache(maxsize=64)
def _unit_depletion_chain(unit_id: str, as_of: str) -> Dict:
    """单个单元一期的折耗与减值。期初净值缺失时取上一评估期的期末净值（逐期滚动）。"""
    dates = [str(d) for d in _su()["evaluation_dates"]]
    k = dates.index(as_of)
    fin = _su()["finance"]
    book = _asset_book().get((unit_id, as_of))
    if book is None:
        raise KernelError(f"缺少 {unit_id} 在 {as_of} 的资产账面记录，请在 系统 · 数据导入 导入单元资产账面价值")
    if book["opening_nbv_wan"] is not None:
        opening, source = float(book["opening_nbv_wan"]), "财务台账"
    elif k > 0:
        opening, source = _unit_depletion_chain(unit_id, dates[k - 1])["closing_nbv_wan"], f"上期（{dates[k - 1]}）期末净值滚动"
    else:
        raise KernelError(f"{unit_id} 首个评估期 {as_of} 缺少期初资产净值")
    deck = economics.deck_for(as_of)
    su, _ = _units()
    opex_factor = float(su.set_index("unit_id").loc[unit_id, "opex_factor"])
    ev = _evaluate(unit_id, as_of, deck, "sec")
    pd_t = float(ev["total_t"] + ev["pdnp"]["reserves_t"])
    dep = finance.depletion(opening, float(book["capex_additions_wan"]), float(ev["production_in_period_t"]), pd_t)
    try:
        ev_imp = _evaluate(unit_id, as_of, deck, "impairment")
        econ = economics.economic_limit_rate(deck, "impairment", opex_factor=opex_factor)
    except KeyError as exc:
        raise KernelError(str(exc.args[0])) from None
    pd_imp = float(ev_imp["total_t"] + ev_imp["pdnp"]["reserves_t"])
    end = as_of[:7]
    lo = workload.idx_to_ym(workload.ym_to_idx(end) - 2)
    mon = _monthly()
    recent = mon[mon["well_id"].isin(set(_wells_of([unit_id]))) & (mon["ym"] >= lo) & (mon["ym"] <= end)]
    monthly_rate = float(recent["oil_t"].sum()) / 3.0
    well_months = float((recent["days_on"] > 0).sum()) / 3.0
    opex_t = well_months * float(econ["opex_per_well_month_usd"]) / monthly_rate if monthly_rate > 0 else 0.0
    rec = finance.recoverable_amount(pd_imp, monthly_rate, float(econ["net_revenue_usd_per_tonne"]), opex_t,
                                     float(fin["discount_rate"]), float(fin["fx_cny_per_usd"]))
    imp = finance.impairment(dep["carrying_after_depletion_wan"], rec["recoverable_wan"])
    return dict(unit_id=unit_id, as_of=as_of, opening_nbv_wan=opening, opening_source=source,
                capex_additions_wan=float(book["capex_additions_wan"]), production_t=float(ev["production_in_period_t"]),
                pdp_t=float(ev["total_t"]), pdnp_t=float(ev["pdnp"]["reserves_t"]), pd_reserves_t=pd_t,
                depletion_rate_pct=100 * dep["depletion_rate"] if dep["depletion_rate"] is not None else None,
                depletion_wan=dep["depletion_wan"], carrying_wan=dep["carrying_after_depletion_wan"],
                impairment_price_usd_bbl=float(econ["price_usd_bbl"]), pd_reserves_impairment_t=pd_imp,
                monthly_rate_t=monthly_rate, opex_usd_per_t=opex_t, net_revenue_usd_per_t=float(econ["net_revenue_usd_per_tonne"]),
                cash_usd_per_t=rec["cash_usd_per_t"], life_years=rec["life_years"], recoverable_wan=rec["recoverable_wan"],
                recoverable_note=rec["note"], **imp)


@cached_service
def unit_depletion_impairment(scope: str, as_of: str = "2026-12-31", trace_id: Optional[str] = None) -> Dict:
    """产量法折耗额与减值测试：期初净值 + 本期投入 → 折耗 → 折耗后账面价值与减值测试价下的可收回金额比较。"""
    sc = _scope(scope)
    as_of, deck = _as_of(as_of)
    su, _ = _units()
    names = su.set_index("unit_id")["unit_name"]
    fin = _su()["finance"]
    rows = [_unit_depletion_chain(u, as_of) for u in sc["unit_ids"]]
    tot = {k: float(sum(r[k] for r in rows)) for k in ("opening_nbv_wan", "capex_additions_wan", "production_t",
                                                     "pd_reserves_t", "depletion_wan", "carrying_wan", "recoverable_wan",
                                                     "impairment_wan", "closing_nbv_wan")}
    rate = 100 * tot["production_t"] / (tot["pd_reserves_t"] + tot["production_t"]) if tot["pd_reserves_t"] + tot["production_t"] > 0 else None
    rnd = lambda r: {k: (_r(v, 2 if k.endswith("_pct") or k.endswith("_per_t") or k.endswith("bbl") or k == "life_years" else 1)
                         if isinstance(v, float) else v) for k, v in r.items()}
    return dict(**_env(trace_id), scope=_scope_out(sc), as_of=as_of, price_deck_id=deck, currency="万元",
                summary=dict(rnd(tot), depletion_rate_pct=_r(rate, 2), n_units=len(rows),
                             n_impaired=sum(1 for r in rows if r["impaired"]),
                             headroom_pct=_r(100 * (tot["recoverable_wan"] - tot["carrying_wan"]) / tot["carrying_wan"], 1)
                             if tot["carrying_wan"] > 0 else None),
                units=[dict(rnd(r), unit_name=names[r["unit_id"]]) for r in rows],
                assumptions=dict(fx_cny_per_usd=fin["fx_cny_per_usd"], discount_rate=fin["discount_rate"],
                                 impairment_scenario=economics.scenario_params(deck, "impairment")["label"]),
                method=["产量法折耗率 = 本期产量 /（期末证实已开发储量 PDP + PDNP + 本期产量），储量取 SEC 价口径",
                        "折耗额 =（期初资产净值 + 本期资本化投入）× 折耗率",
                        "可收回金额 = 减值测试价下证实已开发储量的未来净现金流折现值：单元指数递减剖面，初始月产取近 3 个月月均，"
                        "吨油净现金流 = 吨油净收入 − 吨油操作成本",
                        "减值额 = max(0，折耗后账面价值 − 可收回金额)；减值后期末净值作为下一期期初"],
                note="简化口径：不含 PUD 未来投资与弃置费，剖面为单元级指数递减；采油厂/公司为下属单元相加（减值按单元逐个测试后相加）")


def persist_unit_evaluation(as_of: str, scenarios=SCENARIOS) -> Dict:
    """把一期全部单元、全部价格情景的构成评估结果写入 sec_eval_record（历史评估成果管理）。

    这是写操作，只给 CLI 用，不注册为智能体工具、不开放 HTTP 接口。
    """
    as_of, deck = _as_of(as_of)
    su, _ = _units()
    env = _env(trace.new_trace_id("eval"))
    rows, summary = [], []
    for u in su["unit_id"]:
        for s in scenarios:
            ev = _evaluate(u, as_of, deck, s)
            detail = json.dumps(dict(price_deck_id=deck, q_econ_well_t_per_d=ev["q_econ_well_t_per_d"],
                                     period_start_ym=ev["period_start_ym"]), ensure_ascii=False)
            for c in ev["components"] + [dict(key="total", reserves_t=ev["total_t"],
                                              n_items=ev["n_wells"], basis="四项之和")]:
                rows.append(dict(as_of=as_of, unit_id=u, scenario=s, component=c["key"],
                                 reserves_t=round(float(c["reserves_t"]), 1), n_wells=int(c["n_items"]),
                                 method=c["basis"], detail_json=detail,
                                 model_version=env["model_version"], trace_id=env["trace_id"],
                                 created_at=trace.now_iso()))
            summary.append(dict(unit_id=u, scenario=s, total_t=round(ev["total_t"], 1),
                                components={c["key"]: round(c["reserves_t"], 1) for c in ev["components"]}))
    db.replace_rows(pd.DataFrame(rows), "sec_eval_record", ["as_of", "unit_id", "scenario", "component"])
    trace.audit(env["trace_id"], "kernel", "persist_unit_evaluation",
                dict(as_of=as_of, scenarios=list(scenarios), n_records=len(rows)))
    for w in _CACHED:                      # 入库后对账的期初来源会变，清掉服务结果缓存
        w.cache_clear()
    return dict(trace_id=env["trace_id"], as_of=as_of, price_deck_id=deck,
                n_records=len(rows), summary=summary)


def warm_unit_evaluations() -> Dict:
    """预热页面最常用的评估：最近两期、SEC 价口径的全部单元，外加对账要用的"期末数据 + 期初价格册"。

    快照在库里时几乎是瞬时的；不在时逐井重算并写入快照，下次重启就不用再算。
    """
    su, _ = _units()
    dates = [str(d) for d in _su()["evaluation_dates"]][-2:]
    jobs = [(d, economics.deck_for(d)) for d in dates]
    if len(dates) == 2:
        jobs.append((dates[1], economics.deck_for(dates[0])))
    n_snapshot = n_computed = 0
    for u in su["unit_id"]:
        for a, deck in jobs:
            if (u, a, deck, "sec") in _EVAL_MEMO:
                continue
            if _snapshot_get(u, a, deck, "sec") is not None:
                n_snapshot += 1
            else:
                n_computed += 1
            _evaluate(u, a, deck, "sec")
    return dict(n_from_snapshot=n_snapshot, n_computed=n_computed)
