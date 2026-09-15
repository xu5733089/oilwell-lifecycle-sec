"""算法内核服务层（方案 §2.1 的 L2）。

**全部数值的唯一产地。** 智能体、HTTP 接口、CLI、报告生成都只能调这里，
任何一层都不许自己算数。每个返回体都带 model_version / label_def_version /
data_source / trace_id 四个追溯字段。
"""
from __future__ import annotations

import functools
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import db, trace
from ..config import config, indicators as indicator_spec, path
from ..features.build import align, build, encode, feature_columns
from ..models import registry
from ..models.analog_retrieval import _resample_curve
from ..models.attribution import local_attribution
from ..reserves import crosscheck, dca, volumetric, workload
from ..sec import checklist as sec_checklist
from ..sec import classify as sec_classify
from ..sec import composition as sec_composition
from ..sec import economics
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
    for f in (_units, _monthly, _events, _codes, _new_wells, _new_well_model_estimates, _evaluate):
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


@cached_service
def predict_lifecycle(well_code: str, obs_days: Optional[int] = None,
                      targets: Optional[List[str]] = None,
                      trace_id: Optional[str] = None) -> Dict:
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
    X = align(X, b["feature_cols"])

    preds = b["model"].predict(X)
    conf = b["conformal"]
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

    curve = None
    if {"q_peak", "t_peak", "eur"} <= set(out):
        curve = dca.synthesize_curve(out["q_peak"]["p50"], out["t_peak"]["p50"],
                                     out["eur"]["p50"])

    shap_like = []
    if "t_peak" in b["model"].models:
        shap_like = local_attribution(b["model"], X.iloc[0], "t_peak", 0.5, top_k=6)

    return dict(**_env(trace_id), well_code=w["well_code_anon"], obs_days=obs_days,
                cum_to_date_t=_r(cum), results=out, curve=curve,
                explain=dict(attribution_target="t_peak", top_features=shap_like,
                             method="特征消融归因（leave-one-covariate-out）",
                             calibration=dict(alpha=conf.alpha,
                                              delta={k: _r(v) for k, v in conf.delta.items()},
                                              test_coverage=b.get("coverage", {}))),
                note="P10/P50/P90 为分位数本身（p10 为数值小者）；储量口径下 P90 为低估计。")


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


@cached_service
def fit_dca(well_code: str, model: str = "auto", d_min_year: float = 0.075,
            price_deck_id: str = DEFAULT_PRICE_DECK,
            trace_id: Optional[str] = None) -> Dict:
    w = _resolve(well_code)
    p = _prod(w["well_id"])
    on = p[(p["hours_on"] > 0) & (p["oil_t"] > 0)]
    if len(on) < 90:
        raise KernelError(f"井 {w['well_code_anon']} 有效生产不足 90 天，无法可靠拟合递减曲线")

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
    for key, fname in (("agent", "eval_agent.json"), ("algo", "eval_algo.json")):
        f = path("artifacts_dir") / fname
        out[key] = _json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    out["hint"] = "为空表示尚未运行：python -m src.cli eval-agent / eval-algo"
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


@functools.lru_cache(maxsize=256)
def _evaluate(unit_id: str, as_of: str, deck: str, scenario: str) -> Dict:
    """单元评估（内核结果原样缓存，调用方不得修改返回的 dict）。

    deck 与 as_of 分开传：价格修订要用"期末的数据 + 期初的价格册"重算一次。
    """
    su, _ = _units()
    u = su[su["unit_id"] == unit_id].iloc[0]
    ids = set(_wells_of([unit_id]))
    master = _tables()["master"]
    mon = _monthly()
    cfg = _su()
    qe = economics.economic_limit_rate(deck, scenario, opex_factor=float(u["opex_factor"]))["q_econ"]
    return sec_composition.evaluate_unit(
        monthly=mon[mon["well_id"].isin(ids)], master=master[master["well_id"].isin(ids)],
        events=_events(), new_wells=_new_wells(as_of), as_of=as_of, q_econ_well=qe,
        period_months=cfg["period_months"], measure_cfg=cfg["measure"],
        min_fit_months=cfg["decline"]["min_fit_months"],
        type_curve_min_history=cfg["type_curve"]["min_history_months"],
        model_estimates=_new_well_model_estimates(as_of), fit_cache=_FIT_CACHE)


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
    return vals


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

    # 历史评估成果管理：上期结果已入库就用入库值做期初（差额进技术修订），否则按上期数据截面重算
    stored = db.read_df("SELECT unit_id, reserves_t FROM sec_eval_record WHERE as_of = ? "
                        "AND scenario = ? AND component = 'total'", (fa, scenario))
    stored = stored[stored["unit_id"].isin(sc["unit_ids"])].set_index("unit_id")["reserves_t"]
    opening_source = "入库记录" if len(stored) == len(sc["unit_ids"]) else "按上期数据截面重算"
    keys = [k for k, _, _ in sec_reconcile.ROWS]
    values = {k: 0.0 for k in keys}
    closing = 0.0
    for u, v in per_unit.items():
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
                depletion_rate_pct=_r(100 * prod / (closing + prod), 2) if closing + prod > 0 else None,
                depletion_note="产量法折耗率 = 本期产量 /（期末已证实已开发储量 + 本期产量）；"
                               "折耗额还需资产账面价值，本平台不掌握",
                note="技术修订为轧差项（与 20-F 的 revisions of previous estimates 口径一致）；"
                     "合成数据不含复产与 PUD 转 PDP 记录，类别调整计 0")


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
                "wells_rate_change": lambda e: e.get("rate_change_t_per_d") or 0}
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


@cached_service
def unit_indicators(scope: str, as_of_list="2025-12-31,2026-12-31",
                    trace_id: Optional[str] = None) -> Dict:
    """开发与经营指标及评分（运行监控雷达图），多个基准日并列对比。"""
    sc = _scope(scope)
    dates = as_of_list.replace("，", ",").split(",") if isinstance(as_of_list, str) else list(as_of_list)
    cfg = _su()
    su, _ = _units()
    info = su.set_index("unit_id")
    ids = set(_wells_of(sc["unit_ids"]))
    master = _tables()["master"]
    master = master[master["well_id"].isin(ids)]
    uw = _units()[1].set_index("well_id")["unit_id"]
    mon_all = _monthly()
    mon_all = mon_all[mon_all["well_id"].isin(ids)]
    spec = indicator_spec()
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
            s = mon_all[(mon_all["ym"] >= lo) & (mon_all["ym"] <= end)]
            liquid = s["water_m3"].sum() + s["oil_t"].sum() / cfg["oil_density_t_per_m3"]
            return float(100 * s["water_m3"].sum() / liquid) if liquid > 0 else None

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
        scored = sec_indicators.score(values, spec)
        periods.append(dict(as_of=a, period_start_ym=start_ym, period_end_ym=end_ym,
                            indicators=[dict(r, value=_r(r["value"], 2), score=_r(r["score"], 1))
                                        for r in scored["indicators"]],
                            groups=[dict(g, score=_r(g["score"], 1)) for g in scored["groups"]]))
    return dict(**_env(trace_id), scope=_scope_out(sc), periods=periods,
                scoring="得分 = (值 − 差值锚点) / (优值锚点 − 差值锚点) × 100，截断到 0~100；"
                        "分组综合分为组内加权平均。锚点见 conf/indicators.yaml",
                indicator_def_version=spec.get("version"))


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
