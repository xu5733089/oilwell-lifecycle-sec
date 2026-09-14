"""算法内核服务层（方案 §2.1 的 L2）。

**全部数值的唯一产地。** 智能体、HTTP 接口、CLI、报告生成都只能调这里，
任何一层都不许自己算数。每个返回体都带 model_version / label_def_version /
data_source / trace_id 四个追溯字段。
"""
from __future__ import annotations

import functools
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import db, trace
from ..config import config, path
from ..features.build import align, build, encode, feature_columns
from ..models import registry
from ..models.analog_retrieval import _resample_curve
from ..models.attribution import local_attribution
from ..reserves import crosscheck, dca, volumetric
from ..sec import checklist as sec_checklist
from ..sec import classify as sec_classify
from ..sec import economics
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
    for w in _CACHED:
        w.cache_clear()


@functools.lru_cache(maxsize=1)
def _tables():
    return dict(master=db.read_df("SELECT * FROM well_master"),
                static=db.read_df("SELECT * FROM geo_static"),
                labels=db.read_df("SELECT * FROM lifecycle_label"))


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
    for tgt, dfp in preds.items():
        if tgt not in want:
            continue
        lo, hi = conf.apply(tgt, dfp["p10"].to_numpy(float), dfp["p90"].to_numpy(float))
        out[tgt] = dict(p10=_r(lo[0]), p50=_r(dfp["p50"].iloc[0]), p90=_r(hi[0]),
                        unit=units.get(tgt, ""))

    curve = None
    if {"q_peak", "t_peak", "eur"} <= set(out):
        curve = dca.synthesize_curve(out["q_peak"]["p50"], out["t_peak"]["p50"],
                                     out["eur"]["p50"])

    shap_like = []
    if "t_peak" in b["model"].models:
        shap_like = local_attribution(b["model"], X.iloc[0], "t_peak", 0.5, top_k=6)

    return dict(**_env(trace_id), well_code=w["well_code_anon"], obs_days=obs_days,
                results=out, curve=curve,
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
                      t_econ_month=econ["t_econ_month"],
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
