"""递减模型回测：经验 Arps 与 带物理约束的递减模型（LF-BDF + 类比先验 + EUR 上限）并排比。

    python -m src.cli eval-dca                          # 主合成数据集（默认）
    python -m src.cli eval-dca --dataset linear_flow    # 线性流验证队列（复合平板解析解生成，带真值）
    python -m src.cli eval-dca --dataset ndic           # 北达科他州公开产量数据（真实井，无 EUR 真值）

统一回测协议（全部数据集相同，不看未来）：
  · 井按投产先后交错分成两半：A 半做"类比井库"（全历史拟合，得到各区域参数先验），B 半做回测井；
  · 回测井只截取峰后前 H 个月（H = 6 / 12 / 24）拟合，外推：
      1) 未来 12 个月产量（开井日均口径 × 30.4），与实测比；历史够长时再比未来 24 个月；
      2) EUR，与真值比（真实数据没有真值，这一项缺省）；
  · 关注三件事：中位绝对误差、中位偏差、**高估超过 20% 的井占比** —— SEC 已证实储量最怕高估。

**结论由数据算出来，不预先写死。** 三个数据集的"出题方式"不同：主合成数据的递减按 Arps 生成（偏向 Arps），
线性流队列按复合平板解析解生成（与两种拟合公式都不同），NDIC 是真实井 —— 三者并看才公平。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import db
from ..config import config, path
from ..reserves import dca, physics_dca as PD

DATASETS = {
    "synthetic": dict(label="主合成数据集（递减按 Arps 生成，带真值）", artifact="eval_dca.json",
                      tail="主合成数据的递减本身按 Arps 生成，长历史上经验 Arps 天然占优；物理约束与类比先验的价值集中在历史短、外推远的井上。"),
    "linear_flow": dict(label="线性流验证队列（复合平板恒压解析解生成，带真值）", artifact="eval_dca_linear_flow.json",
                        tail="该队列按改造区 + 外围基质的复合平板解析解生成（线性流 → 边界控制流 → 长尾），与经验 Arps、物理约束模型的公式都不同。"),
    "ndic": dict(label="北达科他州公开产量数据（Bakken / Three Forks 真实井，无 EUR 真值）", artifact="eval_dca_ndic.json",
                 tail="真实井没有 EUR 真值，只比未来产量；未来 24 个月的误差最接近“外推是否高估”的问题。"),
}


def method_labels(with_cap: bool) -> Dict[str, str]:
    return {"arps": "经验 Arps（多模型择优）", "physics_wide": "物理约束 · 宽先验",
            "physics_analog": "物理约束 · 类比先验 + EUR 上限" if with_cap else "物理约束 · 类比先验"}


METHODS = method_labels(True)


def _truth() -> Dict[str, float]:
    """合成数据的 EUR 真值；真实数据没有这张表，EUR 列自动缺省。"""
    f = path("data_dir") / "synth_truth.csv"
    if not f.exists():
        return {}
    t = pd.read_csv(f)
    return dict(zip(t["well_id"], t["eur"]))


def _summ(err: pd.Series) -> Dict:
    err = err.replace([np.inf, -np.inf], np.nan).dropna()
    if err.empty:
        return dict(n=0)
    return dict(n=int(len(err)), mdape_pct=round(float(err.abs().median() * 100), 1),
                bias_pct=round(float(err.median() * 100), 1),
                over20_pct=round(float((err > 0.2).mean() * 100), 1),
                p90_pct=round(float(err.quantile(0.9) * 100), 1))


# --------------------------------------------------------------------------- #
# 数据集 → 井列表。每口井：well_id, well_code, block, order_key, months, rates, cum_to(H), eur_true, cap()
def _synthetic_wells() -> Dict:
    from ..api import services as S
    from ..synth.well_generator import Q_ECON_SYNTH

    truth = _truth()
    T = S._tables()
    lab, m = T["labels"], T["master"]
    rows = (lab[lab["label_quality"] == "ok"][["well_id", "t_peak"]]
            .merge(m[["well_id", "well_code_anon", "block", "first_prod_date"]], on="well_id"))
    prod = db.read_df("SELECT well_id, day_index, oil_t, hours_on FROM prod_daily")
    groups = dict(tuple(prod.groupby("well_id")))
    wells = []
    for _, w in rows.iterrows():
        p = groups[w["well_id"]]
        post = p[p["day_index"] >= w["t_peak"]]
        mo, ra = PD.monthly_series(post["day_index"].to_numpy() - w["t_peak"] + 1,
                                   post["oil_t"].to_numpy(), post["hours_on"].to_numpy())

        def cum_to(H, p=p, tp=float(w["t_peak"])):
            return float(p[p["day_index"] <= tp + H * 30.4]["oil_t"].sum())

        def cap(code=w["well_code_anon"], block=w["block"]):
            try:
                rf = S._rf_reference(block)
                ooip = S.estimate_reserves_volumetric(code, mc_samples=2000)["ooip_t"]["p50"]
                return float(ooip) * float(rf["high_pct"]) / 100.0
            except Exception:
                return None

        wells.append(dict(well_id=w["well_id"], well_code=w["well_code_anon"], block=w["block"],
                          order_key=str(w["first_prod_date"]), months=mo, rates=ra, cum_to=cum_to,
                          eur_true=truth.get(w["well_id"]), cap=cap))
    q_econ = Q_ECON_SYNTH if truth else S.economics.economic_limit_rate(S.DEFAULT_PRICE_DECK)["q_econ"]
    return dict(wells=wells, q_econ=q_econ, truth=bool(truth), with_cap=True)


def _linear_flow_wells() -> Dict:
    from ..synth import linear_flow_cohort as LF
    wells = [dict(w, cum_to=(lambda H, vol=w["volumes"]: float(vol[: int(H)].sum())), cap=None) for w in LF.generate()]
    return dict(wells=wells, q_econ=LF.Q_ECON, truth=True, with_cap=False)


def _ndic_wells(max_wells: Optional[int]) -> Dict:
    from ..ingest import ndic_public
    wells = ndic_public.hindcast_wells(max_wells=max_wells)
    for w in wells:
        w["cum_to"] = (lambda H, vol=w["volumes"], pre=w.get("cum_before_peak_t", 0.0): float(pre + vol[: int(H)].sum()))
        w["cap"] = None
    return dict(wells=wells, q_econ=ndic_public.Q_ECON_T_PER_D, truth=False, with_cap=False)


# --------------------------------------------------------------------------- #
def _hindcast(wells: List[Dict], horizons: List[int], q_econ: float, min_analogs: int):
    wells = sorted(wells, key=lambda w: w["order_key"])
    library, evalset = wells[::2], wells[1::2]
    lib_fits: Dict[str, List] = {}
    for w in library:
        mo, ra = w["months"], w["rates"]
        if len(mo) >= 24:
            lib_fits.setdefault(w["block"], []).append(PD.fit(mo - mo[0], ra, q_econ=q_econ, cum_to_now=0.0))
    priors = {b: PD.analog_prior(v, min_analogs) for b, v in lib_fits.items()}

    rows = []
    for w in evalset:
        months, rates = np.asarray(w["months"], float), np.asarray(w["rates"], float)
        if len(months) == 0 or months.max() < 20:
            continue
        cap = w["cap"]() if w.get("cap") else None
        for H in horizons:
            sel, fut = months < H, (months > H) & (months < H + 12)
            fut24 = (months > H) & (months < H + 24)
            if sel.sum() < 5 or fut.sum() < 8:
                continue
            tt, t_now = months - months[0], H - months[0]
            has24 = bool(fut24.sum() >= 18 and months.max() >= H + 23)
            cum_to = float(w["cum_to"](H))
            r = dict(well_id=w["well_id"], well_code=w["well_code"], block=w["block"], H=H, cum_to_t=cum_to,
                     actual_next12_t=float(np.sum(rates[fut]) * 30.4),
                     actual_next24_t=float(np.sum(rates[fut24]) * 30.4) if has24 else np.nan,
                     eur_true=w.get("eur_true"))
            fa = dca.fit_best(tt[sel], rates[sel], d_min_year=0.075)
            r["arps_next12_t"] = float(np.sum(dca.rate(tt[fut], fa)) * 30.4)
            r["arps_next24_t"] = float(np.sum(dca.rate(tt[fut24], fa)) * 30.4) if has24 else np.nan
            r["arps_eur"] = cum_to + dca.eur(fa, q_econ, t_start_month=t_now)["remaining"]
            for name, prior, eur_cap in (("physics_wide", None, None),
                                         ("physics_analog", priors.get(w["block"]), cap)):
                f = PD.fit(tt[sel], rates[sel], q_econ=q_econ, cum_to_now=cum_to, eur_cap_t=eur_cap, prior=prior)
                r[f"{name}_next12_t"] = float(np.sum(PD.rate(tt[fut], f)) * 30.4)
                r[f"{name}_next24_t"] = float(np.sum(PD.rate(tt[fut24], f)) * 30.4) if has24 else np.nan
                r[f"{name}_eur"] = cum_to + PD.remaining(f, q_econ, t_now)[0]
                r[f"{name}_cap_binding"] = bool(f.cap_binding)
            rows.append(r)
    return pd.DataFrame(rows), lib_fits, priors


def _summarize(df: pd.DataFrame, horizons: List[int], truth: bool, labels: Dict[str, str], dataset: str) -> Dict:
    table = []
    for H in horizons:
        d = df[df["H"] == H] if len(df) else df
        for key, label in labels.items():
            if d.empty:
                continue
            rec = dict(horizon_months=H, method=key, method_cn=label, n_wells=int(len(d)),
                       next12=_summ(d[f"{key}_next12_t"] / d["actual_next12_t"] - 1.0))
            d24 = d[d["actual_next24_t"].notna()]
            if len(d24):
                rec["next24"] = _summ(d24[f"{key}_next24_t"] / d24["actual_next24_t"] - 1.0)
            if truth and d["eur_true"].notna().any():
                rec["eur"] = _summ(d[f"{key}_eur"] / d["eur_true"] - 1.0)
            if f"{key}_cap_binding" in d:
                rec["cap_binding_pct"] = round(float(d[f"{key}_cap_binding"].mean() * 100), 1)
            table.append(rec)

    metric, metric_cn = ("eur", "EUR") if truth else ("next24", "未来 24 个月产量")
    worst = []
    for H in horizons if len(df) else []:
        d = df[df["H"] == H]
        d = d[d["eur_true"].notna()] if truth else d[d["actual_next24_t"].notna()]
        pred, true = ("{}_eur", "eur_true") if truth else ("{}_next24_t", "actual_next24_t")
        err = {k: (d[pred.format(k)] / d[true] - 1.0).replace([np.inf, -np.inf], np.nan) for k in ("arps", "physics_analog")}
        for key, other in (("arps", "physics_analog"), ("physics_analog", "arps")):
            e = err[key].dropna()
            for i in e.sort_values(ascending=False).head(5).index:
                worst.append(dict(horizon_months=H, method=key, method_cn=labels[key], well_code=d.loc[i, "well_code"],
                                  block=d.loc[i, "block"], metric=metric_cn,
                                  eur_true=round(float(d.loc[i, true]), 1), eur_pred=round(float(d.loc[i, pred.format(key)]), 1),
                                  error_pct=round(float(e.loc[i] * 100), 1), other_method_cn=labels[other],
                                  other_error_pct=round(float(err[other].loc[i] * 100), 1)))

    def pick(H, key, field):
        return next((r.get(metric, {}).get(field) for r in table
                     if r["horizon_months"] == H and r["method"] == key), None)

    lines = []
    for H in horizons:
        a, ph = pick(H, "arps", "over20_pct"), pick(H, "physics_analog", "over20_pct")
        ea, ep = pick(H, "arps", "mdape_pct"), pick(H, "physics_analog", "mdape_pct")
        if a is None or ph is None:
            continue
        better = "物理约束更稳" if (ph < a and (ep or 0) <= (ea or 0) * 1.1) else \
            "经验 Arps 更准" if (ep or 0) > (ea or 0) * 1.1 else "两者相当"
        lines.append(f"峰后 {H} 个月：{metric_cn}中位误差 Arps {ea}% / 物理约束 {ep}%，"
                     f"高估超 20% 的井 Arps {a}% / 物理约束 {ph}% —— {better}")
    conclusion = ("；".join(lines) + "。" + DATASETS[dataset]["tail"]) if lines else "样本不足，未得出结论。"
    return dict(table=table, worst_overestimates=worst, conclusion=conclusion, metric=metric)


def run(horizons: List[int] | None = None, dataset: str = "synthetic", max_wells: Optional[int] = None) -> Dict:
    if dataset not in DATASETS:
        raise ValueError(f"未知数据集 {dataset!r}，可选 {list(DATASETS)}")
    cfg = config().get("physics_dca", {})
    horizons = horizons or [int(h) for h in cfg.get("hindcast_horizons", [6, 12, 24])]
    src = {"synthetic": _synthetic_wells, "linear_flow": _linear_flow_wells,
           "ndic": lambda: _ndic_wells(max_wells)}[dataset]()
    labels = method_labels(src["with_cap"])
    df, lib_fits, priors = _hindcast(src["wells"], horizons, src["q_econ"], int(cfg.get("min_analogs", 5)))
    summary = _summarize(df, horizons, src["truth"], labels, dataset)
    out = dict(dataset=dataset, dataset_label=DATASETS[dataset]["label"], horizons=horizons, q_econ=src["q_econ"],
               n_wells_total=len(src["wells"]), n_library=int(sum(len(v) for v in lib_fits.values())),
               n_eval_wells=int(df["well_id"].nunique()) if len(df) else 0,
               priors={b: v for b, v in priors.items()}, truth_available=src["truth"], methods=labels, **summary)
    f = path("artifacts_dir") / DATASETS[dataset]["artifact"]
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    if len(df):
        df.to_csv(f.with_name(f.stem + "_rows.csv"), index=False)     # 逐井回测明细（本地产物，概率汇总据此估误差分布）
    out["report_path"] = str(f)
    return out
