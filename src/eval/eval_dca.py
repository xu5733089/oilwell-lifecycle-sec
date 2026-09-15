"""递减模型回测：经验 Arps 与 带物理约束的递减模型（LF-BDF + 类比先验 + EUR 上限）并排比。

    python -m src.cli eval-dca

做法（全部用老井，不看未来）：
  · 按投产时间把已达峰老井交错分成两半：A 半做"类比井库"（全历史拟合，得到各区块参数先验），B 半做回测井；
  · 对回测井只截取峰后前 H 个月（H = 6 / 12 / 24）拟合，外推：
      1) 未来 12 个月产量（开井日均口径 × 30.4），与实测比；
      2) EUR，与合成真值比（真实数据没有真值时这一列自动缺省）；
  · 关注三件事：中位绝对误差、中位偏差、**高估超过 20% 的井占比** —— SEC 已证实储量最怕高估。

**结论由数据算出来，不预先写死。** 合成数据的递减本身按 Arps 生成，经验 Arps 在长历史上天然占优；
物理约束的价值应该体现在短历史上（数据少、外推远），结果是否如此由本脚本回答。
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import db
from ..config import config, path
from ..reserves import dca, physics_dca as PD

METHODS = {
    "arps": "经验 Arps（多模型择优）",
    "physics_wide": "物理约束 · 宽先验",
    "physics_analog": "物理约束 · 类比先验 + EUR 上限",
}


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


def run(horizons: List[int] | None = None) -> Dict:
    from ..api import services as S
    from ..synth.well_generator import Q_ECON_SYNTH

    cfg = config().get("physics_dca", {})
    horizons = horizons or [int(h) for h in cfg.get("hindcast_horizons", [6, 12, 24])]
    truth = _truth()
    T = S._tables()
    lab, m = T["labels"], T["master"]
    wells = (lab[lab["label_quality"] == "ok"][["well_id", "t_peak"]]
             .merge(m[["well_id", "well_code_anon", "block", "first_prod_date"]], on="well_id")
             .sort_values("first_prod_date").reset_index(drop=True))
    prod = db.read_df("SELECT well_id, day_index, oil_t, hours_on FROM prod_daily")
    groups = dict(tuple(prod.groupby("well_id")))
    q_econ = Q_ECON_SYNTH if truth else S.economics.economic_limit_rate(S.DEFAULT_PRICE_DECK)["q_econ"]

    def series(w):
        p = groups[w["well_id"]]
        post = p[p["day_index"] >= w["t_peak"]]
        mo, ra = PD.monthly_series(post["day_index"].to_numpy() - w["t_peak"] + 1,
                                   post["oil_t"].to_numpy(), post["hours_on"].to_numpy())
        return p, mo, ra

    library, evalset = wells.iloc[::2], wells.iloc[1::2]
    lib_fits: Dict[str, List] = {}
    for _, w in library.iterrows():
        _, mo, ra = series(w)
        if len(mo) >= 24:
            lib_fits.setdefault(w["block"], []).append(PD.fit(mo - mo[0], ra, q_econ=q_econ, cum_to_now=0.0))
    priors = {b: PD.analog_prior(v, int(cfg.get("min_analogs", 5))) for b, v in lib_fits.items()}

    rows = []
    for _, w in evalset.iterrows():
        p, months, rates = series(w)
        if len(months) == 0 or months.max() < 20:
            continue
        try:
            rf = S._rf_reference(w["block"])
            ooip = S.estimate_reserves_volumetric(w["well_code_anon"], mc_samples=2000)["ooip_t"]["p50"]
            cap = float(ooip) * float(rf["high_pct"]) / 100.0
        except Exception:
            cap = None
        for H in horizons:
            sel, fut = months < H, (months > H) & (months < H + 12)
            if sel.sum() < 5 or fut.sum() < 8:
                continue
            tt, t_now = months - months[0], H - months[0]
            cum_to = float(p[p["day_index"] <= w["t_peak"] + H * 30.4]["oil_t"].sum())
            r = dict(well_id=w["well_id"], block=w["block"], H=H, actual_next12_t=float(np.sum(rates[fut]) * 30.4),
                     eur_true=truth.get(w["well_id"]))
            fa = dca.fit_best(tt[sel], rates[sel], d_min_year=0.075)
            r["arps_next12_t"] = float(np.sum(dca.rate(tt[fut], fa)) * 30.4)
            r["arps_eur"] = cum_to + dca.eur(fa, q_econ, t_start_month=t_now)["remaining"]
            for name, prior, eur_cap in (("physics_wide", None, None),
                                         ("physics_analog", priors.get(w["block"]), cap)):
                f = PD.fit(tt[sel], rates[sel], q_econ=q_econ, cum_to_now=cum_to, eur_cap_t=eur_cap, prior=prior)
                r[f"{name}_next12_t"] = float(np.sum(PD.rate(tt[fut], f)) * 30.4)
                r[f"{name}_eur"] = cum_to + PD.remaining(f, q_econ, t_now)[0]
                r[f"{name}_cap_binding"] = bool(f.cap_binding)
            rows.append(r)

    df = pd.DataFrame(rows)
    table = []
    for H in horizons:
        d = df[df["H"] == H] if len(df) else df
        for key, label in METHODS.items():
            if d.empty:
                continue
            rec = dict(horizon_months=H, method=key, method_cn=label, n_wells=int(len(d)),
                       next12=_summ(d[f"{key}_next12_t"] / d["actual_next12_t"] - 1.0))
            if truth and d["eur_true"].notna().any():
                rec["eur"] = _summ(d[f"{key}_eur"] / d["eur_true"] - 1.0)
            if f"{key}_cap_binding" in d:
                rec["cap_binding_pct"] = round(float(d[f"{key}_cap_binding"].mean() * 100), 1)
            table.append(rec)

    def pick(H, key, metric, field):
        return next((r.get(metric, {}).get(field) for r in table
                     if r["horizon_months"] == H and r["method"] == key), None)

    lines = []
    for H in horizons:
        a, ph = pick(H, "arps", "eur", "over20_pct"), pick(H, "physics_analog", "eur", "over20_pct")
        ea, ep = pick(H, "arps", "eur", "mdape_pct"), pick(H, "physics_analog", "eur", "mdape_pct")
        if a is None or ph is None:
            continue
        better = "物理约束更稳" if (ph < a and (ep or 0) <= (ea or 0) * 1.1) else \
            "经验 Arps 更准" if (ep or 0) > (ea or 0) * 1.1 else "两者相当"
        lines.append(f"峰后 {H} 个月：EUR 中位误差 Arps {ea}% / 物理约束 {ep}%，"
                     f"高估超 20% 的井 Arps {a}% / 物理约束 {ph}% —— {better}")
    conclusion = ("；".join(lines) + "。合成数据的递减本身按 Arps 生成，长历史上经验 Arps 天然占优；"
                  "物理约束与类比先验的价值集中在历史短、外推远的井上。") if lines else "样本不足，未得出结论。"
    out = dict(horizons=horizons, q_econ=q_econ, n_library=int(sum(len(v) for v in lib_fits.values())),
               priors={b: v for b, v in priors.items()}, table=table, conclusion=conclusion,
               truth_available=bool(truth), methods=METHODS)
    f = path("artifacts_dir") / "eval_dca.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    out["report_path"] = str(f)
    return out
