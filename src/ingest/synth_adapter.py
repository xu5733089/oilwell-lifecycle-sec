"""合成数据适配器。

三条数据轨道（合成 / 公开 / 院内真实）共用同一套内部表结构，
每条轨道只写一个适配器。真实数据到位那天，新增对应适配器
把源字段映射到同样的 DataFrame 即可，上层零改动 —— 这是方案 §3.1 的抗风险设计。

除井、静态、日产、事件四张基础表外，本适配器还负责写入：
  · sec_unit / unit_well   评估单元层级与单元-井归属（合成轨道按 conf/units.yaml 的区块 × 层系规则）
  · unit_plan_monthly      产量与工作量计划（合成轨道模拟规划部门下达的计划值）
  · unit_location          开发方案部署井位（PUD 入账与五年规则跟踪）
  · unit_asset_book        单元资产账面价值（产量法折耗与减值测试）
真实轨道从储量单元台账、计划系统、开发方案与财务台账导入同样几张表（见系统 · 数据导入）。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .. import db
from ..config import config, path, units
from ..reserves.workload import identify_new_wells, period_bounds
from ..synth.well_generator import generate

TABLES = ("well_master", "geo_static", "prod_daily", "well_event")
DERIVED = ("sec_unit", "unit_well", "unit_plan_monthly", "unit_location", "unit_asset_book")


def load(n_wells: int | None = None, seed: int | None = None) -> dict:
    cfg = config()["synth"]
    return generate(
        n_wells=n_wells or cfg["n_wells"],
        seed=seed if seed is not None else cfg["seed"],
        blocks=cfg["blocks"],
        layers=cfg["layers"],
        data_end=cfg.get("data_end", "2026-12-31"),
        recent_frac=cfg.get("recent_frac", 0.10),
        extension_frac=cfg.get("extension_frac", 0.25),
        reactivation_frac=cfg.get("reactivation_frac", 0.0),
    )


def unit_tables(master: pd.DataFrame, conf: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按区块 × 层系把井归入 SEC 单元。规则外的井不归入任何单元（如实计数，不硬塞）。"""
    rows, rule = [], {}
    comp = conf["company"]
    for plant in conf["plants"]:
        for u in plant["units"]:
            rows.append(dict(unit_id=u["id"], unit_name=u["name"],
                             plant_id=plant["id"], plant_name=plant["name"],
                             company_id=comp["id"], company_name=comp["name"],
                             area_type=u.get("area_type", "老区"),
                             opex_factor=float(u.get("opex_factor", 1.0)),
                             data_source="SYNTHETIC"))
            rule[(u["block"], u["layer"])] = u["id"]
    pairs = [(rule[(b, l)], wid) for wid, b, l in
             master[["well_id", "block", "layer"]].itertuples(index=False) if (b, l) in rule]
    return pd.DataFrame(rows), pd.DataFrame(pairs, columns=["unit_id", "well_id"])


def plan_table(data: Dict, unit_well: pd.DataFrame, years, seed: int) -> pd.DataFrame:
    """模拟规划部门下达的月度计划：在当年实际构成上加计划偏差。

    计划值是合成轨道模拟出来的业务输入。生成时用到措施增油真值只是为了让计划"像样"；
    内核与服务只读 unit_plan_monthly 表，从不读真值文件。
    """
    years = {int(y) for y in years}
    prod = data["prod_daily"][["well_id", "dt", "oil_t"]].copy()
    prod["ym"] = prod["dt"].str[:7]
    prod = prod[prod["ym"].str[:4].astype(int).isin(years)]
    mon = prod.groupby(["well_id", "ym"], as_index=False)["oil_t"].sum()
    first = data["well_master"][["well_id", "first_prod_date"]].copy()
    first["fp_ym"] = first["first_prod_date"].str[:7]
    mon = mon.merge(first[["well_id", "fp_ym"]], on="well_id").merge(unit_well, on="well_id")
    mon["is_new"] = mon["fp_ym"].str[:4] == mon["ym"].str[:4]

    ev = data["well_event"][["well_id", "dt"]].copy()
    ev["ev_ym"] = ev["dt"].str[:7]
    inc = data["measure_inc_monthly"].merge(ev[["well_id", "ev_ym"]], on="well_id")
    inc = inc[(inc["ev_ym"].str[:4] == inc["ym"].str[:4]) & inc["ym"].str[:4].astype(int).isin(years)]
    inc = inc.merge(unit_well, on="well_id")

    new_wells = (first.merge(unit_well, on="well_id").groupby(["unit_id", "fp_ym"]).size()
                 .rename("n_new").reset_index().rename(columns={"fp_ym": "ym"}))
    n_meas = (ev.merge(unit_well, on="well_id").groupby(["unit_id", "ev_ym"]).size()
              .rename("n_meas").reset_index().rename(columns={"ev_ym": "ym"}))
    agg = (mon.groupby(["unit_id", "ym"])
           .apply(lambda g: pd.Series(dict(total=g["oil_t"].sum(),
                                           new_oil=g.loc[g["is_new"], "oil_t"].sum())),
                  include_groups=False)
           .reset_index())
    agg = (agg.merge(inc.groupby(["unit_id", "ym"])["inc_oil_t"].sum().rename("meas_inc").reset_index(),
                     on=["unit_id", "ym"], how="left")
              .merge(new_wells, on=["unit_id", "ym"], how="left")
              .merge(n_meas, on=["unit_id", "ym"], how="left")
              .fillna(0.0))

    rng = np.random.default_rng([seed, 4242])
    n = len(agg)
    old = np.maximum(agg["total"] - agg["new_oil"] - agg["meas_inc"], 0.0)
    return pd.DataFrame(dict(
        unit_id=agg["unit_id"], ym=agg["ym"],
        plan_new_wells=np.maximum(np.round(agg["n_new"] + rng.normal(0, 0.8, n)), 0).astype(int),
        plan_new_oil_t=np.round(agg["new_oil"] * rng.lognormal(-0.04, 0.12, n), 1),
        plan_measure_wells=np.maximum(np.round(agg["n_meas"] + rng.normal(0, 0.6, n)), 0).astype(int),
        plan_measure_inc_t=np.round(agg["meas_inc"] * rng.lognormal(0.0, 0.18, n), 1),
        plan_old_oil_t=np.round(old * rng.lognormal(0.02, 0.04, n), 1),
        data_source="SYNTHETIC"))


def location_table(data: Dict, unit_well: pd.DataFrame, dates, su_cfg: Dict, per_unit: int,
                   seed: int) -> pd.DataFrame:
    """模拟开发方案的部署井位。

    · 已钻井位：各评估期约七成提采新井当初是作为 PUD 入账的井位（上一年末入账）；
    · 未钻井位：挨着老井 450~800 m 布设，计划 2027~2030 年钻，入账日期为两个评估期之一；
    · 每个单元再放三个"反例"：入账超五年仍未钻、远离开发区（连续性不足）、计划钻井晚于五年期限。
    """
    rng = np.random.default_rng([seed, 5151])
    dates = sorted(str(d) for d in dates)
    master = data["well_master"]
    mw = master.merge(unit_well, on="well_id")
    rows, seq = [], {}

    def add(unit_id: str, **kw) -> None:
        seq[unit_id] = seq.get(unit_id, 0) + 1
        rows.append({**dict(location_id=f"LOC-{unit_id.replace('SEC_', '')}-{seq[unit_id]:03d}", unit_id=unit_id,
                            status="planned", drilled_well_id=None, first_booked_as_of=None, capex_wan=None,
                            data_source="SYNTHETIC"), **kw})

    for d in dates:
        nw = identify_new_wells(master, d, su_cfg["period_months"], su_cfg["new_well"]["radius_m"],
                                su_cfg["new_well"]["min_old_neighbors"])
        infill = nw[nw["category"] == "infill"].merge(mw[["well_id", "unit_id", "x_off", "y_off"]], on="well_id")
        booked = f"{int(d[:4]) - 1}-12-31"
        for r in infill.sort_values("well_id").itertuples(index=False):
            if rng.random() < 0.7:
                fp = pd.Timestamp(r.first_prod_date) - pd.DateOffset(months=int(rng.integers(1, 4)))
                add(r.unit_id, x_off=float(r.x_off), y_off=float(r.y_off), planned_drill_ym=fp.strftime("%Y-%m"),
                    first_booked_as_of=booked, drilled_well_id=r.well_id, status="drilled")

    cutoff = f"{dates[-1][:4]}-01-01"
    for unit_id, g in mw.sort_values("well_id").groupby("unit_id"):
        olds = g[g["first_prod_date"] < cutoff]
        if olds.empty:
            continue
        cx, cy = float(olds["x_off"].mean()), float(olds["y_off"].mean())
        for _ in range(per_unit):
            w = olds.iloc[int(rng.integers(0, len(olds)))]
            ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(450, 800)
            add(unit_id, x_off=float(w["x_off"] + dist * np.cos(ang)), y_off=float(w["y_off"] + dist * np.sin(ang)),
                planned_drill_ym=f"{int(rng.integers(2027, 2031))}-{int(rng.integers(1, 13)):02d}",
                first_booked_as_of=dates[0] if rng.random() < 0.5 else dates[-1])
        w = olds.iloc[int(rng.integers(0, len(olds)))]
        add(unit_id, x_off=float(w["x_off"] + 400), y_off=float(w["y_off"] - 300), planned_drill_ym="2026-06",
            first_booked_as_of="2021-12-31")                                   # 入账超五年仍未钻
        add(unit_id, x_off=cx - 4200, y_off=cy - 4200, planned_drill_ym="2028-09",
            first_booked_as_of=None)                                           # 远离开发区，连续性不足
        w = olds.iloc[int(rng.integers(0, len(olds)))]
        add(unit_id, x_off=float(w["x_off"] - 520), y_off=float(w["y_off"] + 380), planned_drill_ym="2033-06",
            first_booked_as_of=None)                                           # 计划晚于五年期限
    return pd.DataFrame(rows)


def asset_book_table(data: Dict, unit_well: pd.DataFrame, dates, su_cfg: Dict, nbv_factor: float) -> pd.DataFrame:
    """模拟财务台账：首期给期初资产净值（按单井投资与井龄折旧估算），各期给资本化投入（新井投资 + 措施费用）。

    第二期起期初净值留空，由内核按上期期末净值滚动 —— 与真实台账"期初 = 上期期末"的勾稽一致。
    """
    fin = su_cfg["finance"]
    dates = sorted(str(d) for d in dates)
    wc, mc = fin["well_capex_wan"], fin["measure_cost_wan"]
    m = data["well_master"].merge(unit_well, on="well_id")
    m["capex"] = m["well_type"].map(lambda t: wc.get(t, wc.get("default", 0.0))).astype(float)
    ev = data["well_event"].merge(unit_well, on="well_id")
    ev["cost"] = ev["event_type"].map(lambda t: mc.get(t, mc.get("default", 0.0))).astype(float)
    units_all = sorted(unit_well["unit_id"].unique())
    rows = []
    for k, d in enumerate(dates):
        start = period_bounds(d, su_cfg["period_months"])[0] + "-01"
        adds = (m[(m["first_prod_date"] >= start) & (m["first_prod_date"] <= d)].groupby("unit_id")["capex"].sum()
                .add(ev[(ev["dt"] >= start) & (ev["dt"] <= d)].groupby("unit_id")["cost"].sum(), fill_value=0.0))
        nbv = None
        if k == 0:
            old = m[m["first_prod_date"] < start].copy()
            age = (pd.Timestamp(start) - pd.to_datetime(old["first_prod_date"])).dt.days / 365.25
            old["nbv"] = old["capex"] * np.clip(1.0 - age / 8.0, 0.1, 1.0) * float(nbv_factor)
            nbv = old.groupby("unit_id")["nbv"].sum()
        for u in units_all:
            rows.append(dict(unit_id=u, as_of=d,
                             opening_nbv_wan=None if nbv is None else round(float(nbv.get(u, 0.0)), 1),
                             capex_additions_wan=round(float(adds.get(u, 0.0)), 1), data_source="SYNTHETIC"))
    return pd.DataFrame(rows)


def ingest(n_wells: int | None = None, seed: int | None = None, reset: bool = True) -> dict:
    data = load(n_wells, seed)
    db.init_schema()
    if reset:
        with db.connect() as conn:
            for t in TABLES + DERIVED + ("lifecycle_label", "prediction", "model_run",
                                         "sec_eval_record", "sec_eval_snapshot", "import_batch"):
                conn.execute(f"DELETE FROM {t}")
    counts = {t: db.write_df(data[t], t) for t in TABLES}

    cfg = config()["synth"]
    su, uw = unit_tables(data["well_master"], units())
    plan = plan_table(data, uw, cfg.get("plan_years", []),
                      seed if seed is not None else cfg["seed"])
    su_cfg = config()["sec_unit"]
    seed_ = seed if seed is not None else cfg["seed"]
    locs = location_table(data, uw, su_cfg["evaluation_dates"], su_cfg, cfg.get("pud_locations_per_unit", 8), seed_)
    book = asset_book_table(data, uw, su_cfg["evaluation_dates"], su_cfg, cfg.get("asset_nbv_factor", 1.0))
    counts.update(sec_unit=db.write_df(su, "sec_unit"), unit_well=db.write_df(uw, "unit_well"),
                  unit_plan_monthly=db.write_df(plan, "unit_plan_monthly"),
                  unit_location=db.write_df(locs, "unit_location"),
                  unit_asset_book=db.write_df(book, "unit_asset_book"))

    # 真值只落文件、不进业务表 —— 真实数据没有真值，业务代码不许依赖它
    ddir = path("data_dir")
    data["truth"].to_csv(ddir / "synth_truth.csv", index=False)
    data["event_truth"].to_csv(ddir / "synth_event_truth.csv", index=False)
    counts["truth_file"] = str(ddir / "synth_truth.csv")
    return counts


def read_truth() -> pd.DataFrame:
    p = path("data_dir") / "synth_truth.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def read_event_truth() -> pd.DataFrame:
    p = path("data_dir") / "synth_event_truth.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
