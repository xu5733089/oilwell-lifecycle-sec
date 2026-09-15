"""合成数据适配器。

三条数据轨道（合成 / 公开 / 院内真实）共用同一套内部表结构，
每条轨道只写一个适配器。真实数据到位那天，新增对应适配器
把源字段映射到同样的 DataFrame 即可，上层零改动 —— 这是方案 §3.1 的抗风险设计。

除井、静态、日产、事件四张基础表外，本适配器还负责写入：
  · sec_unit / unit_well   评估单元层级与单元-井归属（合成轨道按 conf/units.yaml 的区块 × 层系规则）
  · unit_plan_monthly      产量与工作量计划（合成轨道模拟规划部门下达的计划值）
真实轨道分别从储量单元台账与计划系统导入同样三张表。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .. import db
from ..config import config, path, units
from ..synth.well_generator import generate

TABLES = ("well_master", "geo_static", "prod_daily", "well_event")
DERIVED = ("sec_unit", "unit_well", "unit_plan_monthly")


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


def ingest(n_wells: int | None = None, seed: int | None = None, reset: bool = True) -> dict:
    data = load(n_wells, seed)
    db.init_schema()
    if reset:
        with db.connect() as conn:
            for t in TABLES + DERIVED + ("lifecycle_label", "prediction", "model_run",
                                         "sec_eval_record", "sec_eval_snapshot"):
                conn.execute(f"DELETE FROM {t}")
    counts = {t: db.write_df(data[t], t) for t in TABLES}

    cfg = config()["synth"]
    su, uw = unit_tables(data["well_master"], units())
    plan = plan_table(data, uw, cfg.get("plan_years", []),
                      seed if seed is not None else cfg["seed"])
    counts.update(sec_unit=db.write_df(su, "sec_unit"), unit_well=db.write_df(uw, "unit_well"),
                  unit_plan_monthly=db.write_df(plan, "unit_plan_monthly"))

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
